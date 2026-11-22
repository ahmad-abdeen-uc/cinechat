
import math
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .debug_logger import active, event, timed
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data/processed/cinechat.db"
INDEX_PATH = ROOT / "data/vector_store/movies.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RecommendationEngine:
    def __init__(self):
        self.conn = sqlite3.connect(
            DB_PATH.resolve().as_uri() + "?mode=ro",
            uri=True
        )
        self.conn.row_factory = sqlite3.Row

        self._index = None
        self._model = None

        row = self.conn.execute("""
            SELECT AVG(vote_average)
            FROM movies
            WHERE vote_average BETWEEN 0 AND 10
              AND vote_count >= 50
        """).fetchone()

        self.global_rating = row[0] or 6.0

    @property
    def index(self):
        if self._index is None:
            with timed("faiss_load"):
                self._index = faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_READ_ONLY)
            event("RETRIEVAL", faiss_index_loaded=True, index_size=self._index.ntotal)
        return self._index

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(
                MODEL_NAME, device="cpu", local_files_only=True
            )
        return self._model

    @timed("embedding")
    def encode_query(self, query):
        vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.ascontiguousarray(vector, dtype=np.float32)

    def hybrid_score(self, similarity, weighted_rating, votes):
        semantic = max(0.0, min(1.0, float(similarity)))
        # Smoothed ratings mostly fall between 5 and 8, so spread that band over 0..1.
        quality = max(0.0, min(1.0, (weighted_rating - 4.5) / 4.0))
        confidence = min(1.0, math.log1p(votes) / math.log1p(10000))
        return 0.60 * semantic + 0.30 * quality + 0.10 * confidence

    @timed("sql_retrieval")
    def get_movies(self, ids):
        results = {}

        # Batch queries to support different SQLite versions.
        for start in range(0, len(ids), 350):
            batch = ids[start:start + 350]
            placeholders = ",".join("?" for _ in batch)

            query = f"""
                SELECT
                    m.tmdb_id,
                    m.title,
                    m.release_year,
                    m.overview,
                    m.vote_average,
                    m.vote_count,
                    m.adult,
                    GROUP_CONCAT(g.genre, '|') AS genres
                FROM movies m
                LEFT JOIN movie_genres g
                    ON m.tmdb_id = g.movie_id
                WHERE m.tmdb_id IN ({placeholders})
                GROUP BY m.tmdb_id
            """

            for row in self.conn.execute(query, batch):
                results[row["tmdb_id"]] = dict(row)

        if active():
            if missing := set(ids) - results.keys():
                event("DIAGNOSTIC", warning=True, issue="requested metadata missing from SQLite",
                      missing_count=len(missing), sample_ids=sorted(missing)[:10])
        return results

    def weighted_rating(self, rating, votes):
        # Bayesian smoothing prevents movies with
        # one or two votes from dominating results.
        minimum_votes = 250

        return (
            votes * rating +
            minimum_votes * self.global_rating
        ) / (votes + minimum_votes)

    def recommend(
        self,
        query,
        top_k=5,
        candidate_count=800,
        include_genres=None,
        exclude_genres=None,
        min_votes=25
    ):
        include_genres = {
            g.casefold() for g in (include_genres or [])
        }
        exclude_genres = {
            g.casefold() for g in (exclude_genres or [])
        }

        vector = self.encode_query(query)

        similarities, ids = self.index.search(
            vector, candidate_count
        )

        valid_ids = [
            int(mid) for mid in ids[0] if mid != -1
        ]

        movie_data = self.get_movies(valid_ids)
        results = []

        for mid, similarity in zip(
            ids[0], similarities[0]
        ):
            mid = int(mid)

            if mid == -1 or mid not in movie_data:
                continue

            movie = movie_data[mid]

            # Exclude movies explicitly marked adult.
            if movie["adult"] == 1:
                continue

            genres = {
                g.strip().casefold()
                for g in (movie["genres"] or "").split("|")
                if g.strip()
            }

            # Explicit genre filters are hard constraints.
            if include_genres and not include_genres.issubset(genres):
                continue

            if genres & exclude_genres:
                continue

            votes = movie["vote_count"] or 0
            rating = movie["vote_average"]

            if votes < min_votes:
                continue

            if rating is None:
                rating = self.global_rating

            weighted = self.weighted_rating(
                rating, votes
            )

            score = self.hybrid_score(similarity, weighted, votes)

            movie["similarity"] = float(similarity)
            movie["weighted_rating"] = weighted
            movie["recommendation_score"] = score

            results.append(movie)

        results.sort(
            key=lambda x: x["recommendation_score"],
            reverse=True
        )

        return results[:top_k]

    def close(self):
        self.conn.close()
