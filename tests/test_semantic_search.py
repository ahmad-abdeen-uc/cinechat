
from pathlib import Path
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------
# 1. Project paths
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = ROOT / "data/processed/cinechat.db"
FAISS_PATH = ROOT / "data/vector_store/movies.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Missing database: {DB_PATH}")

if not FAISS_PATH.exists():
    raise FileNotFoundError(f"Missing FAISS index: {FAISS_PATH}")

# -----------------------------------
# 2. Load FAISS and embedding model
# -----------------------------------

print("Loading FAISS index...")

index = faiss.read_index(str(FAISS_PATH))

print(f"FAISS vectors: {index.ntotal:,}")
print(f"Vector dimensions: {index.d}")

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

# Validate expected dimensions
assert index.d == 384, "Unexpected FAISS dimension"

# -----------------------------------
# 3. Connect to SQLite (read only)
# -----------------------------------

conn = sqlite3.connect(
    DB_PATH.resolve().as_uri() + "?mode=ro",
    uri=True
)

conn.row_factory = sqlite3.Row

# -----------------------------------
# 4. Test natural-language queries
# -----------------------------------

queries = [
    "Mind-bending science fiction movies about "
    "dreams and reality",

    "Emotional movies about friendship, love, "
    "and finding happiness",

    "Space exploration, astronauts, distant "
    "planets and survival",

    "Funny and heartwarming movies that "
    "can cheer someone up"
]

TOP_K = 5

print("\nGenerating query embeddings...")

query_vectors = model.encode(
    queries,
    normalize_embeddings=True,
    convert_to_numpy=True
)

query_vectors = np.ascontiguousarray(
    query_vectors, dtype=np.float32
)

# -----------------------------------
# 5. Search FAISS
# -----------------------------------

scores, movie_ids = index.search(
    query_vectors,
    TOP_K
)

print("\n========== SEMANTIC SEARCH RESULTS ==========")

try:
    for query, ids, similarities in zip(
        queries, movie_ids, scores
    ):
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print("=" * 70)

        seen = set()

        for rank, (mid, similarity) in enumerate(
            zip(ids, similarities), start=1
        ):
            mid = int(mid)

            if mid == -1:
                continue

            assert mid not in seen, (
                f"Duplicate FAISS movie ID: {mid}"
            )
            seen.add(mid)

            movie = conn.execute("""
                SELECT
                    m.tmdb_id,
                    m.title,
                    m.release_year,
                    m.vote_average,
                    m.overview,
                    (
                        SELECT GROUP_CONCAT(g.genre, ', ')
                        FROM movie_genres g
                        WHERE g.movie_id = m.tmdb_id
                    ) AS genres
                FROM movies m
                WHERE m.tmdb_id = ?
            """, (mid,)).fetchone()

            if movie is None:
                raise ValueError(
                    f"Movie ID {mid} missing from SQLite!"
                )

            print(f"\n{rank}. {movie['title']} "
                  f"({movie['release_year']})")

            print(f"   TMDB ID: {mid}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   Genres: {movie['genres']}")
            print(f"   Rating: {movie['vote_average']}")

            overview = movie["overview"] or "No overview"
            print(f"   Plot: {overview[:180]}...")

        print(f"\nMovies retrieved: {len(seen)}")

    print("\n========== VALIDATION ==========")
    print("FAISS retrieval completed.")
    print("All returned movie IDs exist in SQLite.")
    print("No duplicate IDs were returned per query.")

finally:
    conn.close()
