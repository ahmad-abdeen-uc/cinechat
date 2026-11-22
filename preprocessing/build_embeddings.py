
import os
import sqlite3
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "cinechat.db"
INDEX_DIR = PROJECT_ROOT / "data" / "vector_store"
INDEX_PATH = INDEX_DIR / "movies.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

INDEX_DIR.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

# ------------------------------------
# 1. Connect to the SQLite database
# ------------------------------------

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

movies = conn.execute("""
    SELECT tmdb_id, title, overview, tagline
    FROM movies
    ORDER BY tmdb_id
""").fetchall()

print(f"Movies found in SQLite: {len(movies):,}")

# ------------------------------------
# 2. Load genres and keywords
# ------------------------------------

genres = defaultdict(list)
keywords = defaultdict(list)

for row in conn.execute(
    "SELECT movie_id, genre FROM movie_genres ORDER BY movie_id"
):
    genres[row["movie_id"]].append(row["genre"])

for row in conn.execute(
    "SELECT movie_id, keyword FROM movie_keywords ORDER BY movie_id"
):
    keywords[row["movie_id"]].append(row["keyword"])

conn.close()

# ------------------------------------
# 3. Create semantic text per movie
# ------------------------------------

texts = []
movie_ids = []
skipped = 0

for movie in movies:
    mid = movie["tmdb_id"]

    title = movie["title"] or ""
    overview = movie["overview"] or ""
    tagline = movie["tagline"] or ""

    genre_list = genres[mid]
    keyword_list = keywords[mid]

    # Do not embed a title alone when no
    # descriptive information is available.
    if not (overview or tagline or genre_list or keyword_list):
        skipped += 1
        continue

    parts = [
        f"Movie: {title}",
        f"Genres: {', '.join(genre_list)}",
        f"Keywords: {', '.join(keyword_list[:15])}",
        f"Tagline: {tagline}",
        f"Plot: {overview[:750]}"
    ]

    text = "\n".join(
        part for part in parts
        if part.split(": ", 1)[-1].strip()
    )

    texts.append(text)
    movie_ids.append(mid)

print(f"Movies selected for embeddings: {len(texts):,}")
print(f"Movies skipped (insufficient text): {skipped:,}")

if not texts:
    raise RuntimeError("No movies available for embedding")

if len(movie_ids) != len(set(movie_ids)):
    raise RuntimeError("Duplicate movie IDs found")

# ------------------------------------
# 4. Load embedding model
# ------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

dimension = model.get_sentence_embedding_dimension()

print(f"Model: {MODEL_NAME}")
print(f"Vector dimensions: {dimension}")
print(f"Maximum input tokens: {model.max_seq_length}")

# ------------------------------------
# 5. Build FAISS index
# ------------------------------------

# Normalize embeddings and use inner
# product for cosine similarity search.
base_index = faiss.IndexFlatIP(dimension)

# Allows FAISS to store TMDB IDs directly.
index = faiss.IndexIDMap2(base_index)

print("\nGenerating embeddings...")

for start in range(0, len(texts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(texts))

    batch_texts = texts[start:end]

    vectors = model.encode(
        batch_texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    vectors = np.ascontiguousarray(
        vectors, dtype=np.float32
    )

    ids = np.asarray(
        movie_ids[start:end], dtype=np.int64
    )

    index.add_with_ids(vectors, ids)

    if end % 3200 == 0 or end == len(texts):
        print(f"Processed {end:,}/{len(texts):,} movies")

# ------------------------------------
# 6. Validate and save index
# ------------------------------------

if index.ntotal != len(movie_ids):
    raise RuntimeError("FAISS index count mismatch")

temp_path = INDEX_DIR / "movies.faiss.tmp"

faiss.write_index(index, str(temp_path))
os.replace(temp_path, INDEX_PATH)

print("\n========== FAISS SUMMARY ==========")
print(f"Total indexed movies: {index.ntotal:,}")
print(f"Vector dimensions: {dimension}")
print(f"Index size: {INDEX_PATH.stat().st_size / (1024**2):.2f} MB")
print(f"Saved to: {INDEX_PATH}")

print("\nFAISS indexing completed successfully!")
