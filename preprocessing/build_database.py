
import csv
import json
import math
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data" / "processed" / "cleaned_movies.csv"
DATABASE = PROJECT_ROOT / "data" / "processed" / "cinechat.db"
TEMP_DB = DATABASE.with_suffix(".building.db")

csv.field_size_limit(10_000_000)

MOVIE_COLUMNS = [
    "tmdb_id", "title", "original_title", "overview",
    "release_date", "release_year", "runtime",
    "original_language", "spoken_languages",
    "production_countries", "production_companies",
    "collection", "adult", "vote_average", "vote_count",
    "popularity", "budget", "revenue", "status",
    "tagline", "poster_path", "imdb_id"
]

def to_int(value):
    try:
        return int(value) if value != "" else None
    except (ValueError, TypeError):
        return None

def to_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None

def parse_list(value):
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def clean_text(value):
    value = (value or "").strip()
    return value if value else None

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing: {SOURCE}")

# Build in a temporary database first.
# The previous completed database remains safe if building fails.
if TEMP_DB.exists():
    TEMP_DB.unlink()

conn = sqlite3.connect(TEMP_DB)
conn.execute("PRAGMA foreign_keys = ON")

try:
    conn.executescript("""
        CREATE TABLE movies (
            tmdb_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            original_title TEXT,
            overview TEXT,
            release_date TEXT,
            release_year INTEGER,
            runtime INTEGER,
            original_language TEXT,
            spoken_languages TEXT,
            production_countries TEXT,
            production_companies TEXT,
            collection TEXT,
            adult INTEGER,
            vote_average REAL,
            vote_count INTEGER,
            popularity REAL,
            budget INTEGER,
            revenue INTEGER,
            status TEXT,
            tagline TEXT,
            poster_path TEXT,
            imdb_id TEXT
        );

        CREATE TABLE movie_genres (
            movie_id INTEGER NOT NULL,
            genre TEXT NOT NULL COLLATE NOCASE,
            PRIMARY KEY (movie_id, genre),
            FOREIGN KEY (movie_id) REFERENCES movies(tmdb_id)
        );

        CREATE TABLE movie_cast (
            movie_id INTEGER NOT NULL,
            actor_name TEXT NOT NULL COLLATE NOCASE,
            PRIMARY KEY (movie_id, actor_name),
            FOREIGN KEY (movie_id) REFERENCES movies(tmdb_id)
        );

        CREATE TABLE movie_directors (
            movie_id INTEGER NOT NULL,
            director_name TEXT NOT NULL COLLATE NOCASE,
            PRIMARY KEY (movie_id, director_name),
            FOREIGN KEY (movie_id) REFERENCES movies(tmdb_id)
        );

        CREATE TABLE movie_keywords (
            movie_id INTEGER NOT NULL,
            keyword TEXT NOT NULL COLLATE NOCASE,
            PRIMARY KEY (movie_id, keyword),
            FOREIGN KEY (movie_id) REFERENCES movies(tmdb_id)
        );
    """)

    placeholders = ", ".join(["?"] * len(MOVIE_COLUMNS))
    columns = ", ".join(MOVIE_COLUMNS)

    queries = {
        "movies": (
            f"INSERT INTO movies ({columns}) "
            f"VALUES ({placeholders})"
        ),
        "movie_genres": (
            "INSERT OR IGNORE INTO movie_genres VALUES (?, ?)"
        ),
        "movie_cast": (
            "INSERT OR IGNORE INTO movie_cast VALUES (?, ?)"
        ),
        "movie_directors": (
            "INSERT OR IGNORE INTO movie_directors VALUES (?, ?)"
        ),
        "movie_keywords": (
            "INSERT OR IGNORE INTO movie_keywords VALUES (?, ?)"
        )
    }

    batches = {name: [] for name in queries}

    def flush_batches():
        for name, records in batches.items():
            if records:
                conn.executemany(queries[name], records)
                records.clear()

    with open(SOURCE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            movie_id = to_int(row["tmdb_id"])

            if movie_id is None:
                raise ValueError("Invalid movie ID in cleaned CSV")

            adult = row["adult"].strip().lower()
            adult_value = (
                1 if adult == "true"
                else 0 if adult == "false"
                else None
            )

            movie = {
                "tmdb_id": movie_id,
                "title": row["title"],
                "original_title": clean_text(row["original_title"]),
                "overview": clean_text(row["overview"]),
                "release_date": clean_text(row["release_date"]),
                "release_year": to_int(row["release_year"]),
                "runtime": to_int(row["runtime"]),
                "original_language": clean_text(row["original_language"]),
                "spoken_languages": row["spoken_languages"],
                "production_countries": row["production_countries"],
                "production_companies": row["production_companies"],
                "collection": clean_text(row["collection"]),
                "adult": adult_value,
                "vote_average": to_float(row["vote_average"]),
                "vote_count": to_int(row["vote_count"]),
                "popularity": to_float(row["popularity"]),
                "budget": to_int(row["budget"]),
                "revenue": to_int(row["revenue"]),
                "status": clean_text(row["status"]),
                "tagline": clean_text(row["tagline"]),
                "poster_path": clean_text(row["poster_path"]),
                "imdb_id": clean_text(row["imdb_id"])
            }

            batches["movies"].append(
                tuple(movie[col] for col in MOVIE_COLUMNS)
            )

            relationships = {
                "movie_genres": "genres",
                "movie_cast": "cast",
                "movie_directors": "directors",
                "movie_keywords": "keywords"
            }

            for table, csv_column in relationships.items():
                for name in parse_list(row[csv_column]):
                    if isinstance(name, str) and name.strip():
                        batches[table].append(
                            (movie_id, name.strip())
                        )

            # Insert in batches to avoid using excessive RAM.
            if len(batches["movies"]) >= 500:
                flush_batches()

        flush_batches()

    # Create indexes for fast filtering and lookups.
    conn.executescript("""
        CREATE INDEX idx_movies_title
            ON movies(title COLLATE NOCASE);

        CREATE INDEX idx_movies_year
            ON movies(release_year);

        CREATE INDEX idx_movies_rating
            ON movies(vote_average DESC, vote_count DESC);

        CREATE INDEX idx_genres
            ON movie_genres(genre, movie_id);

        CREATE INDEX idx_cast
            ON movie_cast(actor_name, movie_id);

        CREATE INDEX idx_directors
            ON movie_directors(director_name, movie_id);

        CREATE INDEX idx_keywords
            ON movie_keywords(keyword, movie_id);
    """)

    conn.commit()

    print("\n========== DATABASE SUMMARY ==========")

    for table in queries:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count:,} records")

    expected = 45433

    actual = conn.execute(
        "SELECT COUNT(*) FROM movies"
    ).fetchone()[0]

    if actual != expected:
        raise ValueError(
            f"Expected {expected} movies, got {actual}"
        )

    print("\nDatabase integrity:")
    print(conn.execute("PRAGMA integrity_check").fetchone()[0])

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError(f"Foreign key errors: {violations[:5]}")

    print("\nMovies titled 'The Dark Knight':")

    rows = conn.execute("""
        SELECT tmdb_id, title, release_year
        FROM movies
        WHERE title = ? COLLATE NOCASE
        ORDER BY release_year
    """, ("The Dark Knight",)).fetchall()

    for row in rows:
        print(row)

finally:
    conn.close()

# Publish completed database only after successful build.
TEMP_DB.replace(DATABASE)

print(f"\nDatabase successfully created: {DATABASE}")
