
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "cinechat.db"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

# Read-only connection: this script cannot modify the database.
conn = sqlite3.connect(
    DB_PATH.resolve().as_uri() + "?mode=ro",
    uri=True
)
conn.row_factory = sqlite3.Row


def run_query(title, query, params=()):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)

    rows = conn.execute(query, params).fetchall()

    if not rows:
        print("No movies found.")
        return

    for row in rows:
        movie = dict(row)
        print(
            f"ID: {movie['tmdb_id']} | "
            f"{movie['title']} ({movie['release_year']}) | "
            f"Rating: {movie['vote_average']} | "
            f"Votes: {movie['vote_count']}"
        )

    print(f"\nResults returned: {len(rows)}")


try:
    # TEST 1: Movies that share the same title
    run_query(
        "TEST 1: Movies titled 'The Dark Knight'",
        """
        SELECT tmdb_id, title, release_year,
               vote_average, vote_count
        FROM movies
        WHERE title = ? COLLATE NOCASE
        ORDER BY release_year
        """,
        ("The Dark Knight",)
    )

    # TEST 2: Christopher Nolan movies
    run_query(
        "TEST 2: Christopher Nolan movies",
        """
        SELECT m.tmdb_id, m.title, m.release_year,
               m.vote_average, m.vote_count
        FROM movies m
        JOIN movie_directors d
          ON m.tmdb_id = d.movie_id
        WHERE d.director_name = ? COLLATE NOCASE
        ORDER BY m.vote_count DESC
        LIMIT 10
        """,
        ("Christopher Nolan",)
    )

    # TEST 3: Highly rated action movies after 2010,
    # excluding horror, with at least 100 votes
    run_query(
        "TEST 3: Action movies after 2010, excluding horror",
        """
        SELECT m.tmdb_id, m.title, m.release_year,
               m.vote_average, m.vote_count
        FROM movies m
        WHERE m.release_year > ?
          AND m.vote_average >= ?
          AND m.vote_count >= ?
          AND EXISTS (
              SELECT 1 FROM movie_genres g
              WHERE g.movie_id = m.tmdb_id
                AND g.genre = 'Action'
          )
          AND NOT EXISTS (
              SELECT 1 FROM movie_genres g
              WHERE g.movie_id = m.tmdb_id
                AND g.genre = 'Horror'
          )
        ORDER BY m.vote_average DESC, m.vote_count DESC
        LIMIT 10
        """,
        (2010, 7.0, 100)
    )

    # TEST 4: Leonardo DiCaprio movies
    run_query(
        "TEST 4: Leonardo DiCaprio movies",
        """
        SELECT m.tmdb_id, m.title, m.release_year,
               m.vote_average, m.vote_count
        FROM movies m
        JOIN movie_cast c
          ON m.tmdb_id = c.movie_id
        WHERE c.actor_name = ? COLLATE NOCASE
        ORDER BY m.vote_count DESC
        LIMIT 10
        """,
        ("Leonardo DiCaprio",)
    )

    # TEST 5: Science fiction movies after 2000
    run_query(
        "TEST 5: Science Fiction movies after 2000",
        """
        SELECT m.tmdb_id, m.title, m.release_year,
               m.vote_average, m.vote_count
        FROM movies m
        JOIN movie_genres g
          ON m.tmdb_id = g.movie_id
        WHERE g.genre = 'Science Fiction'
          AND m.release_year > ?
          AND m.vote_count >= ?
        ORDER BY m.vote_average DESC, m.vote_count DESC
        LIMIT 10
        """,
        (2000, 100)
    )

    print("\nAll database test queries executed successfully!")

finally:
    conn.close()
