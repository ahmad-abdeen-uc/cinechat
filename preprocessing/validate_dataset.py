
import csv
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_movies.csv"
csv.field_size_limit(10_000_000)

stats = Counter()
seen_ids = set()
sample_movies = {}

target_titles = {
    "interstellar",
    "inception",
    "the dark knight",
    "toy story",
    "the matrix"
}

list_columns = [
    "genres",
    "keywords",
    "cast",
    "directors",
    "spoken_languages",
    "production_countries",
    "production_companies"
]

with open(FILE, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        stats["total"] += 1

        movie_id = row["tmdb_id"]

        if movie_id in seen_ids:
            stats["duplicate_ids"] += 1
        seen_ids.add(movie_id)

        if not row["title"].strip():
            stats["missing_titles"] += 1

        if not row["overview"].strip():
            stats["missing_overviews"] += 1

        parsed = {}

        for col in list_columns:
            try:
                value = json.loads(row[col])
                if not isinstance(value, list):
                    raise ValueError("Not a list")
                parsed[col] = value
            except (ValueError, TypeError):
                stats["invalid_json"] += 1
                parsed[col] = []

        if not parsed["genres"]:
            stats["missing_genres"] += 1

        if not parsed["keywords"]:
            stats["missing_keywords"] += 1

        if not parsed["cast"]:
            stats["missing_cast"] += 1

        if not parsed["directors"]:
            stats["missing_directors"] += 1

        if not row["overview"].strip() and not (
            parsed["genres"] or parsed["keywords"]
        ):
            stats["weak_semantic_records"] += 1

        try:
            rating = float(row["vote_average"])
            if not 0 <= rating <= 10:
                stats["invalid_ratings"] += 1
        except ValueError:
            stats["missing_ratings"] += 1

        try:
            year = int(row["release_year"])
            if not 1888 <= year <= 2026:
                stats["unusual_years"] += 1
        except ValueError:
            stats["missing_years"] += 1

        if row["title"].strip().lower() in target_titles:
            sample_movies[row["title"]] = {
                "year": row["release_year"],
                "genres": parsed["genres"],
                "directors": parsed["directors"],
                "cast": parsed["cast"][:3],
                "keywords": parsed["keywords"][:5],
                "rating": row["vote_average"],
                "overview": row["overview"][:150]
            }

print("\n========== DATA VALIDATION ==========")

for name, value in stats.items():
    print(f"{name}: {value:,}")

print(f"\nUnique movie IDs: {len(seen_ids):,}")

print("\n========== SAMPLE MOVIES ==========")

for title, info in sample_movies.items():
    print(f"\nMOVIE: {title}")
    for key, value in info.items():
        print(f"  {key}: {value}")

print("\nValidation completed!")
print("No files were modified.")
