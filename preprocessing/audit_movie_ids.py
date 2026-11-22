
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = [
    "movies_metadata.csv",
    "credits.csv",
    "keywords.csv"
]

file_ids = {}

for filename in FILES:
    path = DATA_DIR / filename
    ids = []
    invalid = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):
            raw_id = row.get("id", "").strip()

            if raw_id.isdigit():
                ids.append(int(raw_id))
            else:
                invalid.append((row_number, raw_id))

    counts = Counter(ids)
    file_ids[filename] = set(counts)

    duplicates = {
        movie_id: count
        for movie_id, count in counts.items()
        if count > 1
    }

    print(f"\n{'=' * 55}")
    print(f"FILE: {filename}")
    print(f"{'=' * 55}")
    print(f"Valid rows: {len(ids):,}")
    print(f"Unique movie IDs: {len(counts):,}")
    print(f"Duplicate IDs: {len(duplicates):,}")
    print(f"Extra duplicate rows: {sum(c - 1 for c in duplicates.values()):,}")
    print(f"Invalid IDs: {len(invalid):,}")
    print(f"Sample duplicate IDs: {list(duplicates.items())[:10]}")
    print(f"Sample invalid IDs: {invalid[:10]}")

movies = file_ids["movies_metadata.csv"]
credits = file_ids["credits.csv"]
keywords = file_ids["keywords.csv"]

print(f"\n{'=' * 55}")
print("DATASET MATCHING REPORT")
print(f"{'=' * 55}")

print(f"Unique metadata movies: {len(movies):,}")
print(f"Movies with credits: {len(movies & credits):,}")
print(f"Movies with keywords: {len(movies & keywords):,}")
print(f"Movies with both: {len(movies & credits & keywords):,}")

print(f"Metadata movies missing credits: {len(movies - credits):,}")
print(f"Metadata movies missing keywords: {len(movies - keywords):,}")

print(f"Credit IDs not in metadata: {len(credits - movies):,}")
print(f"Keyword IDs not in metadata: {len(keywords - movies):,}")

print("\nAudit completed. No source files were modified.")
