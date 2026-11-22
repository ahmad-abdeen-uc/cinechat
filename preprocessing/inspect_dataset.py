
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

files = [
    "movies_metadata.csv",
    "credits.csv",
    "keywords.csv"
]

for filename in files:
    filepath = DATA_DIR / filename

    print(f"\n{'=' * 55}")
    print(f"FILE: {filename}")
    print("=" * 55)

    if not filepath.exists():
        print("ERROR: File not found!")
        continue

    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        print("\nCOLUMNS:")
        print(reader.fieldnames)

        total_rows = 0
        invalid_ids = 0
        missing_overviews = 0

        for row in reader:
            total_rows += 1

            movie_id = row.get("id", "")

            if not movie_id or not movie_id.isdigit():
                invalid_ids += 1

            if filename == "movies_metadata.csv":
                overview = row.get("overview", "")
                if not overview or overview.strip().lower() in ("", "nan"):
                    missing_overviews += 1

        print(f"\nTOTAL ROWS: {total_rows:,}")
        print(f"INVALID IDs: {invalid_ids:,}")

        if filename == "movies_metadata.csv":
            print(f"MISSING OVERVIEWS: {missing_overviews:,}")

print("\nDataset inspection completed!")
