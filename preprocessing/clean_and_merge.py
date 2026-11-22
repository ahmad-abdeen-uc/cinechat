
import ast
import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned_movies.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

csv.field_size_limit(10_000_000)

def clean(value):
    text = str(value or "").strip()
    return "" if text.lower() in ("nan", "none", "null") else text

def movie_id(row):
    value = clean(row.get("id"))
    return int(value) if value.isdigit() else None

def parse_nested(value):
    value = clean(value)
    if not value:
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return []

def get_names(value):
    items = parse_nested(value)
    if not isinstance(items, list):
        return []

    result = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        name = clean(item.get("name"))
        if name and name.casefold() not in seen:
            result.append(name)
            seen.add(name.casefold())

    return result

def numeric(value):
    try:
        result = float(clean(value))
        return result if math.isfinite(result) else ""
    except (ValueError, TypeError):
        return ""

def integer(value):
    number = numeric(value)
    return int(number) if number != "" and number >= 0 else ""

def get_year(value):
    match = re.match(r"^(\d{4})(?:-|$)", clean(value))
    return int(match.group(1)) if match else ""

def json_array(items):
    return json.dumps(items, ensure_ascii=False)

def load_rows(filename):
    with open(RAW / filename, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)

# 1. Load metadata and resolve duplicate movie IDs.
movies = {}
invalid_ids = 0
duplicate_metadata = 0

def quality(row):
    fields = ["title", "overview", "genres", "release_date",
              "runtime", "original_language", "vote_count"]
    completeness = sum(bool(clean(row.get(k))) for k in fields)
    return completeness, len(clean(row.get("overview")))

for row in load_rows("movies_metadata.csv"):
    mid = movie_id(row)

    if mid is None:
        invalid_ids += 1
        continue

    if mid in movies:
        duplicate_metadata += 1
        if quality(row) > quality(movies[mid]):
            movies[mid] = row
    else:
        movies[mid] = row

print(f"Unique metadata movies: {len(movies):,}")
print(f"Invalid metadata IDs skipped: {invalid_ids}")
print(f"Duplicate metadata rows resolved: {duplicate_metadata}")

# 2. Load credits, keeping the richest row per movie ID.
credits = {}

for row in load_rows("credits.csv"):
    mid = movie_id(row)
    if mid is None:
        continue

    size = len(row.get("cast") or "") + len(row.get("crew") or "")

    if mid not in credits:
        credits[mid] = row
    else:
        previous = credits[mid]
        previous_size = len(previous.get("cast") or "") + len(previous.get("crew") or "")
        if size > previous_size:
            credits[mid] = row

print(f"Unique credits records: {len(credits):,}")

# 3. Combine keywords from duplicate rows without repeating names.
keywords = {}

for row in load_rows("keywords.csv"):
    mid = movie_id(row)
    if mid is None:
        continue

    bucket = keywords.setdefault(mid, {})
    for name in get_names(row.get("keywords")):
        bucket.setdefault(name.casefold(), name)

print(f"Unique keyword records: {len(keywords):,}")

# 4. Build one clean row per movie.
columns = [
    "tmdb_id", "title", "original_title", "overview",
    "genres", "keywords", "cast", "directors",
    "release_date", "release_year", "runtime",
    "original_language", "spoken_languages",
    "production_countries", "production_companies",
    "collection", "adult", "vote_average", "vote_count",
    "popularity", "budget", "revenue", "status",
    "tagline", "poster_path", "imdb_id"
]

written = 0
missing_overview = 0
missing_credits = 0
missing_keywords = 0
missing_title = 0

with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for mid, movie in sorted(movies.items()):
        title = clean(movie.get("title")) or clean(movie.get("original_title"))

        if not title:
            missing_title += 1
            continue

        credit = credits.get(mid, {})
        cast = get_names(credit.get("cast"))
        crew = parse_nested(credit.get("crew"))

        directors = []
        if isinstance(crew, list):
            directors = list(dict.fromkeys(
                clean(person.get("name"))
                for person in crew
                if isinstance(person, dict)
                and person.get("job") == "Director"
                and clean(person.get("name"))
            ))

        movie_keywords = list(keywords.get(mid, {}).values())
        overview = clean(movie.get("overview"))

        if not overview:
            missing_overview += 1
        if mid not in credits:
            missing_credits += 1
        if mid not in keywords:
            missing_keywords += 1

        collection = parse_nested(movie.get("belongs_to_collection"))
        collection_name = clean(collection.get("name")) if isinstance(collection, dict) else ""

        release_date = clean(movie.get("release_date"))
        adult_value = clean(movie.get("adult")).lower()
        if adult_value not in ("true", "false"):
            adult_value = ""

        record = {
            "tmdb_id": mid,
            "title": title,
            "original_title": clean(movie.get("original_title")),
            "overview": overview,
            "genres": json_array(get_names(movie.get("genres"))),
            "keywords": json_array(movie_keywords),
            "cast": json_array(cast),
            "directors": json_array(directors),
            "release_date": release_date,
            "release_year": get_year(release_date),
            "runtime": integer(movie.get("runtime")),
            "original_language": clean(movie.get("original_language")),
            "spoken_languages": json_array(get_names(movie.get("spoken_languages"))),
            "production_countries": json_array(get_names(movie.get("production_countries"))),
            "production_companies": json_array(get_names(movie.get("production_companies"))),
            "collection": collection_name,
            "adult": adult_value,
            "vote_average": numeric(movie.get("vote_average")),
            "vote_count": integer(movie.get("vote_count")),
            "popularity": numeric(movie.get("popularity")),
            "budget": integer(movie.get("budget")),
            "revenue": integer(movie.get("revenue")),
            "status": clean(movie.get("status")),
            "tagline": clean(movie.get("tagline")),
            "poster_path": clean(movie.get("poster_path")),
            "imdb_id": clean(movie.get("imdb_id"))
        }

        writer.writerow(record)
        written += 1

print("\n========== CLEANING SUMMARY ==========")
print(f"Movies written: {written:,}")
print(f"Movies without a usable title: {missing_title:,}")
print(f"Movies without an overview: {missing_overview:,}")
print(f"Movies without credits: {missing_credits:,}")
print(f"Movies without keywords: {missing_keywords:,}")
print(f"\nOutput: {OUTPUT}")
print("Original CSV files were not modified.")
