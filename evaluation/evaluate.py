"""Measure CineChat parsing and retrieval accuracy on a fixed set of realistic requests.

Usage (from the project root):
    python evaluation/evaluate.py                    # full pipeline: Qwen via Ollama + rules + search
    python evaluation/evaluate.py --rules-only       # rule layer only, no Ollama needed
    python evaluation/evaluate.py --app-dir ../old   # score another copy of the app (before/after)

Metrics
- Intent field accuracy: share of expected fields (genres, years, people, intent...) parsed correctly.
- Constraint satisfaction: share of returned movies that obey the expected hard filters.
- Relevance@5: share of returned movies whose keywords or overview mention an expected topic term.
- Answer rate: share of requests that should return movies and did.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGUAGE_CODES = {"korean": "ko", "japanese": "ja", "french": "fr", "spanish": "es", "german": "de",
                  "italian": "it", "chinese": "zh", "hindi": "hi", "english": "en"}


def normalize(value):
    if isinstance(value, str):
        return " ".join(value.casefold().replace("’", "'").split())
    if isinstance(value, list):
        return sorted(normalize(item) for item in value)
    return value


def field_value(intent, name):
    value = getattr(intent, name)
    if name == "language" and value:
        value = LANGUAGE_CODES.get(value.casefold(), value.casefold())
    return normalize(value)


def check_movies(conn, movies, expect, case):
    """Return (constraint_ok, constraint_total, relevant, relevance_total)."""
    ok = total = relevant = 0
    for movie in movies:
        mid = movie.tmdb_id
        row = conn.execute("SELECT release_year, runtime, original_language, overview FROM movies WHERE tmdb_id=?",
                           (mid,)).fetchone()
        genres = {g for (g,) in conn.execute("SELECT genre FROM movie_genres WHERE movie_id=?", (mid,))}
        keywords = [k.casefold() for (k,) in conn.execute("SELECT keyword FROM movie_keywords WHERE movie_id=?", (mid,))]
        checks = []
        if "genres_include" in expect:
            wanted = set(expect["genres_include"])
            checks.append(bool(wanted & genres) if expect.get("genre_match") == "any" else wanted <= genres)
        if "genres_exclude" in expect:
            checks.append(not (set(expect["genres_exclude"]) & genres))
        if "year_min" in expect:
            checks.append(row[0] is not None and row[0] >= expect["year_min"])
        if "year_max" in expect:
            checks.append(row[0] is not None and row[0] <= expect["year_max"])
        if "runtime_max" in expect:
            checks.append(row[1] is not None and row[1] <= expect["runtime_max"])
        if "language" in expect:
            checks.append(row[2] == expect["language"])
        for role, table, column in (("actor", "movie_cast", "actor_name"), ("director", "movie_directors", "director_name")):
            if role in expect:
                checks.append(bool(conn.execute(f"SELECT 1 FROM {table} WHERE movie_id=? AND {column}=? COLLATE NOCASE",
                                                (mid, expect[role])).fetchone()))
        if "person" in expect:
            checks.append(bool(conn.execute(
                "SELECT 1 FROM movie_cast WHERE movie_id=? AND actor_name=? COLLATE NOCASE UNION "
                "SELECT 1 FROM movie_directors WHERE movie_id=? AND director_name=? COLLATE NOCASE",
                (mid, expect["person"], mid, expect["person"])).fetchone()))
        for stem in case.get("forbidden_keywords", []):
            checks.append(not any(stem in keyword for keyword in keywords))
        ok += sum(checks)
        total += len(checks)
        if case.get("relevance"):
            text = " ".join(keywords) + " " + (row[3] or "").casefold()
            relevant += any(term in text for term in case["relevance"])
    return ok, total, relevant, len(movies) if case.get("relevance") else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", default=str(HERE.parent))
    parser.add_argument("--rules-only", action="store_true", help="skip Ollama; score the rule layer alone")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cases", default="gold_set.json", help="gold_set.json or holdout_set.json")
    args = parser.parse_args()

    app_dir = Path(args.app_dir).resolve()
    sys.path.insert(0, str(app_dir))
    from src.intent_parser import MovieIntent, _validate_request
    from src.movie_search import MovieSearch

    cases = json.loads((HERE / args.cases).read_text(encoding="utf-8"))
    search = MovieSearch()
    if args.rules_only:
        search.parser.parse = lambda text: _validate_request(MovieIntent(), text)
    conn = sqlite3.connect(f"file:{app_dir / 'data/processed/cinechat.db'}?mode=ro", uri=True)

    rows, field_ok, field_total = [], 0, 0
    c_ok = c_total = rel = rel_total = answered = should_answer = 0
    started = time.perf_counter()
    for case in cases:
        expect = case["expect"]
        t0 = time.perf_counter()
        try:
            result = search.search(case["request"])
            error = None
        except Exception as exc:  # keep scoring the rest
            result, error = None, f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - t0
        intent = result.intent if result else None
        fields = {}
        for name, value in expect.items():
            if name in {"status", "person", "language"} and name != "language":
                continue
            if name == "language":
                got = field_value(intent, "language") if intent else None
            else:
                got = field_value(intent, name) if intent else None
            fields[name] = got == normalize(value)
        if "person" in expect:
            fields["person"] = bool(intent) and expect["person"] in {normalize(intent.actor), normalize(intent.director)}
        if "status" in expect:
            fields["status"] = bool(result) and result.status == expect["status"]
        field_ok += sum(fields.values())
        field_total += len(fields)

        movies = result.movies if result else []
        if expect.get("intent") not in {"unsupported"}:
            should_answer += 1
            answered += bool(movies)
        ok, total, relevant, relevance_total = check_movies(conn, movies, expect, case)
        c_ok += ok
        c_total += total
        rel += relevant
        rel_total += relevance_total
        rows.append({
            "id": case["id"], "request": case["request"], "status": result.status if result else "error",
            "fields": fields, "titles": [f"{m.title} ({m.release_year})" for m in movies],
            "constraint": [ok, total], "relevance": [relevant, relevance_total],
            "warnings": result.warnings if result else [error], "seconds": round(seconds, 2),
        })
        marks = "".join("." if good else "x" for good in fields.values())
        print(f"{case['id']:>2} [{marks:<6}] {rows[-1]['status']:<11} {len(movies)} movies  {case['request'][:60]}")

    summary = {
        "mode": "rules-only" if args.rules_only else "full (LLM + rules)",
        "app_dir": str(app_dir),
        "intent_field_accuracy": round(field_ok / field_total, 3),
        "constraint_satisfaction": round(c_ok / c_total, 3) if c_total else None,
        "relevance_at_5": round(rel / rel_total, 3) if rel_total else None,
        "answer_rate": round(answered / should_answer, 3),
        "requests": len(cases),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    search.close()


if __name__ == "__main__":
    main()
