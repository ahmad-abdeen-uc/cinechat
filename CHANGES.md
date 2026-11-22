# CineChat accuracy improvements (Residency Day 2)

These changes make the chatbot understand requests more accurately and return more relevant
movies. The architecture is unchanged: Qwen2.5 3B (Ollama) extracts the intent, rules check it,
SQLite applies hard filters, FAISS + MiniLM ranks by meaning, and the reply only uses verified
catalog facts.

## Results

Measured with `evaluation/evaluate.py` and the full pipeline (Qwen2.5 3B + rules + search).
The gold set (30 requests) was written while diagnosing the problems. The hold-out set
(15 requests) was written afterwards with new phrasing, to check that the fixes generalize.

| Metric | Gold: before | Gold: after | Hold-out: before | Hold-out: after |
|---|---|---|---|---|
| Intent field accuracy | 56.1% | 100% | 58.3% | 100% |
| Constraint satisfaction | 71.0% | 100% | 70.0% | 100% |
| Relevance@5 | 70.0% | 86.7% | 60.0% | 96.0% |
| Answer rate | 79.3% | 96.6% | 86.7% | 100% |

Before the last fix (a person's name read as a movie title), the hold-out intent accuracy was
95.8%; that case was fixed after it was seen, so the hold-out figure is slightly optimistic.
The one unanswered gold request ("thrillers after 2018") is a data limit: the catalog ends in 2017.

The original app scored the same intent accuracy with and without the LLM (56.1%), which showed
that the rule layer was overwriting the model's extractions.

## What changed

| # | Problem found | Fix | Files |
|---|---|---|---|
| 1 | Curly quotes typed on macOS flipped meanings: "I don’t want horror" *included* Horror. The quote-normalizing code had been saved with corrupted characters and replaced every `?` instead. | `normalize_text()` maps typographic quotes and dashes to ASCII at every entry point; corrupted regex characters repaired. | `intent_parser.py`, `cinechat_pipeline.py`, `movie_search.py` |
| 2 | The rule layer erased names the LLM found ("movies with Leonardo DiCaprio", lowercase names). | Grounded LLM names are kept when the patterns miss them; "with X" and "X westerns/comedies" patterns added; names in lowercase are found by looking them up in the cast and director tables. | `intent_parser.py`, `movie_search.py` |
| 3 | "Tom Hanks movies" searched for films he *directed* (3 instead of 71). | Actor/director role resolved from catalog film counts unless the user used an explicit role word ("directed by", "starring"). | `movie_search.py` |
| 4 | Any theme exclusion ("without violence", "no gore") refused the whole request. | Theme exclusions became soft filters on catalog keywords (and on cast, for "without Adam Sandler"), with a "best effort" note. | `movie_search.py` |
| 5 | Decades and comparatives were ignored ("90s thriller", "late 80s", "like Inception but funnier"). | Decade parsing (early/mid/late), comparative tones (funnier, darker...), and blending the reference film with the requested tone/themes. | `intent_parser.py`, `movie_search.py` |
| 6 | Reference titles needed exact spelling; "Titanic" always asked which version. | Fuzzy title matching (rapidfuzz) and a "dominant version" rule, both with a visible note. State-changing commands ("I watched Titanic") still ask, as the team designed. | `movie_search.py` |
| 7 | The small LLM put topics into the genre list ("Robots"), where they were silently dropped; topic-only requests fell back to the generic top-rated list. | Non-genre "genres" are kept as themes; "about X" is captured by rule; requests with no filters search by meaning of the request text. | `intent_parser.py`, `movie_search.py` |
| 8 | Ranking favored popular films: raw MiniLM similarities sit in a narrow band and were outweighed by rating. | Similarity is rescaled within each candidate pool; weights now 0.60 relevance / 0.30 quality / 0.10 confidence; very obscure titles (under 20 votes) are dropped when enough known ones remain. | `movie_search.py`, `recommendation_engine.py` |
| 9 | "Why it fits" reasons were generic ("The catalog lists it under Comedy"). | Reasons are still pre-approved and spoiler-free, but now specific: catalog keywords that match the request, keywords shared with the reference film, the requested actor/director, and release year. | `response_generator.py`, `movie_search.py` |
| 10 | Reply generation timed out on slower machines and fell back to templates. | Timeout configurable via `CINECHAT_RESPONSE_TIMEOUT` (default 60 s). | `response_generator.py` |
| 11 | No way to measure accuracy. | `evaluation/` with gold and hold-out sets, scoring script and saved results. | `evaluation/` |

Housekeeping: `requirements.txt` converted from UTF-16 to UTF-8 and `rapidfuzz` added; README has
macOS setup and evaluation instructions.

## Tests

All 120 unit tests pass with Ollama and MiniLM available. Three tests were updated because the
intended behavior changed (dominant title version, soft theme exclusions), and three new tests
cover role correction, soft theme exclusion and fuzzy titles.
