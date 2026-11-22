import json
import math
import re
from dataclasses import dataclass, field
from itertools import islice
from os import getenv
from typing import Literal

from .debug_logger import active, event, fingerprint, mark, readable, timed
from .intent_parser import IntentParser, MovieIntent, normalize_text
from .recommendation_engine import MODEL_NAME as EMBEDDING_MODEL, RecommendationEngine


LANGUAGE_CODES = {
    "arabic": "ar", "cantonese": "cn", "chinese": "zh", "dutch": "nl",
    "english": "en", "french": "fr", "german": "de", "hindi": "hi",
    "italian": "it", "japanese": "ja", "korean": "ko", "mandarin": "zh",
    "portuguese": "pt", "russian": "ru", "spanish": "es", "swedish": "sv",
}

UNSUPPORTED_REQUEST_FILTERS = {
    r"\b(?:over|longer than|at least|more than)\s+\d+(?:\.\d+)?\s*"
    r"(?:minutes?|mins?|hours?|hrs?)\b": "Minimum runtime is unavailable.",
    r"\b(?:dubbed|subtitled|subtitles)\s+(?:in|into)\b":
        "Dubbed audio and subtitle availability are unavailable.",
    r"\b(?:under|below|less than|at most)\s+\d+(?:\.\d+)?\s*(?:/\s*10|out of 10)\b":
        "Maximum audience rating is unavailable.",
}


# Explicit role words. Without them, "Tom Hanks movies" uses the role with more catalog films.
ACTOR_CUE = re.compile(r"\b(?:starring|featuring|acted by|actor|actress|with|stars?)\b", re.I)
DIRECTOR_CUE = re.compile(r"\b(?:directed by|director|direction|filmmaker)\b", re.I)
THEME_STOPWORDS = {"anything", "something", "movies", "movie", "films", "film", "scenes", "scene", "content",
                   "too", "much", "a", "an", "the", "any", "lots", "of", "please", "really"}
# A popular film wins an exact-title tie when it has this many times the votes of every other match.
DOMINANT_VOTE_RATIO = 10
# Semantic results need at least this many votes when enough better-known candidates remain.
MIN_SEMANTIC_VOTES = 20


@dataclass(frozen=True)
class MovieResult:
    tmdb_id: int
    title: str
    release_year: int | None
    genres: list[str]
    rating: float | None
    vote_count: int | None
    overview: str | None
    recommendation_score: float
    keywords: tuple[str, ...] = ()


@dataclass
class MovieSearchResult:
    intent: MovieIntent
    strategy: Literal["none", "structured", "semantic", "similar", "movie_info"] = "none"
    status: Literal["ok", "clarify", "unsupported", "no_results"] = "ok"
    movies: list[MovieResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clarification_question: str | None = None
    reference_title: str | None = None
    reference_keywords: tuple[str, ...] = ()


def _title_and_year(value: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.+?)\s*\((\d{4})\)", value.strip())
    return (match[1], int(match[2])) if match else (value.strip(), None)


def _eligibility(
    intent: MovieIntent, language: str | None, reference_id: int | None,
    excluded_movie_ids: frozenset[int],
):
    clauses = ["m.adult IS NOT 1"]
    parameters = []

    def add(clause, value):
        clauses.append(clause)
        parameters.append(value)

    for clause, value in (
        ("m.release_year >= ?", intent.year_min),
        ("m.release_year <= ?", intent.year_max),
        ("m.runtime BETWEEN 1 AND ?", intent.runtime_max),
        ("m.vote_average >= ?", intent.min_rating),
        ("m.original_language = ? COLLATE NOCASE", language),
        ("m.tmdb_id <> ?", reference_id),
    ):
        if value is not None:
            add(clause, value)

    if excluded_movie_ids:
        add("m.tmdb_id NOT IN (SELECT value FROM json_each(?))", json.dumps(sorted(excluded_movie_ids)))

    for value in intent.exclude_titles:
        title, year = _title_and_year(value)
        match = "(m.title = ? COLLATE NOCASE OR COALESCE(m.original_title, '') = ? COLLATE NOCASE)"
        parameters.extend((title, title))
        if year is not None:
            match += " AND m.release_year IS ?"
            parameters.append(year)
        clauses.append(f"NOT ({match})")

    for table, column, value in (
        ("movie_cast", "actor_name", intent.actor),
        ("movie_directors", "director_name", intent.director),
    ):
        if value:
            add(f"EXISTS (SELECT 1 FROM {table} r WHERE r.movie_id = m.tmdb_id "
                f"AND r.{column} = ? COLLATE NOCASE)", value)

    genre_exists = (
        "EXISTS (SELECT 1 FROM movie_genres g "
        "WHERE g.movie_id = m.tmdb_id AND g.genre = ? COLLATE NOCASE)"
    )
    if intent.genres_include:
        operator = " OR " if intent.genre_match == "any" else " AND "
        clauses.append("(" + operator.join([genre_exists] * len(intent.genres_include)) + ")")
        parameters.extend(intent.genres_include)
    for genre in intent.genres_exclude:
        add("NOT " + genre_exists, genre)

    return " AND ".join(clauses), parameters


class MovieSearch:
    def __init__(self, parser=None, engine=None):
        self.parser = parser if parser is not None else IntentParser()
        self.engine = engine if engine is not None else RecommendationEngine()
        self._owns_parser = parser is None
        self._owns_engine = engine is None

    def close(self):
        try:
            if self._owns_engine:
                self.engine.close()
        finally:
            if self._owns_parser:
                self.parser.close()

    @timed("sql_retrieval")
    def find_title(self, title):
        title, year = _title_and_year(title)
        return self.engine.conn.execute(
            """
            SELECT tmdb_id, title, release_year, overview, tagline
            FROM movies
            WHERE (title = ? COLLATE NOCASE OR original_title = ? COLLATE NOCASE)
              AND (? IS NULL OR release_year = ?)
            ORDER BY release_year, tmdb_id
            """,
            (title, title, year, year),
        ).fetchall()

    _title_cache = None
    last_title_note = None

    def resolve_title(self, title):
        """Exact match first, then a dominant popular version, then a close spelling match."""
        self.last_title_note = None
        matches = self.find_title(title)
        if len(matches) > 1:
            votes = {row["tmdb_id"]: row["vote_count"] or 0 for row in self.engine.conn.execute(
                "SELECT tmdb_id, vote_count FROM movies WHERE tmdb_id IN (SELECT value FROM json_each(?))",
                (json.dumps([row["tmdb_id"] for row in matches]),),
            )}
            ranked = sorted(matches, key=lambda row: votes[row["tmdb_id"]], reverse=True)
            top, runner_up = votes[ranked[0]["tmdb_id"]], votes[ranked[1]["tmdb_id"]]
            if top >= 500 and top >= DOMINANT_VOTE_RATIO * max(runner_up, 1):
                event("RETRIEVAL", title_resolution="dominant_version", title=title,
                      chosen=ranked[0]["tmdb_id"], votes=top, runner_up_votes=runner_up)
                year = ranked[0]["release_year"] or "year unknown"
                self.last_title_note = (f"I used {ranked[0]['title']} ({year}), the best-known film with that title. "
                                        "Add the release year if you meant another version.")
                return [ranked[0]]
            return matches
        if matches:
            return matches
        name, year = _title_and_year(title)
        if len(name) < 4:
            return []
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return []
        if MovieSearch._title_cache is None:
            rows = self.engine.conn.execute(
                "SELECT title, MAX(vote_count) FROM movies WHERE vote_count >= 50 GROUP BY title COLLATE NOCASE"
            ).fetchall()
            MovieSearch._title_cache = {row[0]: row[1] or 0 for row in rows}
        titles = list(MovieSearch._title_cache)
        found = process.extract(name, titles, scorer=fuzz.ratio, processor=str.casefold, limit=5, score_cutoff=80)
        # Spelling similarity first, with a small bonus for well-known films to break near-ties.
        found = sorted(
            ((match, score + 1.25 * math.log10(max(MovieSearch._title_cache[match], 1)), index)
             for match, score, index in found),
            key=lambda item: item[1], reverse=True,
        )
        if found and (len(found) == 1 or found[0][1] - found[1][1] >= 3):
            corrected = found[0][0] + (f" ({year})" if year else "")
            event("RETRIEVAL", title_resolution="fuzzy", requested=title, corrected=found[0][0], score=found[0][1])
            if corrected.casefold() == title.casefold():
                return []
            rows = self.resolve_title(corrected)
            if rows:
                self.last_title_note = f"I couldn't find {name!r}, so I used the closest title: {found[0][0]}."
            return rows
        return []

    def _find_person(self, request, intent):
        """Find a catalog person written in lowercase, e.g. "christopher nolan movies"."""
        if intent.actor or intent.director or intent.similar_to:
            return intent
        if intent.movie_title:
            if self.find_title(intent.movie_title):
                return intent
            # The model sometimes labels a person's name as a title ("denzel washington movies").
            person = self._find_person(intent.movie_title, MovieIntent())
            role = "actor" if person.actor else "director" if person.director else None
            if not role:
                return intent
            event("RETRIEVAL", title_was_person=intent.movie_title, role=role)
            return intent.model_copy(update={
                "movie_title": None, role: getattr(person, role),
                "intent": "recommend" if intent.intent == "movie_info" else intent.intent,
            })
        words = re.findall(r"[a-z][a-z'.-]+", request.casefold())
        skip = {"movies", "movie", "films", "film", "with", "by", "starring", "like", "and", "the", "some",
                "good", "best", "top", "recommend", "suggest", "show", "me", "about", "from", "directed"}
        for size in (3, 2):
            for start in range(len(words) - size + 1):
                gram = words[start:start + size]
                if any(word in skip for word in gram):
                    continue
                name = " ".join(gram)
                acted = self.engine.conn.execute(
                    "SELECT COUNT(*) FROM movie_cast WHERE actor_name = ? COLLATE NOCASE", (name,)).fetchone()[0]
                directed = self.engine.conn.execute(
                    "SELECT COUNT(*) FROM movie_directors WHERE director_name = ? COLLATE NOCASE", (name,)
                ).fetchone()[0]
                if max(acted, directed) >= 3:
                    role = "director" if directed > acted or DIRECTOR_CUE.search(request) else "actor"
                    event("RETRIEVAL", person_found_in_text=name, role=role, acted=acted, directed=directed)
                    return intent.model_copy(update={role: name.title()})
        return intent

    def _resolve_people(self, request, intent):
        """Fix actor/director mix-ups using catalog film counts."""
        intent = self._find_person(request, intent)
        updates = {}
        for field, other in (("actor", "director"), ("director", "actor")):
            name = getattr(intent, field)
            if not name or getattr(intent, other) or field in updates or other in updates:
                continue
            counts = {}
            for role, table, column in (("actor", "movie_cast", "actor_name"),
                                        ("director", "movie_directors", "director_name")):
                counts[role] = self.engine.conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} = ? COLLATE NOCASE", (name,)
                ).fetchone()[0]
            if not counts["actor"] and not counts["director"]:
                # Not a person in this catalog (for example "movies with Strong Women"): drop it
                # rather than return nothing.
                updates[field] = None
                event("RETRIEVAL", person_dropped=name, reason="no catalog films")
                continue
            cue = ACTOR_CUE if field == "actor" else DIRECTOR_CUE
            explicit = bool(cue.search(request))
            if counts[field] == 0 and counts[other] > 0 or (
                not explicit and counts[other] > counts[field]
            ):
                updates[field] = None
                updates[other] = name
                event("RETRIEVAL", person_role_corrected=name, from_role=field, to_role=other, counts=counts)
        return intent.model_copy(update=updates) if updates else intent

    def _theme_exclusions(self, intent):
        """Best-effort theme exclusions using catalog keywords and cast names."""
        clauses, parameters, applied = [], [], []
        for theme in intent.themes_avoid:
            name = theme.strip()
            cast = self.engine.conn.execute(
                "SELECT 1 FROM movie_cast WHERE actor_name = ? COLLATE NOCASE LIMIT 1", (name,)
            ).fetchone()
            if cast:
                clauses.append("NOT EXISTS (SELECT 1 FROM movie_cast c WHERE c.movie_id = m.tmdb_id "
                               "AND c.actor_name = ? COLLATE NOCASE)")
                parameters.append(name)
                applied.append(name)
                continue
            words = [word for word in re.findall(r"[a-z]+", name.casefold())
                     if word not in THEME_STOPWORDS and len(word) >= 3]
            if not words:
                continue
            # "violence" -> "violen%", so "violent death" also matches.
            stems = [word[:-2] if len(word) > 6 and word.endswith(("ce", "es")) else word for word in words]
            likes = " OR ".join("k.keyword LIKE ?" for _ in stems)
            clauses.append(f"NOT EXISTS (SELECT 1 FROM movie_keywords k WHERE k.movie_id = m.tmdb_id AND ({likes}))")
            parameters.extend(f"%{stem}%" for stem in stems)
            applied.append(name)
        return clauses, parameters, applied

    def _reference_vector(self, source, result):
        try:
            vector = self.engine.index.reconstruct(source["tmdb_id"]).reshape(1, -1)
            if readable():
                event("RETRIEVAL", vector_source="stored reference embedding", reference_id=source["tmdb_id"])
            return vector
        except RuntimeError:
            movie = self.engine.get_movies([source["tmdb_id"]])[source["tmdb_id"]]
            keywords = self.engine.conn.execute(
                "SELECT keyword FROM movie_keywords WHERE movie_id = ? ORDER BY keyword LIMIT 15",
                (source["tmdb_id"],),
            )
            description = [
                movie["genres"], ", ".join(row[0] for row in keywords),
                source["tagline"], (source["overview"] or "")[:750],
            ]
            if not any(description):
                result.warnings.append("The reference movie has no embedding or descriptive metadata.")
                return None
            result.warnings.append("The reference movie has no stored embedding. Used its SQLite description.")
            if readable():
                event("RETRIEVAL", vector_source="embedded reference description", reference_id=source["tmdb_id"])
            return self.engine.encode_query("\n".join(filter(None, [source["title"], *description])))

    def search(self, request: str) -> MovieSearchResult:
        request = normalize_text(request)
        return self.search_intent(request, self.parser.parse(request))

    def search_intent(
        self, request: str, intent: MovieIntent, *, excluded_movie_ids: frozenset[int] = frozenset()
    ) -> MovieSearchResult:
        request = normalize_text(request)
        result = MovieSearchResult(intent)
        if intent.intent == "unsupported":
            result.status = "unsupported"
            result.warnings.append("This request requires a capability outside the local movie catalog.")
            return result
        if intent.intent in {"clarify", "refine"}:
            result.status = "clarify"
            result.clarification_question = intent.clarification_question or (
                "Can you restate the full request, including the filters you want to keep?"
            )
            return result

        included = set(intent.genres_include)
        excluded = set(intent.genres_exclude)
        conflict = included & excluded if intent.genre_match == "all" else included and included <= excluded
        if conflict or (
            intent.year_min is not None and intent.year_max is not None
            and intent.year_min > intent.year_max
        ):
            result.status = "clarify"
            result.clarification_question = "Which conflicting genre or release-year filters should I keep?"
            return result

        result.warnings.extend(
            message for pattern, message in UNSUPPORTED_REQUEST_FILTERS.items()
            if re.search(pattern, request, re.I)
        )
        if intent.rating_source == "critics":
            result.warnings.append("The local catalog has audience ratings only. Critic ratings cannot be applied.")
        language = None
        if intent.language:
            label = intent.language.strip().casefold()
            language = LANGUAGE_CODES.get(label, label)
            with timed("sql_retrieval"):
                known_language = re.fullmatch(r"[a-z]{2,3}", language) and self.engine.conn.execute(
                    "SELECT 1 FROM movies WHERE original_language = ? LIMIT 1", (language,)
                ).fetchone()
            if not known_language:
                result.warnings.append(f"Unknown original language filter: {intent.language!r}.")
        if result.warnings:
            result.status = "unsupported"
            return result
        intent = self._resolve_people(request, intent)
        result.intent = intent

        source = None
        is_info = intent.intent == "movie_info"
        title = intent.movie_title if is_info else intent.similar_to
        if (is_info or intent.intent == "similar_movies") and not title:
            result.status = "clarify"
            result.clarification_question = "Which movie title did you mean?"
            return result
        if title:
            matches = self.resolve_title(title)
            if len(matches) != 1:
                result.status = "clarify"
                if matches:
                    choices = ", ".join(
                        f"{row['title']} ({row['release_year'] or 'unknown year'}, TMDB {row['tmdb_id']})"
                        for row in matches
                    )
                    result.clarification_question = f"Which movie did you mean? Matches: {choices}."
                else:
                    result.clarification_question = f"I couldn't find {title!r}. What is its exact title?"
                return result
            source = matches[0]
            if self.last_title_note:
                result.warnings.append(self.last_title_note)

        where, parameters = _eligibility(
            intent, language, source["tmdb_id"] if source and not is_info else None,
            frozenset() if is_info else excluded_movie_ids,
        )
        if is_info:
            where += " AND m.tmdb_id = ?"
            parameters.append(source["tmdb_id"])
        else:
            theme_clauses, theme_parameters, applied_themes = self._theme_exclusions(intent)
            if theme_clauses:
                where += " AND " + " AND ".join(theme_clauses)
                parameters.extend(theme_parameters)
            if intent.themes_avoid:
                skipped = [theme for theme in intent.themes_avoid if theme not in applied_themes]
                note = "Theme exclusions use catalog keywords and cast, so they are best effort"
                result.warnings.append(note + (f" (could not apply: {', '.join(skipped)})." if skipped else "."))

        tone = intent.desired_tone if intent.desired_tone != "unspecified" else ""
        semantic_query = " ".join(filter(None, [intent.semantic_query.strip(), tone, *intent.themes]))
        has_filters = any([
            intent.genres_include, intent.genres_exclude, intent.actor, intent.director, intent.year_min,
            intent.year_max, intent.runtime_max, intent.min_rating, language, intent.exclude_titles,
        ])
        if not semantic_query and not has_filters and not source and intent.intent == "recommend" \
                and intent.sort_by == "relevance":
            # Nothing structured to filter on: search by the meaning of the request itself
            # instead of returning the catalog's generic top-rated list.
            semantic_query = request.strip()
        if is_info:
            result.strategy = "movie_info"
        elif source:
            result.strategy = "similar"
        elif intent.sort_by != "relevance" or intent.intent == "top_rated":
            result.strategy = "structured"
        elif semantic_query:
            result.strategy = "semantic"
        else:
            result.strategy = "structured"

        logged = active()
        if logged:
            mark("effective_intent", fingerprint(intent.model_dump(mode="json")))
            mark("sql_query", fingerprint([where, parameters]))
            mark("retrieval_strategy", result.strategy)
            mark("semantic_query", fingerprint(semantic_query))
            event(
                "RETRIEVAL", strategy=result.strategy, sql=True,
                faiss_planned=result.strategy in {"semantic", "similar"},
                effective_constraints=intent.model_dump(mode="json", exclude_defaults=True),
                excluded_movie_ids=sorted(excluded_movie_ids) if not is_info else [],
                semantic_query=semantic_query, sql_filter=where, sql_parameters=parameters,
            )
        with timed("sql_retrieval"):
            eligible = {
                row["tmdb_id"]: row for row in self.engine.conn.execute(
                    "SELECT m.tmdb_id, m.vote_average, m.vote_count, m.release_year "
                    f"FROM movies m WHERE {where}", parameters
                )
            }
        if logged:
            mark("sql_candidates", fingerprint(sorted(eligible)))
            event(
                "RETRIEVAL", sql_eligible=len(eligible),
                count_note="SQL applies predicates together. Per-filter rejection counts and prefilter pool were not queried.",
            )
        if readable():
            event("RETRIEVAL", excluded_ids_still_eligible=sorted(
                excluded_movie_ids & eligible.keys()
            ) if not is_info else [], exclusions_applied=not is_info)
        if not eligible:
            result.status = "no_results"
            result.warnings.append("No catalog movies match all filters.")
            return result

        similarities = None
        if result.strategy in {"semantic", "similar"}:
            vector = self._reference_vector(source, result) if source else self.engine.encode_query(semantic_query)
            if source and vector is not None and (tone or intent.themes):
                # "Like Inception but funnier": blend the reference with the requested tone and themes.
                extra = self.engine.encode_query(" ".join(filter(None, [tone, *intent.themes])))
                vector = 0.75 * vector + 0.25 * extra
                vector = (vector / max(float((vector ** 2).sum()) ** 0.5, 1e-9)).astype("float32")
            if vector is None:
                result.status = "unsupported"
                return result
            index = self.engine.index
            similarities = {}
            if index.ntotal:
                with timed("faiss_search"):
                    scores, ids = index.search(vector, index.ntotal)
                similarities = {
                    int(mid): float(score)
                    for mid, score in zip(ids[0], scores[0]) if int(mid) in eligible
                }
            if logged:
                raw_ids = [int(mid) for mid in ids[0] if mid != -1] if index.ntotal else []
                unique_ids = set(raw_ids)
                mark("query_vector", fingerprint(vector.tobytes()))
                mark("faiss_candidates", fingerprint(sorted(unique_ids)))
                mark("faiss_scores", fingerprint(sorted(similarities.items())))
                event(
                    "RETRIEVAL", faiss=True, index_size=index.ntotal, faiss_returned=len(raw_ids),
                    faiss_unique=len(unique_ids), faiss_remaining=len(similarities),
                    not_sql_eligible=len(unique_ids - eligible.keys()),
                    watched_or_rejected_in_faiss=len(unique_ids & excluded_movie_ids),
                    sql_eligible_without_faiss=len(eligible) - len(similarities),
                )
                if len(eligible) > len(similarities):
                    event("DIAGNOSTIC", warning=True, issue="SQL-eligible movies missing from FAISS results",
                          count=len(eligible) - len(similarities),
                          sample_ids=sorted(eligible.keys() - similarities.keys())[:10])
            if readable():
                flat = vector.ravel()
                top_vector = list(islice(
                    ((int(mid), float(score)) for mid, score in zip(ids[0], scores[0]) if mid != -1), 10
                )) if index.ntotal else []
                event(
                    "RETRIEVAL", embedding_model=EMBEDDING_MODEL,
                    vector_dimensions=len(flat), vector_norm=math.hypot(*flat),
                    vector_preview=flat[:8].tolist() if getenv("CINECHAT_VECTOR_PREVIEW") == "1" else None,
                    search_method=type(index).__name__ + ".search" if index.ntotal else None,
                    candidates_requested=index.ntotal, top_vector=top_vector,
                    top_eligible_vector=list(islice(similarities.items(), 10)),
                )
            if len(similarities) < len(eligible):
                result.warnings.append("Some matching catalog movies have no searchable embedding.")

        relevance = None
        if similarities:
            # Raw MiniLM cosine scores sit in a narrow band; rescale within this pool so relevance
            # is not drowned out by rating. The 50th..max range maps to 0..1.
            values = sorted(similarities.values())
            floor, ceiling = values[len(values) // 2], values[-1]
            spread = max(ceiling - floor, 1e-6)
            relevance = {mid: max(0.0, (score - floor) / spread) for mid, score in similarities.items()}
            known = {mid for mid in similarities if (eligible[mid]["vote_count"] or 0) >= MIN_SEMANTIC_VOTES}
            if len(known) >= max(3 * intent.count, 15):
                relevance = {mid: value for mid, value in relevance.items() if mid in known}

        with timed("ranking"):
            ranked = []
            weights = {}
            candidate_ids = eligible if relevance is None else relevance
            for mid in candidate_ids:
                row = eligible[mid]
                votes = max(0, row["vote_count"] or 0)
                rating = row["vote_average"]
                weighted = self.engine.weighted_rating(rating, votes) if rating is not None else 0.0
                if logged:
                    weights[mid] = weighted
                score = weighted / 10.0
                if relevance is not None:
                    score = self.engine.hybrid_score(relevance[mid], weighted, votes)
                year = row["release_year"]
                if intent.sort_by in {"newest", "oldest"}:
                    order = (year is not None, (year or 0) * (-1 if intent.sort_by == "oldest" else 1))
                elif intent.sort_by == "weighted_rating" or intent.intent == "top_rated":
                    order = (rating is not None, weighted)
                else:
                    order = ()
                ranked.append(((*order, score, votes, -mid), mid, score))

            ordered = sorted(ranked, reverse=True)
            selected = ordered[:intent.count]
        # Use the existing metadata query for all displayed IDs and the diagnostic top ten.
        details_rows = ordered[:max(10, intent.count)] if logged else selected
        details = self.engine.get_movies([mid for _, mid, _ in details_rows])
        if logged:
            mark("ranking", fingerprint(ordered))
            event(
                "RANKING", pool=len(ordered), requested_count=intent.count,
                ordering="Descending sort prefix (if any), score, vote count, then ascending TMDB ID",
                sort_by=intent.sort_by, intent=intent.intent,
                selection_ids=[mid for _, mid, _ in selected],
                filter_note="Hard-filtered candidates never enter the ranking.",
            )
            for position, (key, mid, score) in enumerate(ordered[:10], 1):
                row = details.get(mid, {})
                genre_text = row.get("genres")
                event(
                    "RANKING", position=position, tmdb_id=mid, title=row.get("title"),
                    release_year=eligible[mid]["release_year"], genres=genre_text.split("|") if genre_text else [],
                    rating=eligible[mid]["vote_average"], votes=eligible[mid]["vote_count"],
                    similarity=similarities[mid] if similarities is not None else None,
                    weighted_rating=weights[mid], score=score, sort_key=key,
                    excluded=False, selected=position <= intent.count,
                )
        keywords = {}
        for movie_id, keyword in self.engine.conn.execute(
            "SELECT movie_id, keyword FROM movie_keywords WHERE movie_id IN (SELECT value FROM json_each(?)) "
            "ORDER BY movie_id, keyword",
            (json.dumps([mid for _, mid, _ in selected] + ([source["tmdb_id"]] if source else [])),),
        ):
            keywords.setdefault(movie_id, []).append(keyword)
        if source and not is_info:
            result.reference_title = source["title"]
            result.reference_keywords = tuple(keywords.get(source["tmdb_id"], ()))
        for _, mid, score in selected:
            row = details[mid]
            result.movies.append(MovieResult(
                tmdb_id=mid, title=row["title"], release_year=row["release_year"],
                genres=(row["genres"] or "").split("|") if row["genres"] else [],
                rating=row["vote_average"], vote_count=row["vote_count"],
                overview=row["overview"], recommendation_score=score,
                keywords=tuple(keywords.get(mid, ())),
            ))
        if not result.movies:
            result.status = "no_results"
            result.warnings.append("No searchable movies match all filters.")
        return result
