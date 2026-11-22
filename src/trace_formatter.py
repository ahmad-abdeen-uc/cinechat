"""Render the diagnostic events collected for one request as terminal text."""

import re


RULE = "=" * 66
DIVIDER = "-" * 66


def _text(value, limit=220):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item, limit) for item in value) or "none"
    text = " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(value)).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _ms(value):
    if value is None:
        return "N/A"
    return f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.2f}ms"


def _score(value):
    return f"{value:.4f}" if value is not None else "N/A"


def _section(lines, title):
    lines.extend(("", DIVIDER, title, DIVIDER))


def _fields(lines, pairs):
    lines.extend(f"  {title}: {_text(value)}" for title, value in pairs)


def _table(lines, headings, widths, rows):
    def row(values):
        return "  ".join(_text(value, width).ljust(width) for value, width in zip(values, widths)).rstrip()
    lines.append(row(headings))
    lines.append(row(["-" * width for width in widths]))
    lines.extend(row(values) for values in rows)


def _joined(events):
    combined = {}
    for item in events:
        combined.update(item)
    return combined


def _years(prefs):
    low, high = prefs.get("year_min"), prefs.get("year_max")
    if low is not None and high is not None:
        return f"{low}-{high}"
    if low is not None:
        return f"{low}+"
    if high is not None:
        return f"through {high}"
    return "any"


def _genres(prefs):
    genres = prefs.get("genres_include", [])
    return (" OR " if prefs.get("genre_match") == "any" else " + ").join(genres) or "any"


def _preferences(prefs):
    if not prefs:
        return "none"
    parts = [f"Genres: {_genres(prefs)}", f"Excluded: {_text(prefs.get('genres_exclude', []))}", f"Year: {_years(prefs)}"]
    names = (
        ("desired_tone", "Tone"), ("user_mood", "Mood"), ("runtime_max", "Runtime max"),
        ("actor", "Actor"), ("director", "Director"), ("language", "Language"),
        ("min_rating", "Rating min"), ("rating_source", "Rating source"), ("similar_to", "Similar to"),
        ("themes", "Themes"), ("themes_avoid", "Excluded themes"), ("exclude_titles", "Excluded titles"),
        ("sort_by", "Sort"), ("count", "Count"),
    )
    for name, label in names:
        value = prefs.get(name)
        if value not in (None, "unspecified", [], "relevance"):
            parts.append(f"{label}: {_text(value, 80)}")
    return " | ".join(parts)


def _intent_fields(prefs):
    return [
        ("Classification", prefs.get("intent")), ("Mood", prefs.get("user_mood")),
        ("Desired tone", prefs.get("desired_tone")), ("Included genres", _genres(prefs)),
        ("Excluded genres", prefs.get("genres_exclude", [])), ("Years", _years(prefs)),
        ("Runtime max (min)", prefs.get("runtime_max")), ("Actor", prefs.get("actor")),
        ("Director", prefs.get("director")), ("Similar to", prefs.get("similar_to")),
        ("Semantic query", prefs.get("semantic_query") or "(empty)"),
    ]


def _tokens(tokens):
    return " / ".join(f"{name} {count:,}" for name, count in tokens.items()) or "N/A"


def _intent(lines, data, perf, message):
    _section(lines, "LLM CALL #1 - INTENT PARSING")
    _fields(lines, [
        ("Model", data.get("model")), ("Purpose", "Intent extraction"),
        ("Inference", _ms(perf.get("ollama_intent_ms"))),
        ("Tokens", _tokens(data.get("tokens", {}))), ("Input", message),
    ])
    if "raw_proposed" in data:
        fields = zip(_intent_fields(data["raw_proposed"]), _intent_fields(data["extracted"]))
        _table(lines, ("Field", "LLM proposed", "Python validated"), (20, 30, 38), (
            (label, proposed, f"{_text(validated, 28)} [changed]" if proposed != validated else validated)
            for (label, proposed), (_, validated) in fields
        ))
    elif data.get("model") is None:
        lines.append("  Intent inference was skipped.")
    if data.get("clarification"):
        _fields(lines, [("Clarification", data["clarification"])])
    changes = data.get("changed_by_validation", {})
    if changes:
        _fields(lines, [("Changed fields", list(changes))])


def _memory(lines, sections, retrieval, titles, message):
    _section(lines, "CONVERSATION MEMORY")
    before = _joined(sections.get("STATE BEFORE", []))
    after = _joined(sections.get("STATE AFTER", []))
    _fields(lines, [
        ("Previous state", _preferences(before.get("preferences"))),
        ("New request", message),
        ("Effective state", _preferences(after.get("preferences") or retrieval.get("effective_constraints"))),
    ])
    for event in sections.get("STATE UPDATE", []):
        if "reference_kind" in event:
            for mid in event["resolved_ids"]:
                if "sql_eligible" not in retrieval or not retrieval.get("exclusions_applied"):
                    status = "exclusion unverified (no applicable SQL result)"
                elif mid in retrieval.get("excluded_ids_still_eligible", []):
                    status = "still eligible"
                elif mid in retrieval.get("excluded_movie_ids", []):
                    status = "excluded by SQL"
                else:
                    status = "no ID exclusion applied"
                name = f" | {titles[mid]}" if mid in titles else ""
                _fields(lines, [(event["reference_kind"].replace("_", " ").capitalize(), f"TMDB {mid}{name} | {status}")])
        for kind in ("added", "removed", "replaced", "cleared"):
            changes = event.get(kind)
            if isinstance(changes, dict):
                for key, value in changes.items():
                    if kind == "replaced":
                        value = f"{_text(value['before'])} -> {_text(value['after'])}"
                    _fields(lines, [(f"{kind.capitalize()} {key}", value)])
            elif changes:
                _fields(lines, [(kind.capitalize(), changes)])
    retained = [
        (key, label) for key, label in (
            ("genres_include", "included genres"), ("genres_exclude", "excluded genres"),
            ("year_min", "earliest year"), ("year_max", "latest year"), ("runtime_max", "runtime"),
            ("actor", "actor"), ("director", "director"),
        )
        if (before.get("preferences") or {}).get(key)
        and (before.get("preferences") or {}).get(key) == (after.get("preferences") or {}).get(key)
    ]
    if retained:
        _fields(lines, [("Retained constraints", [label for _, label in retained])])


def _retrieval(lines, data, perf, titles):
    _section(lines, "SQLITE RETRIEVAL")
    prefs = data.get("effective_constraints", {})
    strategy = data.get("strategy")
    _fields(lines, [
        ("Retrieval strategy", strategy), ("Genres required", _genres(prefs)),
        ("Genres excluded", prefs.get("genres_exclude", [])), ("Years", _years(prefs)),
        ("Actor", prefs.get("actor")), ("Director", prefs.get("director")),
        ("Runtime max (min)", prefs.get("runtime_max")), ("Minimum rating", prefs.get("min_rating")),
        ("Language", prefs.get("language")), ("Excluded titles", prefs.get("exclude_titles", [])),
        ("Excluded TMDB IDs", data.get("excluded_movie_ids", [])), ("Movies satisfying SQL", data.get("sql_eligible")),
    ])
    if data.get("status") != "ok":
        _fields(lines, [("Status", data.get("status")), ("Clarification", data.get("clarification"))])

    _section(lines, "EMBEDDING AND FAISS")
    if not data.get("faiss"):
        reason = "structured SQL request" if strategy in {"structured", "movie_info"} else "vector search was skipped"
        lines.append(f"  FAISS: Not used - {reason}.")
        return
    source = data.get("vector_source", "embedded query text")
    query = data.get("semantic_query") or "(empty)"
    if data.get("reference_id") is not None:
        query = f"{source} for TMDB {data['reference_id']}"
    _fields(lines, [
        ("Embedding model", data.get("embedding_model")), ("Query text or source", query),
        ("Vector dimensions", data.get("vector_dimensions")), ("Vector norm", _score(data.get("vector_norm"))),
        ("Embedding duration", _ms(perf.get("embedding_ms"))),
        ("Search method", data.get("search_method")), ("Indexed vectors", data.get("index_size")),
        ("Candidates requested", data.get("candidates_requested")), ("Candidates returned", data.get("faiss_returned")),
        ("Candidates surviving SQL filters", data.get("faiss_remaining")), ("FAISS search duration", _ms(perf.get("faiss_search_ms"))),
    ])
    if data.get("vector_preview") is not None:
        _fields(lines, [("First 8 dimensions", [_score(value) for value in data["vector_preview"]])])
    lines.append("  FAISS similarity is one input to the final recommendation score.")
    for key, title in (
        ("top_vector", "TOP VECTOR MATCHES BEFORE SQL FILTERING"),
        ("top_eligible_vector", "TOP ELIGIBLE VECTOR MATCHES AFTER SQL FILTERING"),
    ):
        lines.extend(("", title))
        _table(lines, ("Rank", "TMDB ID", "Movie (when available)", "Similarity"), (4, 9, 36, 10), (
            (rank, mid, titles.get(mid, "-"), _score(score))
            for rank, (mid, score) in enumerate(data.get(key, []), 1)
        ))


def _ranking(lines, events):
    _section(lines, "PYTHON RERANKING")
    rows = [event for event in events if "position" in event]
    if not rows:
        lines.append("  Ranking was skipped.")
        return
    method = "Hybrid similarity, weighted audience rating, and vote confidence" if rows[0]["similarity"] is not None else "Weighted audience rating"
    summary = _joined([event for event in events if "pool" in event])
    _fields(lines, [("Scoring method", method), ("Sort preference", summary.get("sort_by")), ("Ranked pool", summary.get("pool"))])
    _table(lines, ("Rank", "Movie", "Year", "Similarity", "Weighted rating", "Final score", "Selected"), (4, 32, 4, 10, 15, 11, 8), (
        (row["position"], row["title"] or f"TMDB {row['tmdb_id']}", row["release_year"], _score(row["similarity"]),
         _score(row["weighted_rating"]), _score(row["score"]), row["selected"]) for row in rows
    ))
    for event in events:
        if event.get("comparison") == "displayed_ids_changed":
            _fields(lines, [("Comparison", "Same request and prior state produced a different list"),
                            ("Changed stages", event["changed_stages"]), ("Previous request ID", event["previous_request_id"])])


def _response(lines, events, prefs, perf, message):
    _section(lines, "LLM CALL #2 - RESPONSE GENERATION")
    data = _joined(events)
    tries = [event for event in events if "attempt" in event]
    last = tries[-1] if tries else {}
    validation = "PASS" if last.get("validation") == "grounded" else "FAIL" if tries else "SKIPPED"
    _fields(lines, [
        ("Model", data.get("model")), ("Purpose", "Grounded response generation"),
        ("User request", message), ("Effective preferences", _preferences(data.get("effective_preferences") or prefs)),
    ])
    if movies := data.get("selected_movies"):
        lines.append("  Selected IDs and titles (SQLite; the model receives IDs and approved wording):")
        for mid, title in movies:
            lines.append(f"    {mid} | {_text(title)}")
    if tries:
        tokens_by_attempt = {event["token_attempt"]: event["tokens"] for event in events if "token_attempt" in event}
        for attempt in tries:
            number = attempt["attempt"]
            _fields(lines, [(f"Attempt {attempt['attempt']}", attempt["validation"]),
                            ("Returned IDs", attempt.get("returned_ids", [])),
                            ("Tokens", _tokens(tokens_by_attempt.get(number, {})))])
    _fields(lines, [
        ("Movies explained", len(last.get("returned_ids", [])) if validation == "PASS" else 0),
        ("Grounding validation", validation), ("Calls", data.get("calls", 0)), ("Retries", data.get("retries", 0)),
        ("Fallback", data.get("fallback", False)), ("Inference", _ms(perf.get("ollama_response_ms"))),
    ])
    return data, validation


def format_trace(request_id, turn, sections):
    request = _joined(sections.get("REQUEST", []))
    intent = _joined(sections.get("INTENT PARSER", []))
    retrieval = _joined(sections.get("RETRIEVAL", []))
    ranking = sections.get("RANKING", [])
    response = sections.get("LLM RESPONSE", [])
    perf = _joined(sections.get("PERFORMANCE", []))
    message = request.get("message")
    titles = {int(mid): title for mid, title in request.get("known_titles", {}).items()}
    titles.update({row["tmdb_id"]: row["title"] for row in ranking if row.get("title")})
    titles.update(dict(_joined(response).get("selected_movies", [])))
    header = f"CINECHAT REQUEST #{turn}" if turn is not None else "CINECHAT REQUEST"
    kind = request.get("classification", "unclassified")
    kind = f"Follow-up ({kind})" if request.get("follow_up") else kind.replace("_", " ").capitalize()
    lines = [RULE, header, RULE, "", f"USER: {_text(message, 1200)}", "",
             f"Conversation type: {_text(kind)}", f"Request ID: {request_id}"]
    _intent(lines, intent, perf, message)
    _memory(lines, sections, retrieval, titles, message)
    _retrieval(lines, retrieval, perf, titles)
    _ranking(lines, ranking)
    data, validation = _response(lines, response, retrieval.get("effective_constraints"), perf, message)

    diagnostics = sections.get("DIAGNOSTIC", [])
    if diagnostics or data.get("warnings") or request.get("notice") or request.get("error"):
        _section(lines, "DIAGNOSTICS")
        for event in diagnostics:
            extra = " | ".join(f"{key}={_text(value)}" for key, value in event.items() if key != "issue")
            lines.append(f"  WARNING: {_text(event.get('issue'))}" + (f" | {extra}" if extra else ""))
        for warning in data.get("warnings", request.get("warnings", [])):
            _fields(lines, [("Notice", warning)])
        for key in ("notice", "error"):
            if request.get(key):
                _fields(lines, [(key.capitalize(), request[key])])

    lines.extend(("", RULE, "REQUEST COMPLETE", RULE))
    _fields(lines, [
        (label, _ms(perf.get(key))) for label, key in (
            ("Intent call", "ollama_intent_ms"), ("State updates", "state_update_ms"),
            ("SQL retrieval", "sql_retrieval_ms"), ("Embedding", "embedding_ms"), ("FAISS search", "faiss_search_ms"),
            ("Ranking", "ranking_ms"), ("Response LLM", "ollama_response_ms"), ("Total", "total_ms"),
        )
    ])
    lines.append("  Final movies (SQLite):")
    lines.extend(f"    {number}. {_text(title)}" for number, title in enumerate(data.get("displayed_titles", []), 1))
    if not data.get("displayed_titles"):
        lines.append("    none")
    _fields(lines, [("Grounding validation", validation), ("Fallback", data.get("fallback", request.get("fallback", False)))])
    return "\n".join(lines)
