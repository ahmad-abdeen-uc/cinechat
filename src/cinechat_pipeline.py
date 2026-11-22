import json
from dataclasses import dataclass

from pydantic import ValidationError

from .conversation_state import RESET_REQUEST, ConversationState, plan_turn
from .debug_logger import active, audit_display, compare_ranking, event, request_trace, state_snapshot
from .intent_parser import MovieIntent, normalize_text
from .movie_search import MovieResult, MovieSearch, MovieSearchResult
from .response_generator import OLLAMA_ERRORS, RecommendationPresentation, ResponseGenerator


@dataclass
class CineChatResult:
    response_text: str
    recommended_movie_ids: list[int]
    movies: list[MovieResult]
    intent: MovieIntent | None
    strategy: str
    warnings: list[str]
    fallback_used: bool
    presentation: RecommendationPresentation | None = None
    state: ConversationState | None = None


class CineChatPipeline:
    def __init__(self, movie_search: MovieSearch | None = None, generator: ResponseGenerator | None = None):
        self.movie_search = movie_search if movie_search is not None else MovieSearch()
        self.generator = generator if generator is not None else ResponseGenerator()
        self._owns_search = movie_search is None
        self._owns_generator = generator is None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        try:
            if self._owns_generator:
                self.generator.close()
        finally:
            if self._owns_search:
                self.movie_search.close()

    def run(
        self, message: str, conversation_context: str | None = None, *,
        state: ConversationState | None = None,
    ) -> CineChatResult:
        with request_trace(message):
            if active():
                before = state_snapshot(state)
                event("STATE BEFORE", **before)
            result = self._run(message, conversation_context, state)
            if active():
                audit_display(result.intent, result.movies, result.state)
                event("LLM RESPONSE", displayed_ids=result.recommended_movie_ids,
                      displayed_titles=[movie.title for movie in result.movies],
                      fallback=result.fallback_used, warnings=result.warnings)
                if result.state is not None:
                    event("STATE SAVED", **state_snapshot(result.state))
                compare_ranking(before, result.recommended_movie_ids)
            return result

    def _run(self, message, conversation_context, state):
        message = normalize_text(message)
        def notice(text, warnings=None, fallback=False, *, updated=None):
            event("REQUEST", notice=text, warnings=warnings or [], fallback=fallback)
            return CineChatResult(
                text, [], [], None, "none", warnings or [], fallback,
                state=updated if updated is not None else state,
            )

        if not message.strip():
            return notice("What kind of movie are you looking for?")
        if state is not None and RESET_REQUEST.fullmatch(message):
            event("REQUEST", classification="reset")
            return notice("Conversation reset. What kind of movie would you like?", updated=ConversationState())

        pending = state
        try:
            if state is None:
                event("REQUEST", classification="standalone")
                result = self.movie_search.search(message)
            else:
                parsed = self.movie_search.parser.parse(message)
                turn = plan_turn(message, parsed, state)
                event("REQUEST", classification=turn.kind, follow_up=turn.kind != "new", parser_classification=parsed.intent,
                      clarification=turn.question)
                if turn.question:
                    return notice(turn.question)
                intent = turn.intent
                if turn.kind in {"watched", "rejected"}:
                    ids = turn.reference_ids
                    if turn.reference_title:
                        matches = self.movie_search.find_title(turn.reference_title)
                        if not matches:
                            return notice("I couldn't find that title. What's its exact title and release year?")
                        if len(matches) > 1:
                            choices = ", ".join(
                                f"{row['title']} ({row['release_year'] or 'year unknown'}, TMDB {row['tmdb_id']})"
                                for row in matches
                            )
                            return notice(f"Which version did you mean? {choices}.")
                        ids = (matches[0]["tmdb_id"],)
                    field = "watched_movie_ids" if turn.kind == "watched" else "rejected_movie_ids"
                    event("STATE UPDATE", reference_kind=turn.kind, resolved_ids=list(ids), reference_title=turn.reference_title)
                    pending = state.updated(**{field: getattr(state, field).union(ids)})
                    intent = state.current_intent
                    if intent is None:
                        return notice("I'll leave that out of future recommendations. What would you like to watch?",
                                      updated=pending)
                if intent is None:
                    return notice("What kind of movie would you like?")
                if turn.kind in {"new", "refine"} and intent.intent not in {"clarify", "unsupported"}:
                    previous_titles = state.current_intent.exclude_titles if turn.kind == "refine" else []
                    rejected_ids = {
                        row["tmdb_id"] for title in intent.exclude_titles if title not in previous_titles
                        for row in self.movie_search.find_title(title)
                    }
                    if rejected_ids:
                        event("STATE UPDATE", reference_kind="excluded_titles", resolved_ids=sorted(rejected_ids))
                        pending = pending.updated(rejected_movie_ids=pending.rejected_movie_ids | rejected_ids)
                if active():
                    if turn.kind == "new":
                        event("STATE UPDATE", operation="replace active intent", retained="watched and rejected IDs")
                    event("STATE AFTER", **state_snapshot(pending, intent))
                result = self.movie_search.search_intent(
                    message, intent, excluded_movie_ids=pending.watched_movie_ids | pending.rejected_movie_ids,
                )
        except OLLAMA_ERRORS:
            warning = "The local Ollama model is unavailable for intent extraction."
            return notice(
                "I couldn't reach the local movie assistant. Please try again once Ollama is available.",
                [warning], True,
            )
        except (ValidationError, json.JSONDecodeError):
            warning = "The local model returned an invalid movie intent."
            return notice(
                "I couldn't read the movie request. Could you rephrase it with the filters you want?",
                [warning], True,
            )

        event("RETRIEVAL", status=result.status, strategy=result.strategy, warnings=result.warnings,
              clarification=result.clarification_question)
        generated = self.generator.generate(message, result, conversation_context)
        ids = tuple(movie.tmdb_id for movie in generated.movies)
        next_state = state
        if state is not None and result.status in {"ok", "no_results"} and result.strategy != "movie_info":
            next_state = pending.updated(
                current_intent=result.intent,
                last_successful_intent=result.intent if ids else pending.last_successful_intent,
                last_recommendation_ids=ids or (() if turn.kind == "new" else pending.last_recommendation_ids),
            )
        return CineChatResult(
            generated.response_text, list(ids),
            generated.movies, result.intent, result.strategy, generated.warnings,
            generated.fallback_used, generated.presentation, next_state,
        )
