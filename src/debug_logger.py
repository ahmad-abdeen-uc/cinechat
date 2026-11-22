"""Terminal diagnostics for an opt-in CineChat request."""

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from uuid import uuid4

if __package__:
    from .trace_formatter import format_trace
else:
    from trace_formatter import format_trace

LOGGER = logging.getLogger("cinechat.debug")
_LOCK = Lock()
_RECENT = OrderedDict()
_SECRET = re.compile(
    r"""(?ix)(\b(?:bearer\s+|(?:[\w-]*(?:api[ _-]?key|access[ _-]?key|secret|token|password|passwd)\b)
    ["']?\s*(?:[:=]\s*|\bis\s+)))("[^"]*"|'[^']*'|[^\s,;}]+)"""
)
_CREDENTIAL = re.compile(
    r"(?:\b(?:sk-|gh[pousr]_|github_pat_|hf_|xox[baprs]-)[\w-]{8,}"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\beyJ[\w-]+\.[\w-]+\.[\w-]+\b|(?<=://)[^\s/@]+:[^\s/@]+(?=@))"
)
_PHASES = (
    "intent_parsing", "ollama_intent", "state_update", "sql_retrieval",
    "embedding", "faiss_load", "faiss_search", "ranking", "response_generation", "ollama_response",
)


@dataclass
class RequestTrace:
    request_id: str
    turn: int | None
    message_key: str
    elapsed: dict[str, float] = field(default_factory=dict)
    fingerprints: dict[str, object] = field(default_factory=dict)
    sections: dict[str, list[dict]] | None = None


_CURRENT: ContextVar[RequestTrace | None] = ContextVar("cinechat_request", default=None)


def debug_enabled():
    return os.getenv("CINECHAT_DEBUG") == "1"


def trace_enabled():
    return os.getenv("CINECHAT_TRACE") == "1"


def configure_logger():
    with _LOCK:
        LOGGER.propagate = False
        LOGGER.setLevel(logging.DEBUG)
        if not any(getattr(handler, "_cinechat_console", False) for handler in LOGGER.handlers):
            handler = logging.StreamHandler()
            handler._cinechat_console = True
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
            LOGGER.addHandler(handler)
    return LOGGER


def active():
    return _CURRENT.get() is not None


def readable():
    trace = _CURRENT.get()
    return trace is not None and trace.sections is not None


def token_counts(response):
    return {label: count for label, name in (("input", "prompt_eval_count"), ("output", "eval_count"))
            if isinstance(count := getattr(response, name, None), int)}


def _redact(value):
    if isinstance(value, str):
        return _CREDENTIAL.sub("[REDACTED]", _SECRET.sub(r"\1[REDACTED]", value))
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value]
    return value


def event(section, *, warning=False, **fields):
    trace = _CURRENT.get()
    if trace is not None:
        safe = _redact(fields)
        if trace.sections is not None:
            trace.sections.setdefault(section, []).append(safe)
        else:
            payload = json.dumps(safe, ensure_ascii=True)
            LOGGER.log(logging.WARNING if warning else logging.DEBUG, "[%s] id=%s %s", section, trace.request_id, payload)


def fingerprint(value):
    encoded = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def mark(name, value):
    trace = _CURRENT.get()
    if trace is not None:
        trace.fingerprints[name] = value


@contextmanager
def request_trace(message, *, turn=None):
    human = trace_enabled()
    if active() or not (human or debug_enabled()):
        yield
        return
    configure_logger()
    trace = RequestTrace(uuid4().hex, turn, fingerprint(message), sections={} if human else None)
    token = _CURRENT.set(trace)
    start = perf_counter()
    try:
        event("REQUEST", turn=trace.turn, message=message)
        yield
    except Exception as error:
        event("REQUEST", warning=True, error=type(error).__name__)
        raise
    finally:
        try:
            event("PERFORMANCE", **{
                f"{phase}_ms": round(trace.elapsed[phase] * 1000, 2) if phase in trace.elapsed else None
                for phase in _PHASES
            }, total_ms=round((perf_counter() - start) * 1000, 2))
            if trace.sections is not None:
                try:
                    rendered = format_trace(trace.request_id, trace.turn, trace.sections)
                except Exception as error:
                    LOGGER.warning("Could not format CineChat trace %s (%s)", trace.request_id, type(error).__name__)
                else:
                    LOGGER.info("\n%s", rendered)
        finally:
            _CURRENT.reset(token)


@contextmanager
def timed(phase):
    trace = _CURRENT.get()
    if trace is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        trace.elapsed[phase] = trace.elapsed.get(phase, 0) + perf_counter() - start


def elapsed_ms(phase):
    trace = _CURRENT.get()
    if trace is not None and phase in trace.elapsed:
        return round(trace.elapsed[phase] * 1000, 2)
    return None


def state_snapshot(state, intent=None):
    effective = intent if intent is not None else state.current_intent if state is not None else None
    return {
        "preferences": effective.model_dump(mode="json") if effective is not None else None,
        "watched_movie_ids": sorted(state.watched_movie_ids) if state is not None else [],
        "rejected_movie_ids": sorted(state.rejected_movie_ids) if state is not None else [],
        "last_recommendation_ids": list(state.last_recommendation_ids) if state is not None else [],
    }


def compare_ranking(state_before, displayed_ids):
    trace = _CURRENT.get()
    if trace is None or "ranking" not in trace.fingerprints:
        return
    state_key = fingerprint(state_before)
    key = (trace.message_key, state_key)
    current = {**trace.fingerprints, "displayed_ids": list(displayed_ids)}
    with _LOCK:
        previous = _RECENT.get(key)
        _RECENT[key] = (trace.request_id, current)
        _RECENT.move_to_end(key)
        if len(_RECENT) > 128:
            _RECENT.popitem(last=False)
    event("RANKING", state_key=state_key, fingerprints=trace.fingerprints)
    if previous is not None:
        previous_id, baseline = previous
        changed = sorted(name for name in current.keys() | baseline.keys() if current.get(name) != baseline.get(name))
        differs = "displayed_ids" in changed
        event(
            "RANKING", warning=differs, comparison="displayed_ids_changed" if differs else "same_displayed_ids",
            previous_request_id=previous_id, changed_stages=changed,
            previous_ids=baseline["displayed_ids"], current_ids=displayed_ids,
        )


def audit_display(intent, movies, state):
    ids = [movie.tmdb_id for movie in movies]
    if len(set(ids)) != len(ids):
        event("DIAGNOSTIC", warning=True, issue="duplicate displayed IDs", ids=ids)
    for movie in movies:
        if state is not None and intent and intent.intent != "movie_info":
            if movie.tmdb_id in state.watched_movie_ids:
                event("DIAGNOSTIC", warning=True, issue="watched movie displayed", tmdb_id=movie.tmdb_id)
            if movie.tmdb_id in state.rejected_movie_ids:
                event("DIAGNOSTIC", warning=True, issue="rejected movie displayed", tmdb_id=movie.tmdb_id)
        if intent is None:
            continue
        if excluded := set(intent.genres_exclude).intersection(movie.genres):
            event("DIAGNOSTIC", warning=True, issue="excluded genre displayed", tmdb_id=movie.tmdb_id, genres=sorted(excluded))
        year = movie.release_year
        if ((intent.year_min is not None and (year is None or year < intent.year_min))
                or (intent.year_max is not None and (year is None or year > intent.year_max))):
            event("DIAGNOSTIC", warning=True, issue="release year outside filter", tmdb_id=movie.tmdb_id, year=year)
