import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, field_validator

from .debug_logger import active, event, timed
from .intent_parser import GENRE, GENRE_TERMS, MovieIntent


class ConversationState(BaseModel):
    current_intent: MovieIntent | None = None
    last_successful_intent: MovieIntent | None = None
    watched_movie_ids: frozenset[PositiveInt] = Field(default_factory=frozenset)
    rejected_movie_ids: frozenset[PositiveInt] = Field(default_factory=frozenset)
    last_recommendation_ids: tuple[PositiveInt, ...] = ()

    @field_validator("last_recommendation_ids")
    @classmethod
    def unique_display_ids(cls, ids):
        if len(set(ids)) != len(ids):
            raise ValueError("Displayed recommendation IDs must be unique")
        return ids

    @timed("state_update")
    def updated(self, **changes) -> "ConversationState":
        return ConversationState.model_validate({**self.model_dump(), **changes})


@dataclass
class IntentPatch:
    """Fields absent from the patch keep their previous values."""

    replace: dict[str, object] = field(default_factory=dict)
    add: dict[str, list[str]] = field(default_factory=dict)
    remove: dict[str, list[str]] = field(default_factory=dict)
    clear: set[str] = field(default_factory=set)

    def apply(self, previous: MovieIntent) -> MovieIntent:
        data = previous.model_dump()
        defaults = MovieIntent().model_dump()
        for name in self.clear:
            data[name] = defaults[name]
        data.update(self.replace)
        for name in self.add.keys() | self.remove.keys():
            removed = {value.casefold() for value in self.remove.get(name, [])}
            values = [value for value in data[name] if value.casefold() not in removed]
            unique = {value.casefold(): value for value in values}
            for value in self.add.get(name, []):
                unique.setdefault(value.casefold(), value)
            data[name] = list(unique.values())
        result = MovieIntent.model_validate(data)
        if active():
            before = previous.model_dump(mode="json")
            after = result.model_dump(mode="json")
            edited = set(self.replace) | set(self.add) | set(self.remove) | self.clear
            changed = {name for name in before if before[name] != after[name]}
            event(
                "STATE UPDATE",
                added={name: [value for value in after[name] if value not in before[name]]
                       for name in sorted(changed & self.add.keys())},
                removed={name: [value for value in before[name] if value not in after[name]]
                         for name in sorted(changed & self.remove.keys())},
                cleared=sorted(changed & self.clear),
                replaced={name: {"before": before[name], "after": after[name]}
                          for name in sorted(changed & self.replace.keys())},
                retained=sorted(before.keys() - changed),
            )
            if unexpected := changed - edited:
                event("DIAGNOSTIC", warning=True, issue="preferences changed outside refinement patch",
                      fields=sorted(unexpected))
        return result

    def has_changes(self) -> bool:
        return bool(self.replace or self.add or self.remove or self.clear)


@dataclass(frozen=True)
class Turn:
    kind: Literal["new", "refine", "watched", "rejected"]
    intent: MovieIntent | None = None
    reference_ids: tuple[int, ...] = ()
    reference_title: str | None = None
    question: str | None = None


RESET_REQUEST = re.compile(
    r"\s*(?:forget everything(?:[.!;,]\s*start over)?|start over|reset (?:the )?(?:chat|conversation))[.!]?\s*",
    re.I,
)
REACTION = re.compile(
    r"^\s*(?:(?P<watched>(?:i(?:'ve| have)?\s+)?(?:already\s+)?(?:watched|seen))"
    r"|(?P<rejected>(?:i\s+)?(?:don't like|didn't like)|reject|skip|don't recommend))"
    r"\s+(?P<reference>.+?)\s*[.!]?\s*$", re.I
)
NUMBERS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
ORDINALS = dict(zip("first second third fourth fifth sixth seventh eighth ninth tenth".split(), range(1, 11)))
NUMBER = r"(?:\d+|" + "|".join(NUMBERS) + ")"
POSITIONS = "|".join(ORDINALS)
REFERENCE_HINT = re.compile(
    rf"^(?:the )?(?:{POSITIONS}|last)(?:\s+(?:and|or|one|movie|\d+)\b|$)"
    r"|^(?:some|all|both|a few|a couple)(?: of)? (?:those|these|them)\b"
    r"|^(?:those|these|them|both)\b|^(?:number\s+|#)\d+", re.I,
)
REPLACEMENT = re.compile(rf"(?P<new>{GENRE})\s+instead of\s+(?P<old>{GENRE})", re.I)
ALLOW_GENRE = re.compile(
    rf"(?:(?P<okay>{GENRE})\s+is\s+(?:okay|ok|fine|allowed)(?:\s+now)?|"
    rf"(?:allow|stop excluding)\s+(?P<allow>{GENRE}))", re.I,
)
NEW_TOPIC = re.compile(r"\b(?:new topic|switch(?:ing)? to|different topic)\b|^now\s+(?:recommend|suggest|show|find)\b", re.I)
FILTER_REQUEST = re.compile(
    r"^(?:under|before|after|exclude|excluding|without|no|starring|directed by|at least)\s+", re.I,
)
FOLLOW_UP = re.compile(
    r"^(?:actually\b|also\b|instead\b|make\b|keep\b|no[,!]|give me more\b|more please\b|"
    r"(?:something\s+)?(?:shorter|longer|funnier)\b|(?:leave|drop|clear|remove|set|change|replace|any)\b)|"
    r"\b(?:previous (?:recommendations|results|ones)|(?:make|keep|show) (?:them|those (?:movies|results)))\b",
    re.I,
)
CLEAR_FILTERS = {
    "year": {"year_min", "year_max"}, "release year": {"year_min", "year_max"},
    "runtime": {"runtime_max"}, "language": {"language"}, "rating": {"min_rating", "rating_source"},
    "genre": {"genres_include", "genres_exclude", "genre_match", "semantic_query"},
    "actor": {"actor"}, "director": {"director"},
}
SCALAR_FILTERS = (
    "year_min", "year_max", "runtime_max", "min_rating", "actor", "director",
    "language", "similar_to", "user_mood", "desired_tone", "rating_source", "sort_by",
)
LIST_FILTERS = ("genres_include", "genres_exclude", "themes", "themes_avoid", "exclude_titles")
TIGHTER_BOUND = {"year_min": max, "year_max": min, "runtime_max": min, "min_rating": max}


def _number(value: str) -> int:
    return int(value) if value.isdecimal() else NUMBERS[value]


def _references(reference: str, ids: tuple[int, ...]) -> tuple[int, ...] | None:
    if re.fullmatch(r"(?:all(?: of)? (?:those|these|them)|all|those|these|them)", reference):
        return ids
    if reference in {"both", "both of them", "both of those"}:
        return ids if len(ids) == 2 else ()
    group = re.fullmatch(rf"(?:the )?(first|last) ({NUMBER})(?: (?:movies|ones))?", reference)
    if group:
        count = _number(group[2])
        if 0 < count <= len(ids):
            return ids[:count] if group[1] == "first" else ids[-count:]
        return ()
    position = re.fullmatch(rf"(?:the )?({POSITIONS}|last)(?: (?:one|movie))?", reference)
    if position:
        index = (len(ids) - 1) if position[1] == "last" else ORDINALS[position[1]] - 1
        return ids[index:index + 1] if 0 <= index < len(ids) else ()
    return None


def _reaction(text: str, state: ConversationState) -> Turn | None:
    match = REACTION.fullmatch(text)
    if not match:
        return None
    reference = match["reference"].strip()
    if not match["watched"] and re.fullmatch(GENRE, reference, re.I):
        return None
    kind = "watched" if match["watched"] else "rejected"
    if reference.startswith('"') and reference.endswith('"'):
        return Turn(kind, reference_title=reference[1:-1].strip())
    ids = _references(reference.casefold(), state.last_recommendation_ids)
    if ids:
        return Turn(kind, reference_ids=ids)
    if ids == () or REFERENCE_HINT.search(reference):
        return Turn(kind, question="Which movies did you mean? Give their titles or positions in the last list.")
    return Turn(kind, reference_title=reference)


def _count(text: str) -> int | None:
    match = re.search(
        rf"\b(?:show me|give me|recommend|suggest|make it|top|just|only)\s+({NUMBER})\b"
        rf"|\b({NUMBER})\s+(?:movies|films|results|recommendations)\b", text, re.I,
    )
    return _number((match[1] or match[2]).casefold()) if match else None


def _refinement(text: str, previous: MovieIntent, parsed: MovieIntent) -> Turn:
    patch = IntentPatch()
    replacement = REPLACEMENT.search(text)
    allowed = {
        GENRE_TERMS[(match["okay"] or match["allow"]).casefold()]
        for match in ALLOW_GENRE.finditer(text)
    }
    explicit_replacement = re.search(r"\b(?:instead|replace|change|set)\b", text, re.I)
    for name in SCALAR_FILTERS:
        value = getattr(parsed, name)
        if value is not None and value != MovieIntent.model_fields[name].default:
            old_value = getattr(previous, name)
            if name in TIGHTER_BOUND and old_value is not None and not explicit_replacement:
                value = TIGHTER_BOUND[name](old_value, value)
            if (name in {"actor", "director", "language", "similar_to"} and old_value and value != old_value
                    and re.search(r"\b(?:also|as well)\b", text, re.I) and not explicit_replacement):
                return Turn("refine", question=f"Which {name.replace('_', ' ')} should I use for this search?")
            patch.replace[name] = value
    for name in LIST_FILTERS:
        values = getattr(parsed, name)
        if values:
            patch.add[name] = values.copy()

    if replacement:
        old = GENRE_TERMS[replacement["old"].casefold()]
        new = GENRE_TERMS[replacement["new"].casefold()]
        if old not in previous.genres_include:
            return Turn("refine", question=f"I wasn't requiring {old}. Which genre would you like me to replace?")
        patch.replace["genres_include"] = [new if genre == old else genre for genre in previous.genres_include]
        patch.add.pop("genres_include", None)
        patch.clear.add("semantic_query")
    if allowed:
        if allowed.intersection(parsed.genres_exclude):
            return Turn("refine", question="Should I allow or exclude those genres?")
        patch.remove["genres_exclude"] = sorted(allowed)
        for name in ("genres_include", "genres_exclude"):
            if name in patch.add:
                patch.add[name] = [genre for genre in patch.add[name] if genre not in allowed]

    if not replacement and (added := patch.add.get("genres_include")):
        old_genres = set(previous.genres_include)
        new_genres = set(added)
        old_any = previous.genre_match == "any" and len(old_genres) > 1
        new_any = parsed.genre_match == "any" and len(new_genres) > 1
        if old_any and not new_any and old_genres & new_genres:
            patch.replace.update(genres_include=added, genre_match="all")
            patch.add.pop("genres_include")
        elif old_any and new_any and (old_genres <= new_genres or new_genres <= old_genres):
            patch.replace["genres_include"] = added if new_genres <= old_genres else previous.genres_include
            patch.add.pop("genres_include")
        elif new_any and old_genres & new_genres and not old_any:
            patch.add.pop("genres_include")
        elif old_genres and (old_any or new_any):
            return Turn("refine", question="Which genres must the movies match, and which are alternatives?")
        else:
            patch.replace["genre_match"] = "any" if new_any else "all"

    if re.search(r"\bfunnier\b", text, re.I):
        patch.replace["desired_tone"] = "funny"
        if previous.genre_match == "all" or len(previous.genres_include) <= 1:
            patch.add.setdefault("genres_include", []).append("Comedy")
            patch.replace["genre_match"] = "all"
    if re.search(r"\bshorter\b", text, re.I) and parsed.runtime_max is None:
        return Turn("refine", question="What maximum runtime would you like?")

    for label, names in CLEAR_FILTERS.items():
        clear = re.search(
            rf"\b(?:any {label}|no {label} (?:limit|filter|restriction)|"
            rf"(?:clear|remove|drop) (?:the )?{label} (?:limit|filter|restriction))\b", text, re.I,
        )
        if clear:
            patch.clear.update(names)
            if "themes_avoid" in patch.add:
                patch.add["themes_avoid"] = [
                    theme for theme in patch.add["themes_avoid"]
                    if theme.casefold() not in clear[0].casefold()
                ]
            for name in names:
                patch.replace.pop(name, None)
                patch.add.pop(name, None)

    count = _count(text)
    if count is not None:
        if not 1 <= count <= 10:
            return Turn("refine", question="How many movies should I show? Choose between 1 and 10.")
        patch.replace["count"] = count
    if not patch.has_changes():
        return Turn("refine", question="What should I change about the last recommendations?")
    if parsed.similar_to:
        patch.replace["intent"] = "similar_movies"
    elif parsed.intent == "top_rated":
        patch.replace["intent"] = "top_rated"
    effective = patch.apply(previous)
    if active():
        defaults = MovieIntent()
        unsupported_additions = [
            name for name in SCALAR_FILTERS
            if getattr(previous, name) == getattr(defaults, name)
            and getattr(effective, name) != getattr(defaults, name)
            and getattr(parsed, name) != getattr(effective, name)
            and not (name == "desired_tone" and re.search(r"\bfunnier\b", text, re.I))
        ]
        if unsupported_additions:
            event("DIAGNOSTIC", warning=True, issue="potential unrequested preference introduced",
                  fields=unsupported_additions)
    return Turn("refine", intent=effective)


@timed("state_update")
def plan_turn(message: str, parsed: MovieIntent, state: ConversationState) -> Turn:
    text = message.strip().replace("’", "'")
    if parsed.intent == "unsupported":
        return Turn("new", parsed)
    if reaction := _reaction(text, state):
        return reaction
    correction = REPLACEMENT.search(text) or ALLOW_GENRE.search(text)
    if not correction and (NEW_TOPIC.search(text) or parsed.intent == "movie_info"):
        return Turn("new", parsed.model_copy(update={"intent": "recommend"}) if parsed.intent == "refine" else parsed)
    dependent = correction or parsed.intent == "refine" or FOLLOW_UP.search(text)
    if dependent or (state.current_intent is not None and FILTER_REQUEST.search(text)):
        if state.current_intent is None:
            return Turn("refine", question="What would you like to watch? I need an initial request before refining it.")
        if parsed.intent == "clarify" and not correction and not re.search(r"\bfunnier\b", text, re.I):
            return Turn("refine", question=parsed.clarification_question or "Which filter should I change?")
        return _refinement(text, state.current_intent, parsed)
    return Turn("new", parsed)
