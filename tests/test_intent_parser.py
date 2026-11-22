
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intent_parser import IntentParser, MODEL_NAME, MovieIntent

TEST_CASES = [
    {
        "name": "Sad mood, comedy, no horror",
        "query": (
            "I'm feeling sad tonight. Recommend 5 funny "
            "and uplifting movies, but no horror."
        ),
        "expected": {
            "intent": "recommend",
            "user_mood": "sad",
            "desired_tone": "uplifting",
            "count": 5
        },
        "contains": {
            "genres_include": ["Comedy"],
            "genres_exclude": ["Horror"]
        },
        "not_contains": {
            "themes_avoid": ["sadness"]
        }
    },
    {
        "name": "Sad mood without explicit preference",
        "query": (
            "I'm feeling sad. Can you suggest "
            "something good to watch?"
        ),
        "expected": {
            "intent": "recommend",
            "count": 5,
            "user_mood": "sad",
            "desired_tone": "unspecified",
            "genres_include": [],
            "genres_exclude": [],
            "themes_avoid": [],
            "exclude_titles": []
        },
        "not_contains": {
            "genres_include": ["Comedy"]
        }
    },
    {
        "name": "Top-rated action movies after 2010",
        "query": (
            "Show me the top 5 action movies "
            "based on audience ratings after 2010."
        ),
        "expected": {
            "intent": "top_rated",
            "year_min": 2011,
            "sort_by": "weighted_rating",
            "rating_source": "audience",
            "count": 5
        },
        "contains": {
            "genres_include": ["Action"]
        }
    },
    {
        "name": "Movies similar to Inception",
        "query": (
            "Recommend movies like Inception, "
            "but absolutely no horror."
        ),
        "expected": {
            "intent": "similar_movies",
            "similar_to": "Inception"
        },
        "contains": {
            "genres_exclude": ["Horror"]
        }
    },
    {
        "name": "Actor and runtime constraints",
        "query": (
            "Recommend movies starring Leonardo DiCaprio "
            "that are under 120 minutes."
        ),
        "expected": {
            "actor": "Leonardo DiCaprio",
            "runtime_max": 120
        }
    },
    {
        "name": "OR genre filtering",
        "query": (
            "I want an Action OR Comedy movie. "
            "It does not need to be both."
        ),
        "expected": {
            "genre_match": "any",
            "themes_avoid": []
        },
        "contains": {
            "genres_include": ["Action", "Comedy"]
        }
    },
    {
        "name": "AND genre filtering",
        "query": (
            "Recommend movies that are both "
            "Action AND Comedy."
        ),
        "expected": {
            "genre_match": "all"
        },
        "contains": {
            "genres_include": ["Action", "Comedy"]
        }
    },
    {
        "name": "Director and year",
        "query": (
            "Show me Christopher Nolan movies "
            "released after 2000."
        ),
        "expected": {
            "director": "Christopher Nolan",
            "year_min": 2001
        }
    },
    {
        "name": "Default recommendation count",
        "query": (
            "Recommend science fiction movies "
            "about space exploration."
        ),
        "expected": {
            "intent": "recommend",
            "count": 5
        },
        "contains": {
            "genres_include": ["Science Fiction"]
        }
    },
    {
        "name": "Exclude specific movie",
        "query": (
            "Recommend romantic movies, "
            "but don't recommend Titanic."
        ),
        "contains": {
            "genres_include": ["Romance"],
            "exclude_titles": ["Titanic"]
        }
    },
    {
        "name": "Unsupported streaming availability",
        "query": (
            "What movies are available "
            "on Netflix tonight?"
        ),
        "expected": {
            "intent": "unsupported"
        }
    },
    {
        "name": "Follow-up refinement",
        "query": (
            "Actually, make the previous recommendations "
            "under 100 minutes and exclude horror."
        ),
        "expected": {
            "intent": "refine",
            "runtime_max": 100
        },
        "contains": {
            "genres_exclude": ["Horror"]
        }
    }
]

TEST_CASES += [
    {
        "name": "Multiple explicit exclusions and normalized genres",
        "query": 'Suggest SCI-FI films without horror or war. Skip "Alien" and "Aliens".',
        "expected": {"genres_include": ["Science Fiction"],
                     "genres_exclude": ["Horror", "War"],
                     "exclude_titles": ["Alien", "Aliens"]},
    },
    {
        "name": "Contradictory genre requirements",
        "query": "I want horror movies, but no horror or crime.",
        "expected": {"intent": "clarify", "genres_exclude": ["Horror", "Crime"]},
        "clarifies": True,
    },
    {
        "name": "Contradictory years",
        "query": "Recommend movies released after 2015 and before 2000.",
        "expected": {"intent": "clarify"},
        "clarifies": True,
    },
    {
        "name": "Genre substrings and numeric theme",
        "query": "Suggest movies about a transaction and the year 2049.",
        "expected": {"genres_include": [], "genres_exclude": [], "runtime_max": None,
                     "actor": None, "director": None, "min_rating": None},
    },
    {
        "name": "Current mood and a different viewing tone",
        "query": "I feel anxious. Please suggest something uplifting.",
        "expected": {"user_mood": "anxious", "desired_tone": "uplifting",
                     "genres_include": [], "genres_exclude": [], "themes_avoid": []},
    },
    {
        "name": "Title containing a genre",
        "query": 'Like "War", without horror.',
        "expected": {"intent": "similar_movies", "similar_to": "War",
                     "genres_include": [], "genres_exclude": ["Horror"]},
    },
    {
        "name": "Unspecified meaning of recent",
        "query": "Only recent movies, without horror.",
        "expected": {"intent": "clarify", "genres_exclude": ["Horror"]},
        "clarifies": True,
    },
    {
        "name": "Adversarial model output cannot add hard filters",
        "query": "Suggest movies about friendship.",
        "model_result": {"genres_include": ["Comedy"], "genres_exclude": ["War"],
                         "year_min": 2000, "year_max": 2020, "runtime_max": 90,
                         "actor": "Tom Hanks", "director": "Christopher Nolan",
                         "exclude_titles": ["Titanic"], "min_rating": 8,
                         "rating_source": "audience"},
        "expected": {"genres_include": [], "genres_exclude": [], "runtime_max": None,
                     "actor": None, "director": None, "exclude_titles": [],
                     "min_rating": None, "rating_source": "unspecified"},
    },
    {
        "name": "Preserve exclusions across separate clauses",
        "query": 'No war or horror. Avoid movies about grief. Do not recommend "Titanic".',
        "model_result": {},
        "expected": {"genres_exclude": ["War", "Horror"], "genres_include": [],
                     "themes_avoid": ["grief"], "exclude_titles": ["Titanic"]},
    },
    {
        "name": "Spelled-out count before a genre and a viewing tone",
        "query": "I'm feeling sad tonight. Recommend five funny and uplifting movies, but no horror.",
        "model_result": {},
        "expected": {"intent": "recommend", "user_mood": "sad", "desired_tone": "uplifting",
                     "genres_include": ["Comedy"], "genres_exclude": ["Horror"], "count": 5},
    },
    {
        "name": "Chat example preserves humor and the horror exclusion",
        "query": "I'm feeling sad. Suggest something funny and uplifting, but no horror.",
        "model_result": {},
        "expected": {"intent": "recommend", "user_mood": "sad", "desired_tone": "uplifting",
                     "genres_include": ["Comedy"], "genres_exclude": ["Horror"]},
    },
]



def evaluate(actual, test):
    failures = []

    expected_fields = {
        "year_min": None, "year_max": None, "exclude_titles": [],
        "min_rating": None, "runtime_max": None, "genres_exclude": [],
        **test.get("expected", {})
    }
    for field in test.get("contains", {}):
        if field not in test.get("expected", {}):
            expected_fields.pop(field, None)
    for field, expected in expected_fields.items():
        actual_value = getattr(actual, field)

        if actual_value != expected:
            failures.append(
                f"{field}: expected {expected!r}, "
                f"got {actual_value!r}"
            )

    for field, values in test.get("contains", {}).items():
        actual_values = getattr(actual, field)

        for value in values:
            if value not in actual_values:
                failures.append(
                    f"{field}: missing {value!r} "
                    f"in {actual_values!r}"
                )

    for field, values in test.get("not_contains", {}).items():
        actual_values = getattr(actual, field)

        for value in values:
            if value in actual_values:
                failures.append(
                    f"{field}: unexpected {value!r} "
                    f"in {actual_values!r}"
                )

    if test.get("clarifies") and not actual.clarification_question:
        failures.append("clarification_question: expected a nonempty question")
    return failures


def make_test(case):
    def check(self):
        if "model_result" in case:
            parser = IntentParser()
            parser.close()
            response = SimpleNamespace(
                message=SimpleNamespace(
                    content=MovieIntent(**case["model_result"]).model_dump_json()
                )
            )
            parser.client = Mock()
            parser.client.chat.return_value = response
        else:
            parser = self.live_parser
        result = parser.parse(case["query"])
        if "model_result" in case:
            parser.client.chat.assert_called_once()
            request = parser.client.chat.call_args.kwargs
            self.assertEqual(request["model"], MODEL_NAME)
            self.assertEqual(request["messages"][-1]["content"], case["query"])
            self.assertEqual(request["format"], MovieIntent.model_json_schema())
        failures = evaluate(result, case)
        self.assertFalse(failures, f"{case['name']}\n" + "\n".join(failures))

    check.__doc__ = case["name"]
    return check


class TestIntentParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_parser = IntentParser()
        cls.addClassCleanup(cls.live_parser.close)


for number, case in enumerate(TEST_CASES, start=1):
    setattr(TestIntentParser, f"test_{number:02d}", make_test(case))


if __name__ == "__main__":
    unittest.main(verbosity=2)
