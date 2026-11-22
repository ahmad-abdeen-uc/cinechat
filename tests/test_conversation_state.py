import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversation_state import ConversationState, IntentPatch, plan_turn
from src.intent_parser import MovieIntent


class TestConversationState(unittest.TestCase):
    def setUp(self):
        self.intent = MovieIntent(
            genres_include=["Action"], genres_exclude=["Horror"], year_min=2011,
            actor="Example Actor", count=3,
        )
        self.state = ConversationState(current_intent=self.intent, last_recommendation_ids=(30, 12, 9))

    def plan(self, text, parsed=None, state=None):
        return plan_turn(text, parsed or MovieIntent(), state if state is not None else self.state)

    def test_patch_keeps_adds_removes_replaces_and_clears_independently(self):
        intent = IntentPatch(
            add={"genres_include": ["Comedy"]}, remove={"genres_exclude": ["horror"]},
            replace={"runtime_max": 100}, clear={"actor"},
        ).apply(self.intent)
        self.assertEqual(intent.genres_include, ["Action", "Comedy"])
        self.assertEqual(intent.genres_exclude, [])
        self.assertEqual(intent.runtime_max, 100)
        self.assertIsNone(intent.actor)
        self.assertEqual(intent.year_min, 2011)
        self.assertEqual(intent.count, 3)
        self.assertEqual(self.intent.genres_exclude, ["Horror"])

    def test_funnier_preserves_hard_filters(self):
        intent = self.plan("Make them funnier.").intent
        self.assertEqual(intent.genres_include, ["Action", "Comedy"])
        self.assertEqual(intent.desired_tone, "funny")
        self.assertEqual(intent.genres_exclude, ["Horror"])
        self.assertEqual(intent.year_min, 2011)
        self.assertEqual(intent.actor, "Example Actor")
        self.assertIsNone(intent.runtime_max)
        self.assertIsNone(intent.min_rating)

    def test_funnier_does_not_broaden_existing_or_filter(self):
        current = MovieIntent(genres_include=["Action", "Drama"], genre_match="any")
        state = self.state.updated(current_intent=current)
        intent = self.plan("Make them funnier.", state=state).intent
        self.assertEqual(intent.genres_include, current.genres_include)
        self.assertEqual(intent.genre_match, "any")
        self.assertEqual(intent.desired_tone, "funny")

    def test_unrepresentable_mixed_genres_ask_for_clarification(self):
        turn = self.plan("Also Comedy or Drama.", MovieIntent(genres_include=["Comedy", "Drama"], genre_match="any"))
        self.assertIsNone(turn.intent)
        self.assertIn("alternatives", turn.question)

    def test_resolves_group_single_last_and_all_from_display_order(self):
        examples = {
            "I've already watched the first two.": (30, 12),
            "I've already seen the second one.": (12,),
            "I've already watched the last one.": (9,),
            "I've already seen all of those.": (30, 12, 9),
            "I've already seen those.": (30, 12, 9),
        }
        for message, expected in examples.items():
            with self.subTest(message=message):
                turn = self.plan(message)
                self.assertEqual(turn.kind, "watched")
                self.assertEqual(turn.reference_ids, expected)

    def test_missing_and_ambiguous_references_do_not_guess(self):
        for text in ("I've seen the fourth one.", "I've seen some of those.", "I've seen the first or second."):
            with self.subTest(text=text):
                self.assertIsNotNone(self.plan(text).question)
        self.assertIsNotNone(self.plan("I've seen those.", state=ConversationState()).question)

    def test_rejection_uses_display_order(self):
        turn = self.plan("I didn't like the last one.")
        self.assertEqual(turn.kind, "rejected")
        self.assertEqual(turn.reference_ids, (9,))

    def test_genre_correction_and_explicit_allowance(self):
        correction = self.plan("No, I meant Comedy instead of Action.").intent
        self.assertEqual(correction.genres_include, ["Comedy"])
        self.assertEqual(correction.year_min, 2011)
        self.assertEqual(correction.genres_exclude, ["Horror"])
        allowed = self.plan("Actually, horror is okay now.", MovieIntent(genres_include=["Horror"])).intent
        self.assertEqual(allowed.genres_include, ["Action"])
        self.assertEqual(allowed.genres_exclude, [])

    def test_runtime_refinement_and_explicit_clear(self):
        intent = self.plan("Something shorter, under 100 minutes.", MovieIntent(runtime_max=100)).intent
        self.assertEqual(intent.runtime_max, 100)
        self.assertEqual(intent.count, 3)
        self.assertEqual(intent.year_min, 2011)
        state = self.state.updated(current_intent=intent)
        cleared = self.plan("Actually, no runtime limit.", MovieIntent(themes_avoid=["runtime limit"]), state).intent
        self.assertIsNone(cleared.runtime_max)
        self.assertEqual(cleared.themes_avoid, [])
        self.assertIsNotNone(self.plan("Something shorter.").question)

    def test_additive_bounds_do_not_broaden_stricter_filters(self):
        state = self.state.updated(current_intent=MovieIntent(year_min=2011, runtime_max=90))
        turn = self.plan("Also after 2000 and under 100 minutes.", MovieIntent(year_min=2001, runtime_max=100), state)
        self.assertEqual(turn.intent.year_min, 2011)
        self.assertEqual(turn.intent.runtime_max, 90)
        turn = self.plan("Set the runtime limit to under 100 minutes instead.", MovieIntent(runtime_max=100), state)
        self.assertEqual(turn.intent.runtime_max, 100)
        self.assertEqual(turn.intent.year_min, 2011)

    def test_new_topic_keeps_only_the_new_intent(self):
        parsed = MovieIntent(genres_include=["Romance"])
        turn = self.plan("Now recommend romance movies instead.", parsed)
        self.assertEqual(turn.kind, "new")
        self.assertEqual(turn.intent, parsed)

    def test_explicit_new_topic_overrides_parser_refine_label(self):
        turn = self.plan("Now recommend romance movies instead.", MovieIntent(intent="refine", genres_include=["Romance"]))
        self.assertEqual(turn.kind, "new")
        self.assertEqual(turn.intent.intent, "recommend")
        self.assertEqual(turn.intent.genres_include, ["Romance"])

    def test_first_request_can_be_filter_only_or_reference_a_title(self):
        for text, parsed in (
            ("No horror.", MovieIntent(genres_exclude=["Horror"])),
            ('Recommend movies like "Them".', MovieIntent(intent="similar_movies", similar_to="Them")),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.plan(text, parsed, ConversationState()).intent, parsed)

    def test_refinement_without_context_asks_for_initial_request(self):
        turn = self.plan("Make them funnier.", state=ConversationState())
        self.assertIn("initial request", turn.question)
        self.assertIsNone(turn.intent)

    def test_state_validates_unique_positive_movie_ids(self):
        for ids in ((1, 1), (0,), (-1,)):
            with self.subTest(ids=ids), self.assertRaises(ValidationError):
                ConversationState(last_recommendation_ids=ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
