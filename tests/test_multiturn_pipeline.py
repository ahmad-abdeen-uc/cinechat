import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cinechat_pipeline import CineChatPipeline
from src.conversation_state import ConversationState
from src.intent_parser import IntentParser, MovieIntent
from src.movie_search import MovieSearch
from src.recommendation_engine import RecommendationEngine
from src.response_generator import ResponseGenerator
from test_response_generator import approved_draft, ollama_response


class TestMultiturnPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngine()
        cls.addClassCleanup(cls.engine.close)

    def setUp(self):
        self.intent_client = Mock()
        self.response_client = Mock()
        self.intent_client.chat.return_value = ollama_response(MovieIntent().model_dump())
        self.response_client.chat.side_effect = (
            lambda **call: ollama_response(approved_draft(call["messages"]))
        )
        with patch("src.intent_parser.Client", return_value=self.intent_client):
            parser = IntentParser()
        self.addCleanup(parser.close)
        search = MovieSearch(parser=parser, engine=self.engine)
        self.pipeline = CineChatPipeline(search, ResponseGenerator(client=self.response_client))

    def start(self):
        return self.pipeline.run(
            "Recommend 5 action movies after 2010, but no horror.", state=ConversationState(),
        )

    def check_action(self, result, *, funny=False):
        self.assertTrue(result.movies, result.response_text)
        self.assertEqual(result.intent.year_min, 2011)
        self.assertEqual(result.intent.genres_exclude, ["Horror"])
        self.assertEqual(result.state.last_recommendation_ids, tuple(result.recommended_movie_ids))
        self.assertEqual(result.state.last_successful_intent, result.intent)
        self.assertEqual([movie.tmdb_id for movie in result.movies], result.recommended_movie_ids)
        self.assertEqual(len(set(result.recommended_movie_ids)), len(result.movies))
        for movie in result.movies:
            self.assertIn("Action", movie.genres)
            self.assertNotIn("Horror", movie.genres)
            self.assertGreaterEqual(movie.release_year, 2011)
            if funny:
                self.assertIn("Comedy", movie.genres)
        self.assertIsNone(result.intent.min_rating)
        self.assertIsNone(result.intent.runtime_max)

    def test_three_turn_conversation_uses_one_parse_and_one_response_per_turn(self):
        first = self.start()
        self.check_action(first)
        second = self.pipeline.run("Make them funnier.", state=first.state)
        self.check_action(second, funny=True)
        self.assertEqual(second.intent.desired_tone, "funny")
        watched = set(second.recommended_movie_ids[:2])
        third = self.pipeline.run("I've already watched the first two.", state=second.state)
        self.check_action(third, funny=True)
        self.assertEqual(third.state.watched_movie_ids, watched)
        self.assertFalse(watched.intersection(third.recommended_movie_ids))
        self.assertEqual(self.intent_client.chat.call_count, 3)
        self.assertEqual(self.response_client.chat.call_count, 3)

    def test_consecutive_correction_runtime_and_allowance(self):
        result = self.start()
        corrected = self.pipeline.run("No, I meant Comedy instead of Action.", state=result.state)
        self.assertEqual(corrected.intent.genres_include, ["Comedy"])
        self.assertEqual(corrected.intent.year_min, 2011)
        shorter = self.pipeline.run("Something shorter, under 100 minutes.", state=corrected.state)
        self.assertTrue(shorter.movies, shorter.response_text)
        self.assertEqual(shorter.intent.runtime_max, 100)
        for movie in shorter.movies:
            runtime = self.engine.conn.execute("SELECT runtime FROM movies WHERE tmdb_id = ?", (movie.tmdb_id,))
            self.assertLessEqual(runtime.fetchone()[0], 100)
        allowed = self.pipeline.run("Actually, horror is okay now.", state=shorter.state)
        self.assertEqual(allowed.intent.genres_exclude, [])
        self.assertEqual(allowed.intent.genres_include, ["Comedy"])
        self.assertEqual(allowed.intent.year_min, 2011)
        self.assertEqual(allowed.intent.runtime_max, 100)

    def test_watched_and_rejected_persist_across_new_topic(self):
        first = self.start()
        watched_id = first.recommended_movie_ids[0]
        watched = self.pipeline.run("I've already seen the first one.", state=first.state)
        rejected_id = watched.recommended_movie_ids[-1]
        rejected = self.pipeline.run("I didn't like the last one.", state=watched.state)
        romance = self.pipeline.run("Now recommend romance movies instead.", state=rejected.state)
        self.assertEqual(romance.intent.genres_include, ["Romance"])
        self.assertEqual(romance.intent.genres_exclude, [])
        self.assertIsNone(romance.intent.year_min)
        self.assertEqual(romance.state.watched_movie_ids, {watched_id})
        self.assertEqual(romance.state.rejected_movie_ids, {rejected_id})
        self.assertFalse({watched_id, rejected_id}.intersection(romance.recommended_movie_ids))

    def test_ambiguity_and_out_of_range_preserve_prior_state(self):
        first = self.start()
        result = self.pipeline.run("I've seen the tenth one.", state=first.state)
        self.assertIn("Which movies", result.response_text)
        self.assertEqual(result.state, first.state)
        self.assertFalse(result.movies)
        self.response_client.chat.assert_called_once()

    def test_ambiguous_title_is_never_guessed(self):
        initial = ConversationState()
        result = self.pipeline.run('I already watched "Titanic".', state=initial)
        self.assertIn("Which version", result.response_text)
        self.assertEqual(result.state, initial)
        resolved = self.pipeline.run("I already watched Titanic (1997).", state=initial)
        matches = self.pipeline.movie_search.find_title("Titanic (1997)")
        self.assertEqual(len(matches), 1)
        self.assertEqual(resolved.state.watched_movie_ids, {matches[0]["tmdb_id"]})
        self.response_client.chat.assert_not_called()

    def test_no_results_keep_new_filter_without_replacing_last_display(self):
        first = self.start()
        result = self.pipeline.run("After 2099, please.", state=first.state)
        self.assertFalse(result.movies)
        self.assertEqual(result.state.current_intent.year_min, 2100)
        self.assertEqual(result.state.last_successful_intent, first.intent)
        self.assertEqual(result.state.last_recommendation_ids, first.state.last_recommendation_ids)
        self.assertIn("match all your filters", result.response_text)

    def test_conflict_and_unsupported_keep_prior_state(self):
        first = self.start()
        conflict = self.pipeline.run("Also no Action.", state=first.state)
        self.assertEqual(conflict.state, first.state)
        self.assertFalse(conflict.movies)
        self.assertIn("conflicting", conflict.response_text)
        unsupported = self.pipeline.run("What movies are available on Netflix tonight?", state=first.state)
        self.assertEqual(unsupported.state, first.state)
        self.assertIn("streaming-provider", unsupported.response_text)

    def test_reset_clears_state_without_inference(self):
        first = self.start()
        state = first.state.updated(watched_movie_ids={first.recommended_movie_ids[0]})
        reset = self.pipeline.run("Forget everything. Start over.", state=state)
        self.assertEqual(reset.state, ConversationState())
        self.intent_client.chat.assert_called_once()
        self.response_client.chat.assert_called_once()

    def test_parser_connection_failure_does_not_mutate_state(self):
        first = self.start()
        self.intent_client.chat.side_effect = ConnectionError("offline")
        result = self.pipeline.run("Make them funnier.", state=first.state)
        self.assertEqual(result.state, first.state)
        self.assertTrue(result.fallback_used)
        self.response_client.chat.assert_called_once()

    def test_standalone_search_still_parses_once(self):
        result = self.pipeline.movie_search.search("Top 3 action movies after 2010.")
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.movies), 3)
        self.assertEqual(result.intent.year_min, 2011)
        self.intent_client.chat.assert_called_once()
        self.response_client.chat.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
