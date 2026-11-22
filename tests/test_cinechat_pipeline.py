import json
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cinechat_pipeline import CineChatPipeline
from src.intent_parser import MovieIntent
from test_response_generator import approved_draft, ollama_response


class TestCineChatPipeline(unittest.TestCase):
    request = "Top 2 action movies based on audience ratings after 2010."

    def setUp(self):
        self.intent_client = Mock()
        self.generation_client = Mock()
        self.intent_client.chat.return_value = ollama_response(MovieIntent().model_dump())
        self.generation_client.chat.side_effect = self.approved_response
        with (
            patch("src.intent_parser.Client", return_value=self.intent_client),
            patch("src.response_generator.Client", return_value=self.generation_client),
        ):
            self.pipeline = CineChatPipeline()
        self.addCleanup(self.pipeline.close)

    @staticmethod
    def approved_response(**call):
        return ollama_response(approved_draft(call["messages"]))

    def test_two_calls_through_real_parser_and_read_only_search(self):
        result = self.pipeline.run(self.request, conversation_context="The user prefers concise answers.")
        self.intent_client.chat.assert_called_once()
        self.generation_client.chat.assert_called_once()
        self.assertEqual(result.strategy, "structured")
        self.assertEqual(result.intent.year_min, 2011)
        self.assertEqual(len(result.recommended_movie_ids), 2)
        self.assertEqual(result.recommended_movie_ids, [movie.tmdb_id for movie in result.movies])
        self.assertFalse(result.fallback_used)
        for movie in result.movies:
            self.assertIn("Action", movie.genres)
            self.assertGreaterEqual(movie.release_year, 2011)
            self.assertIn(movie.title, result.response_text)
        payload = json.loads(self.generation_client.chat.call_args.kwargs["messages"][-1]["content"])
        self.assertEqual(payload["user_message"], self.request)
        self.assertIn("concise answers", payload["conversation_context"])

    def test_failed_generation_gets_at_most_one_retry_and_one_search(self):
        self.generation_client.chat.side_effect = None
        self.generation_client.chat.return_value = ollama_response({"bad": "schema"})
        search = self.pipeline.movie_search
        with patch.object(search, "search", wraps=search.search) as run_search:
            result = self.pipeline.run(self.request)
        run_search.assert_called_once_with(self.request)
        self.intent_client.chat.assert_called_once()
        self.assertEqual(self.generation_client.chat.call_count, 2)
        self.assertTrue(result.fallback_used)
        self.assertEqual(len(result.recommended_movie_ids), 2)

    def test_valid_controlled_retry_avoids_fallback(self):
        def retry(**call):
            if self.generation_client.chat.call_count == 1:
                return ollama_response({"bad": "schema"})
            self.assertIn("prior response failed validation", call["messages"][0]["content"])
            return self.approved_response(**call)
        self.generation_client.chat.side_effect = retry
        result = self.pipeline.run(self.request)
        self.assertFalse(result.fallback_used)
        self.assertEqual(self.generation_client.chat.call_count, 2)
        self.intent_client.chat.assert_called_once()

    def test_intent_connection_failure_has_no_recommendations(self):
        self.intent_client.chat.side_effect = ConnectionError("offline")
        result = self.pipeline.run(self.request)
        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.intent)
        self.assertFalse(result.recommended_movie_ids)
        self.assertIn("Ollama", result.response_text)
        self.intent_client.chat.assert_called_once()
        self.generation_client.chat.assert_not_called()

    def test_generation_connection_failure_keeps_verified_results(self):
        self.generation_client.chat.side_effect = ConnectionError("offline")
        result = self.pipeline.run(self.request)
        self.assertEqual(len(result.recommended_movie_ids), 2)
        self.assertTrue(result.fallback_used)
        self.intent_client.chat.assert_called_once()
        self.generation_client.chat.assert_called_once()

    def test_unsupported_request_uses_only_intent_call(self):
        result = self.pipeline.run("What movies are available on Netflix tonight?")
        self.assertIn("streaming-provider availability", result.response_text)
        self.assertFalse(result.recommended_movie_ids)
        self.assertFalse(result.fallback_used)
        self.intent_client.chat.assert_called_once()
        self.generation_client.chat.assert_not_called()

    def test_shutdown_closes_owned_clients_and_database(self):
        conn = self.pipeline.movie_search.engine.conn
        self.pipeline.close()
        self.intent_client.close.assert_called_once()
        self.generation_client.close.assert_called_once()
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
