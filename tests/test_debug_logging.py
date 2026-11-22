import hashlib
import io
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cinechat_pipeline import CineChatPipeline
from src.conversation_state import ConversationState
from src.debug_logger import audit_display, configure_logger, event, request_trace
from src.intent_parser import MovieIntent
from src.movie_search import MovieResult
from src.recommendation_engine import DB_PATH, INDEX_PATH
from test_response_generator import approved_draft, ollama_response


REQUEST = "Recommend 5 action movies after 2010, but no horror."


def artifact_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class TestDebugLogging(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"CINECHAT_DEBUG": "1", "CINECHAT_TRACE": "0"})
        environment.start()
        self.addCleanup(environment.stop)
        self.stream = io.StringIO()
        self.logger = configure_logger()
        self.handler = next(handler for handler in self.logger.handlers if getattr(handler, "_cinechat_console", False))
        original = self.handler.setStream(self.stream)
        self.addCleanup(self.handler.setStream, original)

    def records(self, section):
        pattern = rf"\[{re.escape(section)}\] id=([a-f0-9]+) (.*)"
        return [(match[1], json.loads(match[2])) for match in re.finditer(pattern, self.stream.getvalue())]

    def pipeline(self):
        intent_client = Mock()
        intent_client.chat.return_value = ollama_response(MovieIntent().model_dump())
        response_client = Mock()
        response_client.chat.side_effect = lambda **call: ollama_response(approved_draft(call["messages"]))
        with (
            patch("src.intent_parser.Client", return_value=intent_client),
            patch("src.response_generator.Client", return_value=response_client),
        ):
            pipeline = CineChatPipeline()
        self.addCleanup(pipeline.close)
        return pipeline, intent_client, response_client

    def test_enabled_disabled_and_one_handler(self):
        count = len(self.logger.handlers)
        for _ in range(3):
            self.assertIs(configure_logger(), self.logger)
        self.assertEqual(len(self.logger.handlers), count)
        with patch.dict(os.environ, {"CINECHAT_DEBUG": "0"}), request_trace("Off"):
            event("REQUEST", unexpected=True)
        self.assertEqual(self.stream.getvalue(), "")
        with request_trace("On", turn=1):
            pass
        self.assertEqual(self.records("REQUEST")[0][1], {"turn": 1, "message": "On"})

    def test_request_ids_unique_and_nested_context_uses_same_id(self):
        with request_trace("First"):
            with request_trace("Nested"):
                event("STATE BEFORE", watched_movie_ids=[1])
        with request_trace("Second"):
            pass
        requests = self.records("REQUEST")
        self.assertEqual(len(requests), 2)
        self.assertNotEqual(requests[0][0], requests[1][0])
        self.assertEqual(self.records("STATE BEFORE")[0][0], requests[0][0])
        self.assertEqual(len(self.records("PERFORMANCE")), 2)

    def test_common_credentials_are_redacted(self):
        text = "Authorization: Bearer private-value and OPENAI_API_KEY=synthetic-secret and my API key is secret-phrase"
        with request_trace(text):
            event("INTENT PARSER", extracted={"semantic_query": text + " https://testuser:testpass@example.invalid/a"})
        logs = self.stream.getvalue()
        for value in ("private-value", "synthetic-secret", "testpass", "secret-phrase"):
            self.assertNotIn(value, logs)
        self.assertIn("[REDACTED]", logs)

    def test_state_ranking_display_and_calls_across_follow_up(self):
        pipeline, intent_client, response_client = self.pipeline()
        first = pipeline.run(REQUEST, state=ConversationState())
        second = pipeline.run("I've already watched the first two.", state=first.state)
        before = self.records("STATE BEFORE")[1][1]
        after = self.records("STATE AFTER")[1][1]
        self.assertEqual(before["last_recommendation_ids"], first.recommended_movie_ids)
        self.assertEqual(after["watched_movie_ids"], sorted(first.recommended_movie_ids[:2]))
        self.assertEqual(after["preferences"]["year_min"], 2011)
        self.assertEqual(after["preferences"]["genres_exclude"], ["Horror"])
        update = next(payload for _, payload in self.records("STATE UPDATE") if payload.get("reference_kind") == "watched")
        self.assertEqual(update["resolved_ids"], first.recommended_movie_ids[:2])
        positions = [payload for _, payload in self.records("RANKING") if "position" in payload]
        self.assertEqual(len(positions), 20)
        self.assertEqual([row["tmdb_id"] for row in positions[:5]], first.recommended_movie_ids)
        self.assertFalse(positions[5]["selected"])
        responses = [payload["displayed_ids"] for _, payload in self.records("LLM RESPONSE") if "displayed_ids" in payload]
        self.assertEqual(responses, [first.recommended_movie_ids, second.recommended_movie_ids])
        performance = self.records("PERFORMANCE")[-1][1]
        self.assertGreater(performance["sql_retrieval_ms"], 0)
        self.assertIsNone(performance["faiss_search_ms"])
        self.assertEqual(intent_client.chat.call_count, 2)
        self.assertEqual(response_client.chat.call_count, 2)

    def test_logging_leaves_response_query_count_and_artifacts_unchanged(self):
        pipeline, intent_client, response_client = self.pipeline()
        db = pipeline.movie_search.engine.conn
        hashes = tuple(map(artifact_hash, (DB_PATH, INDEX_PATH)))
        statements = []
        db.set_trace_callback(statements.append)
        try:
            with patch.dict(os.environ, {"CINECHAT_DEBUG": "0"}):
                quiet = pipeline.run(REQUEST, state=ConversationState())
            quiet_count = len(statements)
            statements.clear()
            logged = pipeline.run(REQUEST, state=ConversationState())
            self.assertEqual(len(statements), quiet_count)
        finally:
            db.set_trace_callback(None)
        self.assertEqual(quiet, logged)
        self.assertEqual(quiet.recommended_movie_ids, logged.recommended_movie_ids)
        self.assertEqual(db.total_changes, 0)
        self.assertEqual(tuple(map(artifact_hash, (DB_PATH, INDEX_PATH))), hashes)
        self.assertEqual(intent_client.chat.call_count, 2)
        self.assertEqual(response_client.chat.call_count, 2)

    def test_same_request_comparison_identifies_changed_intent(self):
        pipeline, intent_client, _ = self.pipeline()
        original = pipeline.run(REQUEST, state=ConversationState())
        intent_client.chat.return_value = ollama_response(MovieIntent(sort_by="oldest").model_dump())
        changed = pipeline.run(REQUEST, state=ConversationState())
        self.assertNotEqual(original.recommended_movie_ids, changed.recommended_movie_ids)
        _, comparison = self.records("RANKING")[-1]
        self.assertEqual(comparison["comparison"], "displayed_ids_changed")
        self.assertIn("parsed_intent", comparison["changed_stages"])
        self.assertNotIn("sql_candidates", comparison["changed_stages"])
        self.assertNotIn("sql_query", comparison["changed_stages"])
        self.assertIn("model-supplied sort without an explicit sort cue", self.stream.getvalue())

    def test_model_confusing_tone_with_mood_is_logged_without_changing_it(self):
        pipeline, intent_client, _ = self.pipeline()
        intent_client.chat.return_value = ollama_response(MovieIntent(user_mood="funnier").model_dump())
        with request_trace("Make them funnier."):
            parsed = pipeline.movie_search.parser.parse("Make them funnier.")
        self.assertEqual(parsed.user_mood, "funnier")
        self.assertIn("model-supplied user mood without an explicit mood cue", self.stream.getvalue())

    def test_display_audit_reports_violations_without_changing_payload(self):
        movie = MovieResult(8, "Example", 2009, ["Horror"], None, None, None, 0.8)
        movies = [movie, movie]
        with request_trace("Audit"):
            audit_display(MovieIntent(year_min=2011, genres_exclude=["Horror"]), movies,
                          ConversationState(watched_movie_ids={8}))
        issues = {payload["issue"] for _, payload in self.records("DIAGNOSTIC")}
        self.assertEqual(issues, {"duplicate displayed IDs", "watched movie displayed",
                                  "excluded genre displayed", "release year outside filter"})
        self.assertEqual(movies, [movie, movie])

    def test_readable_mode_takes_precedence_over_json(self):
        pipeline, intent_client, response_client = self.pipeline()
        intent_client.chat.return_value.prompt_eval_count = 123
        intent_client.chat.return_value.eval_count = 20
        with patch.dict(os.environ, {"CINECHAT_TRACE": "1"}), request_trace(REQUEST, turn=1):
            result = pipeline.run(REQUEST, state=ConversationState())
        trace = self.stream.getvalue()
        self.assertEqual(trace.count("CINECHAT REQUEST #1"), 1)
        self.assertRegex(trace, r"Request ID: [a-f0-9]{32}")
        self.assertIn("LLM proposed", trace)
        self.assertIn("Python validated", trace)
        self.assertIn("input 123 / output 20", trace)
        self.assertIn("FAISS: Not used - structured SQL request.", trace)
        self.assertIn("Grounding validation: PASS", trace)
        self.assertNotIn("[RANKING] id=", trace)
        self.assertNotIn('{"', trace)
        self.assertIn(result.movies[0].title, trace)
        intent_client.chat.assert_called_once()
        response_client.chat.assert_called_once()

    def test_trace_semantic_tables_dont_change_rankings_or_add_queries(self):
        pipeline, intent_client, response_client = self.pipeline()
        engine = pipeline.movie_search.engine
        db = engine.conn
        before = tuple(map(artifact_hash, (DB_PATH, INDEX_PATH)))
        mid = engine.index.id_map.at(0)
        vector = engine.index.reconstruct(mid).reshape(1, -1)
        statements = []
        db.set_trace_callback(statements.append)
        try:
            with patch.object(engine, "encode_query", return_value=vector) as encode:
                intent_client.chat.return_value = ollama_response(MovieIntent(semantic_query="space").model_dump())
                with patch.dict(os.environ, {"CINECHAT_DEBUG": "0"}):
                    quiet = pipeline.run(REQUEST, state=ConversationState())
                baseline = len(statements)
                statements.clear()
                with patch.dict(os.environ, {"CINECHAT_TRACE": "1", "CINECHAT_VECTOR_PREVIEW": "1"}):
                    traced = pipeline.run(REQUEST, state=ConversationState())
                self.assertEqual(len(statements), baseline)
                self.assertEqual(encode.call_count, 2)
        finally:
            db.set_trace_callback(None)
        self.assertEqual(quiet, traced)
        self.assertEqual(traced.strategy, "semantic")
        trace = self.stream.getvalue()
        for marker in ("TOP VECTOR MATCHES BEFORE SQL FILTERING", "TOP ELIGIBLE VECTOR MATCHES AFTER SQL FILTERING",
                       "Vector dimensions: 384", "Vector norm: 1.0000", "First 8 dimensions:",
                       "Search method: IndexIDMap2.search", "Candidates returned: 45198"):
            self.assertIn(marker, trace)
        self.assertEqual(tuple(map(artifact_hash, (DB_PATH, INDEX_PATH))), before)
        self.assertEqual(intent_client.chat.call_count, 2)
        self.assertEqual(response_client.chat.call_count, 2)

    def test_trace_shows_follow_up_validation_and_watched_title(self):
        pipeline, intent_client, _ = self.pipeline()
        first = pipeline.run(REQUEST, state=ConversationState())
        watched = first.movies[0]
        self.stream.seek(0)
        self.stream.truncate()
        intent_client.chat.return_value = ollama_response(
            {**MovieIntent(year_min=1980).model_dump(), "genres_include": ["comedy"]}
        )
        message = "I've already watched the first one."
        with patch.dict(os.environ, {"CINECHAT_TRACE": "1"}), request_trace(message, turn=2):
            event("REQUEST", known_titles={str(watched.tmdb_id): watched.title})
            next_turn = pipeline.run(message, state=first.state)
        trace = self.stream.getvalue()
        self.assertNotIn(watched.tmdb_id, next_turn.recommended_movie_ids)
        self.assertIn("Conversation type: Follow-up (watched)", trace)
        self.assertIn(f"TMDB {watched.tmdb_id} | {watched.title} | excluded by SQL", trace)
        self.assertIn("Year: 2011+", trace)
        self.assertRegex(trace, r"(?m)^Included genres\s+comedy\s+any \[changed\]")
        self.assertRegex(trace, r"(?m)^Years\s+1980\+\s+any \[changed\]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
