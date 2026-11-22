import sqlite3
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cinechat_pipeline import CineChatResult
from src.conversation_state import ConversationState
from src.intent_parser import MovieIntent
from src.movie_search import MovieResult
from src.response_generator import RecommendationPresentation


def visible_text(app):
    return "\n".join(element.value.replace("\\", "") for element in [*app.markdown, *app.caption])


class TestCineChatApp(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        movie = MovieResult(111, "Verified Title", 2014, ["Comedy", "Drama"], 7.6, 42, None, 0.8)
        self.result = CineChatResult(
            "Verified Title / this fallback text should not duplicate the card",
            [111], [movie], MovieIntent(), "semantic", [], False,
            RecommendationPresentation(["Here is a verified match."], {111: "Its genres fit your request."}, None),
        )
        patcher = patch("src.cinechat_pipeline.CineChatPipeline")
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = self.factory.return_value
        self.backend.run.return_value = self.result

    def open_app(self):
        self.app.run()
        self.assertFalse(self.app.exception)

    def submit(self, prompt):
        self.app.chat_input[0].set_value(prompt).run()
        self.assertFalse(self.app.exception)

    def test_opens_with_examples_and_lazy_backend(self):
        self.open_app()
        self.assertEqual(self.app.title[0].value, "CineChat")
        self.assertIn("Your AI Movie Recommendation Assistant", visible_text(self.app))
        self.assertIn("Follow-ups keep your filters", visible_text(self.app))
        self.assertEqual(len([button for button in self.app.button if button.key.startswith("example_")]), 5)
        self.assertFalse(self.app.chat_message)
        self.factory.assert_not_called()

    def test_renders_verified_cards_once(self):
        self.open_app()
        self.submit("Find me a comedy")
        self.backend.run.assert_called_once_with("Find me a comedy", state=ConversationState())
        self.assertEqual([message.name for message in self.app.chat_message], ["user", "assistant"])
        content = visible_text(self.app)
        self.assertIn("Find me a comedy", content)
        self.assertEqual(content.count("Verified Title"), 1)
        self.assertIn("Verified Title (2014)", content)
        self.assertIn("TMDB audience rating 7.6/10", content)
        self.assertIn("Comedy · Drama", content)
        self.assertIn("Its genres fit your request.", content)

    def test_example_uses_pipeline_and_persists_across_rerun(self):
        self.open_app()
        self.app.button(key="example_1").click().run()
        prompt = "Recommend mind-bending movies like Inception."
        self.backend.run.assert_called_once_with(prompt, state=ConversationState())
        self.app.run()
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertIn(prompt, visible_text(self.app))
        self.backend.run.assert_called_once()

    def test_multiple_requests_reuse_one_thread_affine_backend(self):
        activity = []

        class ThreadAffineBackend:
            def __init__(self, result):
                self.result = result
                self.connection = sqlite3.connect(":memory:")
                activity.append(("open", threading.get_ident()))

            def run(self, prompt, *, state):
                self.connection.execute("SELECT 1").fetchone()
                activity.append((prompt, threading.get_ident()))
                return self.result

            def close(self):
                self.connection.close()
                activity.append(("close", threading.get_ident()))

        self.factory.side_effect = lambda: ThreadAffineBackend(self.result)
        self.open_app()
        self.submit("First request")
        self.submit("Second request")
        self.assertEqual(len(self.app.chat_message), 4)
        self.factory.assert_called_once()
        st.cache_resource.clear()
        self.assertEqual([action for action, _ in activity], ["open", "First request", "Second request", "close"])
        self.assertEqual(len({thread for _, thread in activity}), 1)

    def test_new_conversation_clears_chat_without_reloading_resources(self):
        self.open_app()
        self.submit("Find an adventure")
        self.app.button(key="clear_chat").click().run()
        self.assertFalse(self.app.chat_message)
        self.assertTrue(self.app.button(key="example_0"))
        self.backend.run.assert_called_once()
        self.submit("Find a western")
        self.factory.assert_called_once()
        self.assertEqual(len(self.app.chat_message), 2)

    def test_passes_updated_state_and_resets_it_on_new_conversation(self):
        state = ConversationState(
            current_intent=MovieIntent(genres_include=["Comedy"]),
            last_recommendation_ids=(111,), watched_movie_ids={99},
        )
        self.result.state = state
        self.open_app()
        self.submit("Find a comedy")
        self.app.run()
        self.assertEqual(self.app.session_state["conversation"], state)
        self.submit("Make them shorter")
        self.backend.run.assert_called_with("Make them shorter", state=state)
        self.app.button(key="clear_chat").click().run()
        self.assertEqual(self.app.session_state["conversation"], ConversationState())
        self.assertFalse(self.app.chat_message)
        self.factory.assert_called_once()

    def test_blank_input_skips_pipeline(self):
        self.open_app()
        self.submit("   ")
        self.factory.assert_not_called()
        self.assertFalse(self.app.chat_message)

    def test_ollama_error_is_explained(self):
        self.backend.run.return_value = CineChatResult(
            "I couldn't reach the local movie assistant.", [], [], None, "none",
            ["The local Ollama model is unavailable for intent extraction."], True,
        )
        self.open_app()
        self.submit("Show me a comedy")
        self.assertIn("couldn't reach the local movie assistant", visible_text(self.app))
        self.assertIn("qwen2.5:3b", self.app.info[0].value)

    def test_unexpected_error_uses_safe_chat_message(self):
        self.backend.run.side_effect = ValueError("private traceback details")
        self.open_app()
        self.submit("Show me a comedy")
        self.assertIn("I couldn't finish that search.", visible_text(self.app))
        self.assertNotIn("private traceback details", visible_text(self.app))

    def test_missing_catalog_is_reported_before_initializing_backend(self):
        with patch("src.recommendation_engine.DB_PATH", ROOT / "missing.db"):
            self.open_app()
            self.submit("Show me a comedy")
        self.assertIn("The movie catalog is missing.", visible_text(self.app))
        self.factory.assert_not_called()

    def test_missing_index_is_reported_for_search_that_needs_it(self):
        self.backend.run.side_effect = RuntimeError("could not open index")
        with patch("src.recommendation_engine.INDEX_PATH", ROOT / "missing.faiss"):
            self.open_app()
            self.submit("Show me a comedy")
        self.assertIn("The movie search index is missing.", visible_text(self.app))

    def test_empty_and_unsupported_searches_use_backend_response_text(self):
        self.open_app()
        for response, intent in (
            ("No verified matches were found.", MovieIntent()),
            ("I don't have current streaming-provider availability data.", MovieIntent(intent="unsupported")),
        ):
            self.backend.run.return_value = CineChatResult(response, [], [], intent, "none", [], False)
            self.submit("Can I watch this?")
            self.assertIn(response, visible_text(self.app))

    def test_generation_fallback_displays_verified_movie_and_warning(self):
        self.result.fallback_used = True
        self.result.warnings = ["Ollama response generation was unavailable. Used a catalog-based response."]
        self.open_app()
        self.submit("Show me a comedy")
        self.assertIn("Verified Title (2014)", visible_text(self.app))
        self.assertIn(self.result.warnings[0], visible_text(self.app))


if __name__ == "__main__":
    unittest.main(verbosity=2)
