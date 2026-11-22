import logging
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import streamlit as st

from src.cinechat_pipeline import CineChatPipeline, CineChatResult
from src.conversation_state import ConversationState
from src.debug_logger import event, readable, request_trace
from src.recommendation_engine import DB_PATH, INDEX_PATH


LOGGER = logging.getLogger(__name__)
EXAMPLES = (
    ("Funny and uplifting", "I'm feeling sad. Suggest something funny and uplifting, but no horror."),
    ("Like Inception", "Recommend mind-bending movies like Inception."),
    ("Top-rated action", "Show me the top five action movies based on audience ratings after 2010."),
    ("Space exploration", "Recommend science fiction movies about space exploration."),
    ("Christopher Nolan", "Suggest highly rated movies by Christopher Nolan."),
)


class SessionPipeline:
    """Keep all SQLite operations, including close, on one session-owned thread."""

    def __init__(self):
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cinechat")
        self._pipeline = None
        self._closed = False

    def run(self, prompt: str, state: ConversationState) -> CineChatResult:
        return self._worker.submit(copy_context().run, self._run, prompt, state).result()

    def _run(self, prompt: str, state: ConversationState) -> CineChatResult:
        if self._pipeline is None:
            self._pipeline = CineChatPipeline()
        return self._pipeline.run(prompt, state=state)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._worker.submit(self._close).result()
        finally:
            self._worker.shutdown()

    def _close(self):
        if self._pipeline is not None:
            self._pipeline.close()


@st.cache_resource(scope="session", on_release=SessionPipeline.close, show_spinner=False)
def session_pipeline() -> SessionPipeline:
    return SessionPipeline()


def plain_markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~$<])", r"\\\1", text)


def clear_chat():
    st.session_state.messages = []
    st.session_state.conversation = ConversationState()
    st.session_state.pop("example_prompt", None)


def choose_example(prompt: str):
    st.session_state.example_prompt = prompt


def trace_previous_titles():
    remaining = set(st.session_state.conversation.last_recommendation_ids)
    titles = {}
    for _, result in reversed(st.session_state.messages):
        if not remaining:
            break
        if isinstance(result, CineChatResult):
            for movie in result.movies:
                if movie.tmdb_id in remaining:
                    titles[str(movie.tmdb_id)] = movie.title
                    remaining.remove(movie.tmdb_id)
    if titles:
        event("REQUEST", known_titles=titles)


def reply(prompt: str, state: ConversationState) -> CineChatResult | str:
    if not DB_PATH.is_file():
        return "The movie catalog is missing. Restore data/processed/cinechat.db and try again."
    try:
        return session_pipeline().run(prompt, state)
    except (ConnectionError, TimeoutError):
        return "I couldn't reach Ollama. Start Ollama locally and check that qwen2.5:3b is installed."
    except sqlite3.Error as error:
        LOGGER.error("Could not read the movie database (%s)", type(error).__name__)
        return "I couldn't read the SQLite movie catalog. Check that cinechat.db is accessible and try again."
    except (OSError, RuntimeError) as error:
        LOGGER.error("Could not load the movie search resources (%s)", type(error).__name__)
        if not INDEX_PATH.is_file():
            return "The movie search index is missing. Restore data/vector_store/movies.faiss and try again."
        return "I couldn't load a local search resource. Check the movie index and the cached MiniLM model."
    except Exception as error:
        LOGGER.error("CineChat request failed (%s)", type(error).__name__)
        return "I couldn't finish that search. Try again, or check the local server log if the problem continues."


def display_result(result: CineChatResult):
    presentation = result.presentation
    if presentation is None or not result.movies:
        st.markdown(plain_markdown(result.response_text))
    else:
        for paragraph in presentation.opening:
            st.markdown(plain_markdown(paragraph))
        titles = Counter(" ".join(movie.title.split()).casefold() for movie in result.movies)
        for movie in result.movies:
            title = " ".join(movie.title.split())
            year = f" ({movie.release_year})" if movie.release_year is not None else " (year unknown)"
            details = []
            if movie.rating is not None:
                details.append(f"TMDB audience rating {movie.rating:.1f}/10")
            if movie.genres:
                details.append(" · ".join(movie.genres))
            if titles[title.casefold()] > 1:
                details.append(f"TMDB ID {movie.tmdb_id}")
            with st.container(border=True):
                st.markdown(f"**{plain_markdown(title + year)}**")
                if details:
                    st.caption(plain_markdown(" | ".join(details)))
                st.markdown(plain_markdown(presentation.reasons[movie.tmdb_id]))
        if presentation.follow_up:
            st.markdown(plain_markdown(presentation.follow_up))

    if result.intent is None and any("unavailable for intent extraction" in warning for warning in result.warnings):
        st.info("Start Ollama locally and check that qwen2.5:3b is installed.")
    shown = presentation.opening if presentation is not None else [result.response_text]
    warnings = [warning for warning in result.warnings if warning not in shown]
    if warnings:
        with st.expander("Search notes", expanded=result.fallback_used):
            for warning in warnings:
                st.markdown(plain_markdown(warning))


def display_message(role: str, content: CineChatResult | str):
    avatar = ":material/movie:" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        if isinstance(content, str):
            st.markdown(plain_markdown(content))
        else:
            display_result(content)


def main():
    st.set_page_config(page_title="CineChat", page_icon="🎬", layout="centered")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation", ConversationState())

    with st.sidebar:
        st.header("CineChat")
        st.button("New Conversation", key="clear_chat", on_click=clear_chat, help="Clear chat history")
        st.markdown("**How it works**")
        st.write("Describe what you want to watch. CineChat searches a local movie catalog and explains why its picks fit.")
        st.markdown("**The catalog**")
        st.write("45,433 movies from the Kaggle Movies Dataset, with 45,198 searchable movie embeddings.")
        st.caption("Historical catalog. No real-time streaming availability.")
        with st.expander("Technical details"):
            st.write("SQLite catalog · FAISS search · MiniLM embeddings · Qwen2.5:3b through Ollama")

    st.title("CineChat")
    st.markdown("**Your AI Movie Recommendation Assistant**")
    st.caption("Follow-ups keep your filters in this session. Use New Conversation to start fresh.")

    typed = st.chat_input("Ask CineChat for a movie…", max_chars=1200)
    prompt = st.session_state.pop("example_prompt", None) or typed
    if not st.session_state.messages and not prompt:
        st.subheader("What are you in the mood for?")
        with st.container(horizontal=True):
            for index, (label, example) in enumerate(EXAMPLES):
                st.button(label, key=f"example_{index}", help=example, on_click=choose_example, args=(example,))

    for role, content in st.session_state.messages:
        display_message(role, content)

    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Enter a movie request to get started.")
            return
        st.session_state.messages.append(("user", prompt))
        display_message("user", prompt)
        with st.chat_message("assistant", avatar=":material/movie:"):
            with st.spinner("Searching the catalog and preparing your reply…", show_time=True):
                with request_trace(prompt, turn=(len(st.session_state.messages) + 1) // 2):
                    if readable():
                        trace_previous_titles()
                    answer = reply(prompt, st.session_state.conversation)
            if isinstance(answer, str):
                st.markdown(plain_markdown(answer))
            else:
                display_result(answer)
                if answer.state is not None:
                    st.session_state.conversation = answer.state
        st.session_state.messages.append(("assistant", answer))


if __name__ == "__main__":
    main()
