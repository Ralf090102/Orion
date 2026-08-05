"""Regression test: auto-title generation must not depend on the session
having exactly 2 messages.

_maybe_generate_title() (backend/websockets/chat.py) used to require
message_count == 2 exactly (the first exchange) before generating a title.
That broke as soon as any earlier attempt in the session had already added
messages -- e.g. a failed generation under memory pressure, which
generate_chat_response()'s except block now persists (user message + error
response) rather than silently dropping (see the "Native Tauri window bugs"
fix earlier the same day). A failed first attempt followed by a successful
retry leaves 4 messages, never 2 again -- so the session stayed "New Chat"
forever even though a real reply had gone through.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.websockets.chat import ChatWebSocketHandler
from src.generation.session_manager import SessionManager


@pytest.fixture
def session_manager():
    return SessionManager(persist_to_disk=False)


@pytest.fixture
def handler(session_manager):
    generator = MagicMock()
    ws = MagicMock()
    h = ChatWebSocketHandler(
        websocket=ws,
        session_id="",  # set per-test once the session exists
        session_manager=session_manager,
        generator=generator,
        config=MagicMock(rag=MagicMock(llm=MagicMock(model="mistral:latest"))),
    )
    h.send_message = AsyncMock()
    return h


@pytest.mark.unit
@pytest.mark.asyncio
class TestTitleGenerationSurvivesEarlierFailedAttempts:
    async def test_title_still_generates_after_a_failed_attempt_left_4_messages(
        self, handler, session_manager
    ):
        session_id = session_manager.create_session()
        handler.session_id = session_id

        # Simulate: a failed generation (persisted per today's other fix) ...
        session_manager.add_message(session_id, role="user", content="What is machine learning?")
        session_manager.add_message(
            session_id, role="assistant", content="I encountered an error while generating the response: ..."
        )
        # ...then a successful retry of the same question.
        session_manager.add_message(session_id, role="user", content="What is machine learning?")
        session_manager.add_message(session_id, role="assistant", content="Machine learning is a subset of AI.")

        assert len(session_manager.get_session(session_id).messages) == 4

        handler.generator.llm_client = None  # not used directly
        with pytest_mock_ollama_client("Machine Learning Basics"):
            await handler._maybe_generate_title("What is machine learning?")

        session = session_manager.get_session(session_id)
        assert session.metadata.get("title") == "Machine Learning Basics"

    async def test_does_not_regenerate_once_a_real_title_exists(self, handler, session_manager):
        session_id = session_manager.create_session()
        handler.session_id = session_id
        session_manager.add_message(session_id, role="user", content="hi")
        session_manager.add_message(session_id, role="assistant", content="hello")
        session_manager.update_session_metadata(session_id, {"title": "A Real Title"})

        with pytest_mock_ollama_client("Should Not Be Used") as mock_generate:
            await handler._maybe_generate_title("hi")
            mock_generate.assert_not_called()

        session = session_manager.get_session(session_id)
        assert session.metadata.get("title") == "A Real Title"


class pytest_mock_ollama_client:
    """Patches src.core.llm.OllamaClient.generate for the duration of a
    `with` block, returning a fixed title. Named lowercase/function-style
    deliberately so it reads naturally at each call site above."""

    def __init__(self, title: str):
        self.title = title
        self._patcher = None

    def __enter__(self):
        from unittest.mock import patch

        self._patcher = patch(
            "src.core.llm.OllamaClient.generate",
            return_value={"message": {"content": self.title}},
        )
        return self._patcher.start()

    def __exit__(self, *exc):
        self._patcher.stop()
