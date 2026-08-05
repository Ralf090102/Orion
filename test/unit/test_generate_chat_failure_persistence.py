"""Regression test: a failed LLM generation must still be saved to the
session's history.

Found live testing the real app: generate_chat_response()'s except block
(src/generation/generate.py) returned immediately on any LLM failure (e.g. a
real Ollama error like "model requires more system memory"), skipping the
"store messages in session" block entirely -- which only runs after that
try/except, on the success path. A failed generation therefore left no trace
in the session at all: not the user's message, not even an error note.
Reopening the conversation later showed it as if nothing had ever been sent,
which is what looked like "exiting and re-entering shows the empty home
screen instead of the actual session" from a real user's perspective.
"""

from unittest.mock import MagicMock

import pytest

from src.generation.generate import AnswerGenerator
from src.generation.session_manager import SessionManager
from src.utilities.config import OrionConfig


@pytest.fixture
def session_manager():
    return SessionManager(persist_to_disk=False)


@pytest.fixture
def generator(session_manager):
    config = OrionConfig()
    # Skip RAG retrieval entirely -- this test is about the LLM-failure
    # persistence path, not retrieval, and "never" keeps it a fast, isolated
    # unit test instead of hitting a real vector store/embedding model.
    config.rag.generation.rag_trigger_mode = "never"
    gen = AnswerGenerator(config)
    # Avoid a real Ollama call: llm_client.generate() raises, matching a
    # real failure like the memory-constrained one found live.
    gen.llm_client = MagicMock()
    gen.llm_client.generate.side_effect = Exception(
        "Ollama API error: model requires more system memory (4.5 GiB) than is available (2.4 GiB)"
    )
    return gen


@pytest.mark.unit
class TestFailedGenerationIsPersisted:
    def test_user_and_error_messages_are_saved_to_the_session(self, generator, session_manager):
        session_id = session_manager.create_session()

        result = generator.generate_chat_response(
            message="What is machine learning?",
            session_id=session_id,
            session_manager=session_manager,
        )

        assert result.metadata.get("llm_generation_failed") is True

        session = session_manager.get_session(session_id)
        assert len(session.messages) == 2, "expected both the user message and the error response to be saved"
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "What is machine learning?"
        assert session.messages[1]["role"] == "assistant"
        assert "model requires more system memory" in session.messages[1]["content"]

    def test_reopening_the_session_shows_the_failed_attempt_not_an_empty_history(
        self, generator, session_manager
    ):
        """Directly exercises the "leave and come back" scenario: a fresh
        get_session() call (simulating navigating back to the conversation)
        must show the real history, not an empty one."""
        session_id = session_manager.create_session()
        generator.generate_chat_response(
            message="Hello?",
            session_id=session_id,
            session_manager=session_manager,
        )

        reloaded = session_manager.get_session(session_id, reload_from_db=True)
        assert len(reloaded.messages) == 2
