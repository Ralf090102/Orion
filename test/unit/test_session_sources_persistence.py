"""Regression test: RAG source citations must survive leaving and
re-entering a conversation, not just render on the live first response.

Found live testing the real app: citations rendered correctly right after a
message was sent, but vanished after navigating away and back. Root-caused
to two independent gaps, both fixed here:

1. `GET /api/chat/sessions/{id}` (backend/api/chat.py) built each response
   `Message` without ever reading `msg.get("sources")`, even though
   SessionManager both stores and reloads sources correctly (its own
   `message_sources` SQLite table). The frontend never had a chance to
   render citations after a reload.
2. The DB-reload path (`_load_session_from_db`) rebuilt each source dict as
   `{citation, content, score, rank}` -- missing `index`, which the
   frontend's `ChatSource` type requires (used to link inline `[N]`
   citation markers back to the source list).
"""

import pytest

from backend.models.chat import Message
from src.generation.session_manager import SessionManager


@pytest.mark.unit
class TestSourcesSurviveSessionReload:
    def test_add_message_with_sources_then_get_session_includes_them(self):
        """Exercises the in-memory (persist_to_disk=False) path directly --
        the same shape backend/api/chat.py's get_session() reads from."""
        sm = SessionManager(persist_to_disk=False)
        session_id = sm.create_session()

        sm.add_message(session_id, role="user", content="What is machine learning?")
        sm.add_message(
            session_id,
            role="assistant",
            content="Machine learning is...",
            rag_triggered=True,
            sources=[
                {"index": 1, "citation": "Machine Learning Basics", "score": 0.87, "text": "..."},
            ],
        )

        session = sm.get_session(session_id)
        assert session.messages[1]["sources"], "sources must be stored on the message"

        # Mirrors backend/api/chat.py's get_session() response construction.
        response_messages = [
            Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                tokens=msg.get("tokens", 0),
                timestamp=msg.get("timestamp", ""),
                sources=msg.get("sources") or [],
            )
            for msg in session.messages
        ]

        assert response_messages[0].sources == []
        # add_message() normalizes "text" (the key AnswerGenerator._format_sources()
        # actually uses) into "content" too, so every downstream reader can
        # rely on "content" -- see test_session_sources_content_key.py. The
        # original "text" key is preserved alongside it.
        assert response_messages[1].sources == [
            {"index": 1, "citation": "Machine Learning Basics", "score": 0.87, "text": "...", "content": "..."}
        ]

    def test_message_without_sources_still_serializes_with_empty_list(self):
        """A plain conversational reply (no RAG) must not break -- sources
        should be an empty list, not missing/None."""
        sm = SessionManager(persist_to_disk=False)
        session_id = sm.create_session()
        sm.add_message(session_id, role="user", content="hi")
        sm.add_message(session_id, role="assistant", content="Hello!")

        session = sm.get_session(session_id)
        response_messages = [
            Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                sources=msg.get("sources") or [],
                timestamp=msg.get("timestamp", ""),
            )
            for msg in session.messages
        ]
        assert all(m.sources == [] for m in response_messages)
