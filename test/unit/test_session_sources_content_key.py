"""Regression test: source preview text must survive persistence, in both
the in-memory and DB-reload paths.

Found live: generate_chat_response() passes AnswerGenerator._format_sources()
output straight to SessionManager.add_message(sources=...). That output
previews source text under the key "text" (see generate.py's
_format_sources(): `"text": ctx.get("text", "")[:200] + "..."`), not
"content". Both add_message()'s in-memory copy (read by
backend/api/chat.py's GET handler, which the frontend's ChatSource.content
expects) and _add_message_sources_to_db()'s INSERT (which read
source.get("content", "")) only ever looked for "content" -- so the preview
text was empty everywhere except the live first-response WS message (which
reads result.sources directly, unnormalized, and happens to accept the
"text" key on the frontend side of that one path).
"""

import pytest

from src.generation.session_manager import SessionManager


@pytest.mark.unit
class TestSourceContentKeySurvivesPersistence:
    def test_in_memory_message_normalizes_text_key_to_content(self):
        sm = SessionManager(persist_to_disk=False)
        session_id = sm.create_session()
        sm.add_message(session_id, role="user", content="what is machine learning?")
        sm.add_message(
            session_id,
            role="assistant",
            content="ML is...",
            rag_triggered=True,
            # Exact shape AnswerGenerator._format_sources() produces.
            sources=[{"index": 1, "citation": "Machine Learning Basics", "text": "Machine Learning Fundamentals...", "score": 0.87}],
        )

        session = sm.get_session(session_id)
        assert session.messages[1]["sources"][0]["content"] == "Machine Learning Fundamentals..."

    def test_db_reload_preserves_source_content_preview(self, tmp_path):
        sm1 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm1.create_session()
        sm1.add_message(session_id, role="user", content="what is machine learning?")
        sm1.add_message(
            session_id,
            role="assistant",
            content="ML is...",
            rag_triggered=True,
            sources=[{"index": 1, "citation": "Machine Learning Basics", "text": "Machine Learning Fundamentals...", "score": 0.87}],
        )

        # Fresh manager instance, same DB -- nothing cached, must round-trip
        # through the message_sources table.
        sm2 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        reloaded = sm2.get_session(session_id)

        assert reloaded.messages[1]["sources"][0]["content"] == "Machine Learning Fundamentals..."
        assert reloaded.messages[1]["sources"][0]["citation"] == "Machine Learning Basics"
