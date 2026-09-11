"""Regression test for the message_id parameter added to
SessionManager.add_message(): generate_chat_response() now passes its
request_id explicitly so the assistant message's persisted id and the
QueryTrace's request_id agree (see src/generation/generate.py,
src/generation/trace.py). Guards both directions: an explicit id must be
honored verbatim, and omitting it must still auto-generate one (the
pre-existing behavior every other caller relies on).
"""

import pytest

from src.generation.session_manager import SessionManager


@pytest.mark.unit
class TestAddMessageIdOverride:
    def test_explicit_message_id_is_honored(self, tmp_path):
        sm = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm.create_session()

        returned_id = sm.add_message(session_id, role="assistant", content="hi", message_id="fixed-id")

        assert returned_id == "fixed-id"

    def test_omitted_message_id_still_auto_generates(self, tmp_path):
        sm = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm.create_session()

        returned_id = sm.add_message(session_id, role="user", content="hi")

        assert returned_id is not None
        assert returned_id != "fixed-id"

    def test_explicit_message_id_survives_a_fresh_reload_from_disk(self, tmp_path):
        sm1 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm1.create_session()
        sm1.add_message(session_id, role="assistant", content="hi", message_id="fixed-id")

        sm2 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        reloaded = sm2.get_session(session_id)

        message_ids = [m["id"] for m in reloaded.messages]
        assert "fixed-id" in message_ids
