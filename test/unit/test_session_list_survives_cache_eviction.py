"""Regression test: list_sessions() must not depend on which sessions
happen to still be in the in-memory cache.

Found live: creating a second chat session made the first one vanish from
the sidebar. Root cause -- list_sessions() only ever iterated
self.sessions.values() (the in-memory cache), never the DB. Meanwhile
update_session_metadata() invalidates a session's cache entry after every
persisted write (see test_session_metadata_db_persistence.py's fix,
earlier the same session) -- and that fires on essentially every real
session, via the auto-generated title right after the first exchange. So
the moment a session got titled, it dropped out of self.sessions, and
list_sessions() silently stopped showing it, even though get_session(id)
for that specific session still worked fine (lazy DB reload on cache miss)
and the session was never actually lost.
"""

import pytest

from src.generation.session_manager import SessionManager


@pytest.mark.unit
class TestListSessionsSurvivesCacheEviction:
    def test_session_still_listed_after_its_cache_entry_is_evicted(self, tmp_path):
        sm = SessionManager(persist_to_disk=True, storage_dir=tmp_path)

        session_a = sm.create_session()
        sm.add_message(session_a, role="user", content="hi")
        sm.add_message(session_a, role="assistant", content="hello")
        # Mirrors _maybe_generate_title() persisting the auto-generated
        # title -- this is what evicts session_a from the cache.
        sm.update_session_metadata(session_a, {"title": "Session A Title"})
        assert session_a not in sm.sessions, "sanity check: cache entry really was evicted"

        session_b = sm.create_session()
        sm.add_message(session_b, role="user", content="hi again")

        listed_ids = {s["session_id"] for s in sm.list_sessions()}
        assert session_a in listed_ids, "session A must still be listed even though its cache entry was evicted"
        assert session_b in listed_ids

        # The listing's title must reflect what was actually persisted too,
        # not just the session's continued presence.
        session_a_entry = next(s for s in sm.list_sessions() if s["session_id"] == session_a)
        assert session_a_entry["metadata"]["title"] == "Session A Title"

    def test_list_sessions_still_works_without_persistence(self):
        """No DB to fall back to -- must keep working off the in-memory
        cache exactly as before."""
        sm = SessionManager(persist_to_disk=False)
        session_id = sm.create_session()
        sm.add_message(session_id, role="user", content="hi")

        listed_ids = {s["session_id"] for s in sm.list_sessions()}
        assert session_id in listed_ids
