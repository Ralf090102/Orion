"""Regression test: session metadata updates (auto-generated titles above
all) must actually survive a reload from disk, not just update the
in-memory cache.

Found live: a real end-to-end run through the Tauri-spawned backend logged
`Failed to update metadata for session ...: no such column: metadata` on
every single title-generation attempt. _update_session_metadata_in_db()
was UPDATE-ing a "metadata" column that has never existed in the sessions
table (see the CREATE TABLE in _init_database(): it's "title" +
"metadata_json") -- so persistence silently failed every time (caught and
only logged), while the in-memory session object still looked correct
right up until the next cache invalidation or process restart, at which
point the DB's stale "title" ('New Chat') would come back. Combined with
update_session_metadata() now invalidating the cache on every persisted
write (see test_websocket_title_generation.py's fix), this meant a
just-generated title could vanish again within the same request.
"""

import pytest

from src.generation.session_manager import SessionManager


@pytest.mark.unit
class TestSessionMetadataSurvivesDbReload:
    def test_updated_title_survives_a_fresh_reload_from_disk(self, tmp_path):
        # First manager: create a session and update its title, simulating
        # what _maybe_generate_title() does after the first real exchange.
        sm1 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm1.create_session()
        sm1.add_message(session_id, role="user", content="hi")
        sm1.add_message(session_id, role="assistant", content="hello")

        ok = sm1.update_session_metadata(session_id, {"title": "Somatosensory System Overview"})
        assert ok is True

        # A brand new SessionManager instance pointed at the same storage
        # dir has nothing cached -- it can only see what actually made it
        # to disk. This is the scenario that exposed the bug: reopening the
        # app (or just the cache-invalidation that follows any persisted
        # metadata update) forces exactly this kind of fresh DB read.
        sm2 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        reloaded = sm2.get_session(session_id)

        assert reloaded is not None
        assert reloaded.metadata.get("title") == "Somatosensory System Overview"

    def test_update_within_the_same_manager_also_reflects_after_cache_invalidation(self, tmp_path):
        sm = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        session_id = sm.create_session()

        sm.update_session_metadata(session_id, {"title": "A Real Title"})
        # update_session_metadata() invalidates the cache itself when
        # persist_to_disk is True (see session_manager.py) -- forcing the
        # very next get_session() to prove the DB round-trip, not just
        # return the pre-invalidation in-memory object.
        session = sm.get_session(session_id, reload_from_db=True)

        assert session.metadata.get("title") == "A Real Title"
