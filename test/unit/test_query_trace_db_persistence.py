"""Regression/feature test: a QueryTrace saved via
SessionManager.save_query_trace() must actually survive a reload from
disk (round-trips through the query_traces table), not just live in
whatever in-memory object wrote it -- mirrors
test_session_metadata_db_persistence.py's pattern for the same reason:
a fresh SessionManager instance on the same storage_dir has nothing
cached, so it can only see what actually made it to disk.
"""

import json
import sqlite3

import pytest

from src.generation.session_manager import SessionManager
from src.generation.trace import QueryTrace


@pytest.mark.unit
class TestQueryTraceSurvivesDbReload:
    def test_saved_trace_round_trips_through_a_fresh_manager(self, tmp_path):
        trace = QueryTrace(
            request_id="req-abc",
            session_id="sess-1",
            query_text="What is machine learning?",
            retrieved=[{"retriever": "semantic", "document_id": "d1", "score": 0.8, "rank": 0}],
            fused=[{"document_id": "d1", "score": 0.8, "search_type": "hybrid"}],
            reranked=[{"document_id": "d1", "score": 0.9}],
            mmr=[{"document_id": "d1", "score": 0.9}],
            context_chunks=[{"source_file": "ml.pdf", "citation_text": "[1]", "final_score": 0.9, "length": 120}],
            model="mistral:latest",
            timing={"total_time": 1.23},
            rag_retrieval_triggered=True,
        )

        sm1 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        sm1.save_query_trace(trace)

        # A brand new instance pointed at the same storage dir -- proves
        # the write actually hit disk, not just sm1's own connection cache.
        sm2 = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        conn = sqlite3.connect(sm2.db_path)
        row = conn.execute(
            "SELECT session_id, trace_json FROM query_traces WHERE request_id = ?",
            ("req-abc",),
        ).fetchone()
        conn.close()

        assert row is not None
        session_id, trace_json = row
        assert session_id == "sess-1"
        reloaded = json.loads(trace_json)
        assert reloaded["request_id"] == "req-abc"
        assert reloaded["retrieved"] == trace.retrieved
        assert reloaded["context_chunks"] == trace.context_chunks
        assert reloaded["model"] == "mistral:latest"

    def test_saving_the_same_request_id_twice_replaces_not_duplicates(self, tmp_path):
        sm = SessionManager(persist_to_disk=True, storage_dir=tmp_path)
        sm.save_query_trace(QueryTrace(request_id="req-dup", session_id="s1", query_text="first"))
        sm.save_query_trace(QueryTrace(request_id="req-dup", session_id="s1", query_text="second"))

        conn = sqlite3.connect(sm.db_path)
        rows = conn.execute(
            "SELECT trace_json FROM query_traces WHERE request_id = ?", ("req-dup",)
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert json.loads(rows[0][0])["query_text"] == "second"

    def test_save_query_trace_is_a_noop_when_persistence_disabled(self, tmp_path):
        sm = SessionManager(persist_to_disk=False)
        # Must not raise -- there's no db_path to write to in-memory-only mode.
        sm.save_query_trace(QueryTrace(request_id="req-x", session_id="s1", query_text="hi"))
