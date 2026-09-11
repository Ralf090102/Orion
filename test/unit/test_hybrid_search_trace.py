"""Feature test: HybridSearcher.search()'s optional `trace` parameter
records pre-fusion (per retriever) and post-fusion candidates into a
QueryTrace, without affecting search behavior -- pure recording, part of
the 2026-09-11 /llm-council per-query retrieval trace decision (see
src/generation/trace.py, Eru's Orion-Roadmap.md).
"""

from unittest.mock import MagicMock

import pytest

from src.generation.trace import QueryTrace
from src.retrieval.search import HybridSearcher, SearchResult
from src.utilities.config import OrionConfig


def _result(doc_id: str, score: float, search_type: str) -> SearchResult:
    return SearchResult(document_id=doc_id, content=f"content for {doc_id}", metadata={}, score=score, search_type=search_type)


@pytest.fixture
def hybrid_searcher():
    semantic_searcher = MagicMock()
    semantic_searcher.search.return_value = [
        _result("sem-1", 0.9, "semantic"),
        _result("sem-2", 0.8, "semantic"),
    ]
    keyword_searcher = MagicMock()
    keyword_searcher.search.return_value = [
        _result("kw-1", 0.7, "keyword"),
    ]
    return HybridSearcher(semantic_searcher, keyword_searcher, config=OrionConfig())


@pytest.mark.unit
class TestHybridSearchTrace:
    def test_trace_records_pre_fusion_candidates_tagged_by_retriever(self, hybrid_searcher):
        trace = QueryTrace(request_id="r1", session_id="s1", query_text="q")

        hybrid_searcher.search("q", k=5, trace=trace)

        semantic_entries = [e for e in trace.retrieved if e["retriever"] == "semantic"]
        keyword_entries = [e for e in trace.retrieved if e["retriever"] == "keyword"]
        assert {e["document_id"] for e in semantic_entries} == {"sem-1", "sem-2"}
        assert {e["document_id"] for e in keyword_entries} == {"kw-1"}

    def test_trace_records_post_fusion_results(self, hybrid_searcher):
        trace = QueryTrace(request_id="r1", session_id="s1", query_text="q")

        results = hybrid_searcher.search("q", k=5, trace=trace)

        fused_ids = {e["document_id"] for e in trace.fused}
        assert fused_ids == {r.document_id for r in results}
        assert len(trace.fused) > 0

    def test_search_behavior_is_unchanged_when_trace_is_none(self, hybrid_searcher):
        # Default (no trace passed) -- existing callers unaffected.
        results = hybrid_searcher.search("q", k=5)
        assert len(results) > 0
