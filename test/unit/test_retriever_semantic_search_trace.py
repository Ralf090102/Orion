"""Regression test found by a bug sweep on the request_id/trace diff
itself: OrionRetriever._perform_search()'s `trace` parameter only reached
HybridSearcher.search() on the "hybrid" branch -- the "semantic" branch
(a real, documented search_type value) never populated
trace.retrieved/trace.fused, even though trace.reranked/trace.mmr are
still populated downstream in query() regardless of search_type. That
left an internally inconsistent QueryTrace for any semantic-only query:
populated rerank/MMR stages but an empty retrieval stage, which reads as
"nothing was retrieved" even when results existed -- exactly the kind of
misleading trace this feature exists to prevent.
"""

from unittest.mock import MagicMock

import pytest

from src.generation.trace import QueryTrace
from src.retrieval.retriever import OrionRetriever
from src.retrieval.search import SearchResult
from src.utilities.config import OrionConfig


@pytest.fixture
def retriever(monkeypatch):
    r = OrionRetriever(config=OrionConfig())
    r._initialized = True
    r._embedding_manager = MagicMock()
    r._vector_store = MagicMock()

    fake_semantic_searcher = MagicMock()
    fake_semantic_searcher.search.return_value = [
        SearchResult(document_id="d1", content="c", metadata={}, score=0.9, search_type="semantic"),
        SearchResult(document_id="d2", content="c", metadata={}, score=0.8, search_type="semantic"),
    ]
    monkeypatch.setattr(
        "src.retrieval.retriever.SemanticSearcher", lambda *a, **kw: fake_semantic_searcher
    )
    return r


@pytest.mark.unit
class TestSemanticSearchTrace:
    def test_semantic_only_search_populates_retrieved_and_fused(self, retriever):
        trace = QueryTrace(request_id="r1", session_id="s1", query_text="q")

        results = retriever._perform_search("q", k=5, search_type="semantic", trace=trace)

        assert len(results) == 2
        assert {e["document_id"] for e in trace.retrieved} == {"d1", "d2"}
        assert all(e["retriever"] == "semantic" for e in trace.retrieved)
        assert {e["document_id"] for e in trace.fused} == {"d1", "d2"}

    def test_semantic_search_behavior_unchanged_when_trace_is_none(self, retriever):
        results = retriever._perform_search("q", k=5, search_type="semantic")
        assert len(results) == 2
