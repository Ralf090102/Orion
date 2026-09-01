"""Regression test for MMRSearcher.search() crashing on numpy.ndarray embeddings.

ChromaDB's collection.query(..., include=["embeddings"]) returns each
embedding as a raw numpy.ndarray. SemanticSearcher.search() used to store
that ndarray directly on SearchResult.embedding without normalizing it to a
plain list (unlike embeddings.py's own encode_single()/encode_batch(), which
always call .tolist() before returning). MMRSearcher.search() then did a bare
`if embedding:` truthiness check on it, which numpy rejects with "The truth
value of an array with more than one element is ambiguous. Use a.any() or
a.all()".

MMRSearcher's own try/except caught this and returned the un-diversified
candidates instead of crashing -- so the bug was invisible in normal use
(masked whenever reranking runs, since _apply_reranking() rebuilds
SearchResults without an embedding, and only reachable when a request
disables reranking while leaving MMR enabled). Found via code investigation,
reproduced directly against the pinned chromadb version.

Fixed two places: SemanticSearcher.search() now normalizes the embedding to
a plain list at the source, and MMRSearcher.search()'s filter uses an
explicit `is not None and len(...) > 0` check instead of bare truthiness, so
it's correct regardless of which type reaches it.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.search import MMRSearcher, SearchResult
from src.utilities.config import OrionConfig


def _make_result(doc_id: str, embedding) -> SearchResult:
    return SearchResult(
        document_id=doc_id,
        content=f"content for {doc_id}",
        metadata={},
        score=0.9,
        search_type="semantic",
        embedding=embedding,
    )


@pytest.fixture
def mock_embedding_manager():
    manager = MagicMock()
    # Plain list[float], matching what embeddings.py's real encode_single()
    # always returns (it calls .tolist() before returning).
    manager.encode_single.return_value = [0.1, 0.2, 0.3]
    return manager


@pytest.mark.unit
class TestMMRSearchHandlesNdarrayEmbeddings:
    def test_ndarray_embeddings_do_not_crash_mmr(self, mock_embedding_manager):
        """The exact repro: candidate results carrying raw numpy.ndarray
        embeddings (as SemanticSearcher used to produce before the fix)
        must not make MMRSearcher.search() fall back to un-diversified
        results via its except-and-swallow path.

        Asserting on result count/membership alone doesn't distinguish real
        MMR selection from the except block's `candidate_results[:k]`
        passthrough -- both can coincidentally produce the same k results
        from 3 candidates. Assert directly that the failure path was never
        hit instead.
        """
        candidates = [
            _make_result("doc1", np.array([0.1, 0.2, 0.3])),
            _make_result("doc2", np.array([0.4, 0.5, 0.6])),
            _make_result("doc3", np.array([0.7, 0.8, 0.9])),
        ]

        searcher = MMRSearcher(embedding_manager=mock_embedding_manager, config=OrionConfig())
        with patch("src.retrieval.search.log_error") as log_error:
            results = searcher.search(query="test query", candidate_results=candidates, k=2)

        log_error.assert_not_called()
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert {r.document_id for r in results}.issubset({"doc1", "doc2", "doc3"})

    def test_mixed_list_and_ndarray_embeddings_do_not_crash_mmr(self, mock_embedding_manager):
        """The realistic case: some candidates already have a cached
        embedding (possibly an ndarray), others need a fresh one from
        encode_batch() (always a plain list) -- both types must coexist
        safely in the same MMR pass."""
        mock_embedding_manager.encode_batch.return_value = [[0.2, 0.3, 0.4]]

        candidates = [
            _make_result("doc1", np.array([0.1, 0.2, 0.3])),  # cached ndarray
            _make_result("doc2", None),  # needs fresh encode_batch() -> list
        ]

        searcher = MMRSearcher(embedding_manager=mock_embedding_manager, config=OrionConfig())
        with patch("src.retrieval.search.log_error") as log_error:
            results = searcher.search(query="test query", candidate_results=candidates, k=2)

        log_error.assert_not_called()
        assert len(results) == 2
        assert {r.document_id for r in results} == {"doc1", "doc2"}

    def test_empty_embedding_is_filtered_out_not_treated_as_crash(self, mock_embedding_manager):
        """An empty embedding (list or ndarray) should be filtered out by
        the presence/emptiness check, same as before -- not a regression
        vector for the fix's `len(embedding) > 0` check."""
        candidates = [
            _make_result("doc1", np.array([0.1, 0.2, 0.3])),
            _make_result("doc2", np.array([])),  # empty -- should be skipped, not crash
        ]

        searcher = MMRSearcher(embedding_manager=mock_embedding_manager, config=OrionConfig())
        results = searcher.search(query="test query", candidate_results=candidates, k=2)

        assert len(results) == 1
        assert results[0].document_id == "doc1"
