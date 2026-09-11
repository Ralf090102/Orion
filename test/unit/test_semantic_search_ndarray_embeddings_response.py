"""Regression test for SemanticSearcher.search() crashing on ChromaDB's raw
ndarray-typed `embeddings` response field.

Found live (2026-09-11) while smoke-testing retrieval after re-ingesting a
knowledge base: every real query hit "Semantic search failed: The truth
value of an array with more than one element is ambiguous", silently
degrading to zero semantic results (search()'s own except-Exception-return-[]
handler swallows it, so HybridSearcher just falls back to keyword-only
matches -- no crash, but a real, permanent loss of retrieval quality on
every single query, not just an edge case).

Root cause: `if results.get("embeddings") and results["embeddings"][0]:`
does a bare truthiness check on ChromaDB's raw response value for
"embeddings" -- which, per this project's pinned chromadb version, comes
back as a numpy.ndarray, not a plain list. This is the same bug *class* as
the ndarray-truthiness issue already fixed in SearchResult.__init__ and
MMRSearcher.search() (see test_mmr_ndarray_embeddings.py and
test_search_result_embedding_normalization.py, 2026-09-11) -- but at a
different, upstream site: checking Chroma's raw response dict directly,
before any value ever reaches SearchResult.__init__ (which only normalizes
values it's actually given).

Fixed by checking presence/emptiness explicitly (`is not None and len(...) >
0`) instead of bare truthiness, at both the outer ("is there an embeddings
list at all") and inner ("does the first query's result set have any
embeddings") level.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval.search import SemanticSearcher
from src.utilities.config import OrionConfig


def _chroma_query_response(n: int = 2) -> dict:
    """Shape ChromaVectorStore.query() actually returns when
    include=["documents", "metadatas", "distances", "embeddings"] --
    embeddings as a numpy.ndarray, exactly as the pinned chromadb version
    does, not a plain list."""
    return {
        "ids": [[f"doc{i}" for i in range(n)]],
        "documents": [[f"content {i}" for i in range(n)]],
        "metadatas": [[{} for _ in range(n)]],
        "distances": [[0.1 * i for i in range(n)]],
        "embeddings": [np.array([[0.1, 0.2, 0.3] for _ in range(n)])],
    }


@pytest.mark.unit
class TestSemanticSearchHandlesNdarrayEmbeddingsResponse:
    def test_ndarray_embeddings_response_does_not_crash_search(self):
        embedding_manager = MagicMock()
        embedding_manager.encode_single.return_value = [0.1, 0.2, 0.3]

        vector_store = MagicMock()
        vector_store.query.return_value = _chroma_query_response(n=2)

        searcher = SemanticSearcher(embedding_manager, vector_store, config=OrionConfig())
        results = searcher.search("test query", k=2)

        assert len(results) == 2
        assert all(r.embedding is not None for r in results)

    def test_missing_embeddings_key_still_works(self):
        """include=[...] might omit "embeddings" entirely -- must not crash
        either, and results should just have no cached embedding."""
        embedding_manager = MagicMock()
        embedding_manager.encode_single.return_value = [0.1, 0.2, 0.3]

        response = _chroma_query_response(n=1)
        del response["embeddings"]

        vector_store = MagicMock()
        vector_store.query.return_value = response

        searcher = SemanticSearcher(embedding_manager, vector_store, config=OrionConfig())
        results = searcher.search("test query", k=1)

        assert len(results) == 1
        assert results[0].embedding is None
