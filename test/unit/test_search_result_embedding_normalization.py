"""SearchResult.__init__ is the single seam that normalizes embedding types.

Every consumer of SearchResult.embedding (currently only MMRSearcher.search())
trusts that it's always list[float] or None, never a raw numpy.ndarray --
regardless of what type was actually passed in at construction (ChromaDB
query results return numpy.ndarray; embeddings.py's encode_single()/
encode_batch() already return plain lists). See
test_mmr_ndarray_embeddings.py for the regression this guarantee replaces.
"""

import numpy as np
import pytest

from src.retrieval.search import SearchResult


def _make_result(embedding) -> SearchResult:
    return SearchResult(
        document_id="doc1",
        content="content",
        metadata={},
        score=0.9,
        search_type="semantic",
        embedding=embedding,
    )


@pytest.mark.unit
class TestSearchResultNormalizesEmbedding:
    def test_ndarray_embedding_becomes_plain_list_of_native_floats(self):
        result = _make_result(np.array([0.1, 0.2, 0.3]))

        assert isinstance(result.embedding, list)
        assert result.embedding == [0.1, 0.2, 0.3]
        assert all(type(x) is float for x in result.embedding)

    def test_empty_ndarray_embedding_becomes_empty_list_not_none(self):
        result = _make_result(np.array([]))

        assert result.embedding == []
        assert result.embedding is not None
        # Still falsy, so consumers' `if embedding:` filters correctly.
        assert not result.embedding

    def test_none_embedding_stays_none(self):
        result = _make_result(None)

        assert result.embedding is None

    def test_plain_list_embedding_passes_through_unchanged(self):
        original = [0.1, 0.2, 0.3]
        result = _make_result(original)

        assert isinstance(result.embedding, list)
        assert result.embedding == original
