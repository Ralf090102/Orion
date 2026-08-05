"""Regression test for OrionRetriever.query()'s error-handling contract.

Before this fix, query()'s except ValueError/except Exception blocks always
returned a plain error *string*, ignoring the caller's formatted=False
request -- even though the method's own docstring documents "Raises:
ValueError: If knowledge base is empty...". Every real caller in the
codebase (run.py's CLI commands, /api/query, /api/ask/stream,
AnswerGenerator.generate_rag_response()/generate_chat_response()) calls
with formatted=False and wraps the call in its own try/except, expecting
an exception -- so the swallowed exception silently handed back a string
where a list of SearchResult was expected, and callers crashed later
trying to treat the string's individual characters as SearchResult
objects (e.g. AttributeError: 'str' object has no attribute 'score').

Found via live testing against an empty knowledge base (a fresh
ORION_DATA_DIR with nothing ingested yet -- exactly what a first
real-world run looks like).
"""

from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig


@pytest.fixture
def empty_kb_retriever():
    """A retriever whose vector store reports zero documents, without
    touching any real embedding model, reranker, or ChromaDB instance."""
    retriever = OrionRetriever(config=OrionConfig())
    retriever._initialized = True
    retriever._vector_store = MagicMock()
    retriever._vector_store.get_collection_stats.return_value = {"document_count": 0}
    return retriever


@pytest.mark.unit
class TestQueryErrorHandlingRespectsFormattedFlag:
    def test_formatted_false_raises_instead_of_returning_a_string(self, empty_kb_retriever):
        """This is the shape every real caller (API, CLI, AnswerGenerator)
        already expects -- see the module docstring."""
        with pytest.raises(ValueError, match="No documents found"):
            empty_kb_retriever.query("test", formatted=False)

    def test_formatted_false_with_return_timing_also_raises(self, empty_kb_retriever):
        """return_timing=True changes the *success* return shape to a tuple,
        but must not change error behavior."""
        with pytest.raises(ValueError, match="No documents found"):
            empty_kb_retriever.query("test", formatted=False, return_timing=True)

    def test_formatted_true_still_returns_a_friendly_string(self, empty_kb_retriever):
        """The CLI's `formatted=True` display mode never wraps query() in
        its own try/except -- it still needs the old swallow-and-stringify
        behavior, unchanged."""
        result = empty_kb_retriever.query("test", formatted=True)
        assert isinstance(result, str)
        assert "No documents found" in result

    def test_formatted_true_with_return_timing_still_returns_string_and_timing(self, empty_kb_retriever):
        result, timing = empty_kb_retriever.query("test", formatted=True, return_timing=True)
        assert isinstance(result, str)
        assert timing.total_time >= 0
