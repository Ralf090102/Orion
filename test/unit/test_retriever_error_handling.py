"""Regression test for OrionRetriever.query()'s error-handling contract.

Before an earlier fix, query()'s except ValueError/except Exception blocks
always returned a plain error *string*, ignoring the caller's formatted=False
request -- even though the method's own docstring documents "Raises:
ValueError: If knowledge base is empty...". Every real caller in the
codebase (run.py's CLI commands, /api/query, /api/ask/stream,
AnswerGenerator.generate_rag_response()/generate_chat_response()) called
with formatted=False and wrapped the call in its own try/except, expecting
an exception -- so the swallowed exception silently handed back a string
where a list of SearchResult was expected, and callers crashed later
trying to treat the string's individual characters as SearchResult
objects (e.g. AttributeError: 'str' object has no attribute 'score').

query()'s formatted/return_timing flags were later removed entirely
(architecture-review candidate #6, 2026-09-11): since every real caller
already used formatted=False, the formatted=True string-swallowing path had
zero live callers left, so query() now always returns
(list[SearchResult], TimingBreakdown) on success and always raises on
error -- the contract these tests already assumed for every real caller.
Formatting for display (e.g. a CLI) is now a separate, explicitly-called
OrionRetriever.format_results() method, not part of query()'s own contract.

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
class TestQueryErrorHandlingAlwaysRaises:
    def test_empty_knowledge_base_raises_value_error(self, empty_kb_retriever):
        """This is the shape every real caller (API, CLI, AnswerGenerator)
        already expects -- see the module docstring."""
        with pytest.raises(ValueError, match="No documents found"):
            empty_kb_retriever.query("test")
