"""Regression test: a failed RAG retrieval in chat mode must be visible in
GenerationResult.metadata, not silently indistinguishable from "RAG wasn't
needed for this message."

Found live (2026-09-11) diagnosing a report of the installed app answering
from outside knowledge instead of citing a known-ingested file: the real
root cause was an empty knowledge base after a reinstall, but
generate_chat_response()'s retrieval try/except (src/generation/generate.py)
only did `logger.warning(...)` and continued -- so the resulting
"No documents found in knowledge base" ValueError left zero trace anywhere
a caller could see. The chat answer looked completely normal, making an
empty/misconfigured knowledge base indistinguishable from RAG legitimately
not triggering, which is exactly what made this bug hard to diagnose.

Chat mode's graceful-degradation design (keep answering conversationally
without context on a retrieval failure) is correct and unchanged here --
this only makes the failure visible in metadata, matching the
llm_generation_failed pattern already used elsewhere in this file.
"""

from unittest.mock import MagicMock

import pytest

from src.generation.generate import AnswerGenerator
from src.utilities.config import OrionConfig


@pytest.fixture
def generator():
    config = OrionConfig()
    # "always" so should_retrieve_rag() actually attempts retrieval for any
    # message -- the whole point of this test is exercising that attempt.
    config.rag.generation.rag_trigger_mode = "always"
    gen = AnswerGenerator(config)
    # Simulate the real failure mode found live: an empty/misconfigured
    # knowledge base raises ValueError from OrionRetriever.query().
    gen.retriever = MagicMock()
    gen.retriever.query.side_effect = ValueError(
        "No documents found in knowledge base. Please run ingestion first."
    )
    # Avoid a real Ollama call -- this test is about retrieval-failure
    # visibility, not LLM generation.
    gen.llm_client = MagicMock()
    gen.llm_client.generate.return_value = {
        "message": {"content": "I don't have specific information on that, but generally speaking..."}
    }
    return gen


@pytest.mark.unit
class TestRagRetrievalFailureIsVisibleInMetadata:
    def test_retrieval_failure_sets_metadata_flag_and_error(self, generator):
        result = generator.generate_chat_response(message="What is the somatosensory system?")

        assert result.metadata.get("rag_retrieval_triggered") is True
        assert result.metadata.get("rag_retrieval_failed") is True
        assert "No documents found in knowledge base" in result.metadata.get("rag_retrieval_error", "")

    def test_chat_still_answers_conversationally_despite_the_failure(self, generator):
        """Graceful degradation must survive this fix unchanged -- chat mode
        should never turn a retrieval failure into a hard error for the
        user, unlike explicit RAG mode."""
        result = generator.generate_chat_response(message="What is the somatosensory system?")

        assert result.answer == "I don't have specific information on that, but generally speaking..."

    def test_successful_retrieval_leaves_the_flag_false(self, generator):
        """Sanity check the flag isn't just always True -- only a real
        failure sets it."""
        generator.retriever.query.side_effect = None
        generator.retriever.query.return_value = ([], MagicMock(embedding_time=0, search_time=0, reranking_time=0, mmr_time=0))

        result = generator.generate_chat_response(message="Hello?")

        assert result.metadata.get("rag_retrieval_failed") is False
        assert "rag_retrieval_error" not in result.metadata
