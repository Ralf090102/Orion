"""Regression/feature test: generate_chat_response() mints (or accepts) a
correlation ID for every chat turn, exposes it in GenerationResult.metadata,
and assembles + persists a structured QueryTrace describing what the
retrieval pipeline actually did.

Added as part of the 2026-09-11 /llm-council decision (see Eru's
Orion-Roadmap.md): a generic log stream wouldn't have caught the
2026-09-11 "not citing my knowledge base" incident, since that failure
lived in Python retrieval logic (fusion/rerank/what-landed-in-the-prompt),
not anywhere a plain "RAG triggered" log line would show. This is the
artifact that would have.
"""

from unittest.mock import MagicMock

import pytest

from src.generation.generate import AnswerGenerator
from src.generation.trace import QueryTrace
from src.retrieval.search import SearchResult
from src.utilities.config import OrionConfig


def _timing_mock():
    return MagicMock(embedding_time=0.01, search_time=0.02, reranking_time=0.01, mmr_time=0.005)


@pytest.fixture
def generator():
    config = OrionConfig()
    config.rag.generation.rag_trigger_mode = "always"
    gen = AnswerGenerator(config)
    gen.retriever = MagicMock()
    gen.retriever.query.return_value = (
        [
            SearchResult(document_id="doc-1", content="chunk one", metadata={}, score=0.9, search_type="hybrid"),
        ],
        _timing_mock(),
    )
    gen.llm_client = MagicMock()
    gen.llm_client.generate.return_value = {"message": {"content": "Here's the answer."}}
    return gen


@pytest.mark.unit
class TestRequestIdInMetadata:
    def test_auto_mints_a_request_id_when_none_passed(self, generator):
        result = generator.generate_chat_response(message="What is X?")

        request_id = result.metadata.get("request_id")
        assert isinstance(request_id, str) and len(request_id) > 0

    def test_reuses_the_caller_supplied_request_id(self, generator):
        result = generator.generate_chat_response(message="What is X?", request_id="my-fixed-id")

        assert result.metadata["request_id"] == "my-fixed-id"


@pytest.mark.unit
class TestQueryTracePersistence:
    def test_trace_is_saved_once_with_populated_stages_on_success(self, generator):
        session_manager = MagicMock()
        session_manager.get_session.return_value = MagicMock()
        session_manager.get_messages.return_value = []
        session_manager.add_message.return_value = "msg-id"

        generator.generate_chat_response(
            message="What is X?",
            session_id="s1",
            session_manager=session_manager,
            request_id="req-123",
        )

        session_manager.save_query_trace.assert_called_once()
        (trace,), _ = session_manager.save_query_trace.call_args
        assert isinstance(trace, QueryTrace)
        assert trace.request_id == "req-123"
        assert trace.rag_retrieval_triggered is True
        assert trace.model is not None
        assert trace.timing is not None

    def test_trace_stages_are_empty_but_present_when_retrieval_not_triggered(self, generator):
        session_manager = MagicMock()
        session_manager.get_session.return_value = MagicMock()
        session_manager.get_messages.return_value = []
        session_manager.add_message.return_value = "msg-id"

        generator.generate_chat_response(
            message="Hello",
            session_id="s1",
            session_manager=session_manager,
            rag_mode="never",
            request_id="req-456",
        )

        (trace,), _ = session_manager.save_query_trace.call_args
        assert trace.rag_retrieval_triggered is False
        assert trace.retrieved == []
        assert trace.fused == []
        assert trace.context_chunks == []
        generator.retriever.query.assert_not_called()

    def test_assistant_message_id_reuses_the_request_id(self, generator):
        session_manager = MagicMock()
        session_manager.get_session.return_value = MagicMock()
        session_manager.get_messages.return_value = []
        session_manager.add_message.return_value = "msg-id"

        generator.generate_chat_response(
            message="What is X?",
            session_id="s1",
            session_manager=session_manager,
            request_id="req-789",
        )

        assistant_call = [
            call for call in session_manager.add_message.call_args_list
            if call.kwargs.get("role") == "assistant"
        ]
        assert len(assistant_call) == 1
        assert assistant_call[0].kwargs.get("message_id") == "req-789"
