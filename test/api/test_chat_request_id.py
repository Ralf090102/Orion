"""Feature test: the chat REST endpoints (non-streaming and SSE) mint a
correlation ID per request and expose it as metadata.request_id in their
response payloads -- part of the 2026-09-11 /llm-council per-query
correlation-ID decision (see src/utilities/request_context.py, Eru's
Orion-Roadmap.md). Follows test_rag_stream.py's pattern: a minimal fake
generator implementing only the interface the endpoint calls, swapped in
via dependency_overrides on top of the shared `client` fixture.
"""

import pytest

from backend.dependencies import get_generator_dependency, get_session_manager_dependency
from src.generation.generate import GenerationResult


class _FakeChatGenerator:
    """Captures the request_id it was called with, so tests can assert the
    endpoint actually passed one through (not just that it minted one for
    its own response payload)."""

    def __init__(self):
        self.last_request_id = None

    def generate_chat_response(self, *args, **kwargs):
        self.last_request_id = kwargs.get("request_id")
        return GenerationResult(
            answer="Hello!",
            sources=[],
            query_type="conversational",
            mode="chat",
            metadata={},
            rag_triggered=False,
        )


class _FakeSessionManagerWithSession:
    def get_session(self, session_id):
        return {"session_id": session_id}  # truthy is all send_message()/event_generator() need


@pytest.fixture
def chat_client(client):
    """Reuses the shared `client` fixture's app/overrides, swapping the
    generator and session_manager for ones the chat endpoints need."""
    from backend.app import app

    generator = _FakeChatGenerator()
    app.dependency_overrides[get_generator_dependency] = lambda: generator
    app.dependency_overrides[get_session_manager_dependency] = lambda: _FakeSessionManagerWithSession()
    return client, generator


@pytest.mark.unit
def test_send_message_response_includes_a_request_id(chat_client):
    client, generator = chat_client

    response = client.post("/api/chat/sessions/s1/message", json={"message": "Hi"})

    assert response.status_code == 200
    request_id = response.json()["metadata"]["request_id"]
    assert isinstance(request_id, str) and len(request_id) > 0
    assert generator.last_request_id == request_id


@pytest.mark.unit
def test_send_message_stream_metadata_chunk_includes_a_request_id(chat_client):
    client, generator = chat_client

    response = client.post("/api/chat/sessions/s1/stream", json={"message": "Hi"})

    assert response.status_code == 200
    body = response.text
    assert '"request_id"' in body
    # generator.last_request_id was set inside the same request; a non-empty
    # value proves the endpoint minted one and actually passed it through,
    # not just echoed a hardcoded placeholder.
    assert generator.last_request_id
    assert generator.last_request_id in body


@pytest.mark.unit
def test_two_separate_requests_get_different_request_ids(chat_client):
    client, generator = chat_client

    r1 = client.post("/api/chat/sessions/s1/message", json={"message": "Hi"})
    r2 = client.post("/api/chat/sessions/s1/message", json={"message": "Hi again"})

    assert r1.json()["metadata"]["request_id"] != r2.json()["metadata"]["request_id"]
