"""Regression test for the WebSocket chat handler swallowing failed-generation
answers.

generate_chat_response() catches LLM failures (e.g. a real Ollama error like
"model requires more system memory") internally and returns a normal
GenerationResult with a friendly `answer` describing what went wrong, rather
than raising -- found live when testing the actual Tauri window hit a real
memory-constrained Ollama failure. Since the failure happens before Ollama
streams anything, the WS handler's on_token callback is never invoked, so
before this fix `result.answer` was silently discarded: the client received
no "token" messages, just "metadata" then "done", leaving the assistant's
message bubble permanently empty with no indication anything went wrong --
this is what looked like "the UI doesn't update" from a real user's
perspective, not a rendering bug.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.websockets.chat import ChatWebSocketHandler


@dataclass
class FakeGenerationResult:
    answer: str
    sources: list = field(default_factory=list)
    query_type: str = "conversational"
    mode: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)
    rag_triggered: bool = False
    timing: Any = None


class FakeWebSocket:
    """Captures every message sent via handler.send_message(), skipping the
    real network/accept/receive plumbing this test doesn't need."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))


@pytest.fixture
def handler():
    generator = MagicMock()
    session_manager = MagicMock()
    session_manager.get_session.return_value = MagicMock()
    ws = FakeWebSocket()
    h = ChatWebSocketHandler(
        websocket=ws,
        session_id="test-session",
        session_manager=session_manager,
        generator=generator,
        config=MagicMock(rag=MagicMock(generation=MagicMock(rag_trigger_mode="auto"), llm=MagicMock(model="mistral:latest"))),
    )
    h.connected = True
    return h, ws, generator


@pytest.mark.unit
@pytest.mark.asyncio
class TestFailedGenerationStillReachesClient:
    async def test_llm_failure_forwards_friendly_answer_as_a_token(self, handler):
        h, ws, generator = handler
        friendly_error = "I encountered an error while generating the response: Ollama API error: model requires more system memory (4.5 GiB) than is available (2.4 GiB)"
        generator.generate_chat_response.return_value = FakeGenerationResult(
            answer=friendly_error,
            metadata={"error": "model requires more system memory", "llm_generation_failed": True},
        )

        await h.handle_user_message("What is machine learning?")

        token_messages = [m for m in ws.sent if m["type"] == "token"]
        assert len(token_messages) == 1, f"expected exactly one token message, got: {ws.sent}"
        assert token_messages[0]["content"] == friendly_error

        # The stream must still terminate normally so the frontend's
        # pending/loading state clears.
        assert any(m["type"] == "done" for m in ws.sent)

    async def test_successful_generation_does_not_get_a_duplicate_answer_message(self, handler):
        """The fallback must only fire when zero tokens were actually
        streamed -- a normal successful response already delivers its
        content via on_token and must not get result.answer appended too."""
        h, ws, generator = handler

        def fake_generate(*args, on_token=None, **kwargs):
            for chunk in ("Machine ", "learning is great."):
                on_token(chunk)
            return FakeGenerationResult(answer="Machine learning is great.", metadata={})

        generator.generate_chat_response.side_effect = fake_generate

        await h.handle_user_message("What is machine learning?")

        token_messages = [m for m in ws.sent if m["type"] == "token"]
        assert [m["content"] for m in token_messages] == ["Machine ", "learning is great."]
