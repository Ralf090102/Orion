"""Feature test: ChatWebSocketHandler mints a correlation ID per user
message and includes it in the "metadata" message sent to the client, and
passes that same id through to generate_chat_response() as request_id --
part of the 2026-09-11 /llm-council per-query correlation-ID decision (see
src/utilities/request_context.py, Eru's Orion-Roadmap.md).

Follows test_websocket_chat_error_handling.py's pattern: a FakeWebSocket
capturing sent messages, a MagicMock generator/session_manager, driving
ChatWebSocketHandler.handle_user_message() directly rather than through a
real network connection.
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
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))


@pytest.fixture
def handler():
    generator = MagicMock()
    generator.generate_chat_response.return_value = FakeGenerationResult(answer="hi", metadata={})
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
class TestWebSocketChatRequestId:
    async def test_metadata_message_includes_a_request_id(self, handler):
        h, ws, generator = handler

        await h.handle_user_message("What is machine learning?")

        metadata_messages = [m for m in ws.sent if m["type"] == "metadata"]
        assert len(metadata_messages) == 1
        request_id = metadata_messages[0]["data"].get("request_id")
        assert isinstance(request_id, str) and len(request_id) > 0

    async def test_generator_receives_the_same_request_id_sent_to_the_client(self, handler):
        h, ws, generator = handler

        await h.handle_user_message("What is machine learning?")

        sent_request_id = [m for m in ws.sent if m["type"] == "metadata"][0]["data"]["request_id"]
        _, call_kwargs = generator.generate_chat_response.call_args
        assert call_kwargs.get("request_id") == sent_request_id

    async def test_two_messages_on_the_same_connection_get_different_request_ids(self, handler):
        h, ws, generator = handler

        await h.handle_user_message("First message")
        await h.handle_user_message("Second message")

        metadata_messages = [m for m in ws.sent if m["type"] == "metadata"]
        request_ids = [m["data"]["request_id"] for m in metadata_messages]
        assert len(request_ids) == 2
        assert request_ids[0] != request_ids[1]
