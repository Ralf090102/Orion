"""
Pydantic models for chat-related API endpoints.

Request/response models for chat sessions, messages, and conversational AI.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ========== SESSION MODELS ==========
class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session."""

    session_id: Optional[str] = Field(
        default=None,
        description="Optional custom session ID (auto-generated if None)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional session metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": None,
                "metadata": {
                    "user": "john_doe",
                    "topic": "machine_learning",
                },
            }
        }


class SessionInfo(BaseModel):
    """Model for session information."""

    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")
    message_count: int = Field(default=0, description="Number of messages in session")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                "created_at": "2026-01-25T14:00:00",
                "updated_at": "2026-01-25T14:30:00",
                "message_count": 12,
                "metadata": {
                    "user": "john_doe",
                    "topic": "machine_learning",
                },
            }
        }


class SessionResponse(BaseModel):
    """Response model for session operations."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    session: SessionInfo = Field(..., description="Session information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Session created successfully",
                "session": {
                    "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                    "created_at": "2026-01-25T14:00:00",
                    "updated_at": "2026-01-25T14:00:00",
                    "message_count": 0,
                    "metadata": {},
                },
            }
        }


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: list[SessionInfo] = Field(..., description="List of sessions")
    total: int = Field(..., description="Total number of sessions")

    class Config:
        json_schema_extra = {
            "example": {
                "sessions": [
                    {
                        "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                        "created_at": "2026-01-25T14:00:00",
                        "updated_at": "2026-01-25T14:30:00",
                        "message_count": 12,
                        "metadata": {},
                    }
                ],
                "total": 1,
            }
        }


class DeleteSessionResponse(BaseModel):
    """Response model for session deletion."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    session_id: str = Field(..., description="Deleted session ID")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Session deleted successfully",
                "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
            }
        }


# ========== MESSAGE MODELS ==========
class Message(BaseModel):
    """Model for a chat message."""

    role: str = Field(
        ...,
        description="Message role (user or assistant)",
        pattern="^(user|assistant)$",
    )
    content: str = Field(..., min_length=1, description="Message content")
    tokens: int = Field(default=0, description="Token count (estimated)")
    timestamp: str = Field(..., description="Message timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is machine learning?",
                "tokens": 5,
                "timestamp": "2026-01-25T14:30:00",
            }
        }


class ChatRequest(BaseModel):
    """Request model for sending a chat message."""

    message: str = Field(
        ...,
        min_length=1,
        description="User message content",
        examples=["What is machine learning?"],
    )
    rag_mode: Optional[str] = Field(
        default=None,
        description="RAG trigger mode: always/auto/manual/never (uses config default if None)",
        pattern="^(always|auto|manual|never)?$",
    )
    include_sources: bool = Field(
        default=False,
        description="Include source citations when RAG is triggered",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0-2.0, uses config default if None)",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is not empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be empty")
        return stripped

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is machine learning?",
                "rag_mode": "auto",
                "include_sources": True,
                "temperature": 0.7,
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat message."""

    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="Assistant's response")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source citations (if RAG triggered and sources enabled)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (rag_triggered, query_type, etc.)",
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                "message": "Machine learning is a subset of AI that enables computers to learn from data. [1]",
                "sources": [
                    {
                        "index": 1,
                        "citation": "ml_basics.pdf (page 12)",
                        "content": "Machine learning is...",
                    }
                ],
                "metadata": {
                    "rag_retrieval_triggered": True,
                    "query_type": "factual",
                    "model": "mistral:latest",
                },
                "processing_time": 1.234,
                "timestamp": "2026-01-25T14:35:00",
            }
        }


class MessageHistoryResponse(BaseModel):
    """Response model for retrieving message history."""

    session_id: str = Field(..., description="Session identifier")
    messages: list[Message] = Field(..., description="List of messages")
    total_messages: int = Field(..., description="Total number of messages")
    total_tokens: int = Field(default=0, description="Total tokens across all messages")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is ML?",
                        "tokens": 4,
                        "timestamp": "2026-01-25T14:30:00",
                    },
                    {
                        "role": "assistant",
                        "content": "Machine learning is...",
                        "tokens": 25,
                        "timestamp": "2026-01-25T14:30:05",
                    },
                ],
                "total_messages": 2,
                "total_tokens": 29,
            }
        }


# ========== STREAMING MODELS ==========
class ChatStreamChunk(BaseModel):
    """Model for streaming chat chunks."""

    type: str = Field(
        ...,
        description="Chunk type (token, source, metadata, done)",
        examples=["token", "source", "metadata", "done"],
    )
    content: Optional[str] = Field(
        default=None,
        description="Content (for token chunks)",
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Data payload (for source/metadata/done chunks)",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "token",
                    "content": "Machine",
                    "data": None,
                },
                {
                    "type": "metadata",
                    "content": None,
                    "data": {
                        "rag_triggered": True,
                        "query_type": "factual",
                    },
                },
                {
                    "type": "done",
                    "content": None,
                    "data": {
                        "processing_time": 1.234,
                        "total_tokens": 45,
                    },
                },
            ]
        }


# ========== WEBSOCKET MODELS ==========
class WebSocketMessage(BaseModel):
    """Model for WebSocket messages (both incoming and outgoing)."""

    type: str = Field(
        ...,
        description="Message type (message, error, ping, pong)",
        examples=["message", "error", "ping"],
    )
    content: Optional[str] = Field(
        default=None,
        description="Message content (for 'message' type)",
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional data",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "message",
                    "content": "What is machine learning?",
                    "data": {
                        "rag_mode": "auto",
                        "include_sources": True,
                    },
                },
                {
                    "type": "error",
                    "content": "Session not found",
                    "data": {"code": 404},
                },
            ]
        }


# ========== CONVERSATION SUMMARY MODELS ==========
class ConversationSummary(BaseModel):
    """Model for conversation summary."""

    session_id: str = Field(..., description="Session identifier")
    total_messages: int = Field(..., description="Total number of messages")
    user_messages: int = Field(..., description="Number of user messages")
    assistant_messages: int = Field(..., description="Number of assistant messages")
    total_tokens: int = Field(default=0, description="Total tokens used")
    duration: Optional[float] = Field(
        default=None,
        description="Conversation duration in seconds",
    )
    rag_triggers: int = Field(
        default=0,
        description="Number of times RAG was triggered",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "85c619ca-cd1e-4567-89ab-cdef01234567",
                "total_messages": 12,
                "user_messages": 6,
                "assistant_messages": 6,
                "total_tokens": 342,
                "duration": 1800.0,
                "rag_triggers": 3,
            }
        }
