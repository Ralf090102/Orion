"""
Pydantic models for Orion backend API

This module contains all the data models used by the FastAPI backend for
request/response validation, serialization, and API documentation.

Models are organized by functionality:
- Ingestion: Document processing and ingestion models
- Query: Search and chat functionality models
- System: System health, configuration, and monitoring models
"""

# Ingestion models
from .ingest import (
    IngestStatus,
    IngestRequest,
    IngestResponse,
    IngestProgress,
)

# Query models
from .query import (
    QueryIntent,
    QueryRequest,
    QueryResponse,
    QueryAnalysis,
    SearchResult,
    ConversationHistory,
    ChatMessage,
)

# System models
from .system import (
    SystemStatus,
    SystemHealth,
    SystemStats,
    ServiceStatus,
    ModelInfo,
    ModelsResponse,
    ConfigUpdate,
    ConfigResponse,
)

__all__ = [
    # Ingestion
    "IngestStatus",
    "IngestRequest",
    "IngestResponse",
    "IngestProgress",
    # Query
    "QueryIntent",
    "QueryRequest",
    "QueryResponse",
    "QueryAnalysis",
    "SearchResult",
    "ConversationHistory",
    "ChatMessage",
    # System
    "SystemStatus",
    "SystemHealth",
    "SystemStats",
    "ServiceStatus",
    "ModelInfo",
    "ModelsResponse",
    "ConfigUpdate",
    "ConfigResponse",
]
