"""
Pydantic models for RAG-related API endpoints.

Request/response models for query, ask, and RAG operations.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ========== SEARCH/QUERY MODELS ==========
class QueryRequest(BaseModel):
    """Request model for semantic query (retrieval only, no LLM).
    
    Separates required argument from optional settings (matches run.py query command).
    """

    # ===== REQUIRED ARGUMENT =====
    query: str = Field(
        ...,
        min_length=1,
        description="Search query string (required)",
        examples=["What is machine learning?"],
    )
    
    # ===== OPTIONAL SETTINGS (override config defaults) =====
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of results to return (uses config default if None)",
    )
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Enable cross-encoder reranking (uses config default if None)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (uses config default if None)",
    )
    verbose: Optional[bool] = Field(
        default=False,
        description="Return detailed timing and metadata",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("query cannot be empty")
        return stripped

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "k": None,  # Uses config default
                "enable_reranking": None,  # Uses config default
                "similarity_threshold": None,  # Uses config default
                "verbose": False,
            }
        }


class SearchResult(BaseModel):
    """Model for a single search result."""

    content: str = Field(..., description="Document chunk content")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata (source, file_name, etc.)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is a subset of artificial intelligence...",
                "score": 0.8542,
                "metadata": {
                    "source": "D:/Books/ml_basics.pdf",
                    "file_name": "ml_basics.pdf",
                    "page": 12,
                },
            }
        }


class QueryResponse(BaseModel):
    """Response model for query operation."""

    results: list[SearchResult] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results returned")
    query: str = Field(..., description="Original query string")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (reranking enabled, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "content": "Machine learning is a subset of AI...",
                        "score": 0.8542,
                        "metadata": {"source": "ml_basics.pdf", "page": 12},
                    }
                ],
                "total_results": 5,
                "query": "What is machine learning?",
                "processing_time": 0.234,
                "metadata": {
                    "reranking_enabled": True,
                    "hybrid_search": True,
                },
            }
        }


# ========== TIMING/PERFORMANCE MODELS ==========
class TimingBreakdown(BaseModel):
    """Model for detailed timing breakdown."""

    embedding_time: float = Field(default=0.0, description="Embedding generation time")
    search_time: float = Field(default=0.0, description="Vector search time")
    reranking_time: float = Field(default=0.0, description="Reranking time")
    mmr_time: float = Field(default=0.0, description="MMR selection time")
    context_preparation_time: float = Field(default=0.0, description="Context prep time")
    prompt_building_time: float = Field(default=0.0, description="Prompt building time")
    llm_generation_time: float = Field(default=0.0, description="LLM generation time")
    total_time: float = Field(default=0.0, description="Total processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "embedding_time": 0.123,
                "search_time": 0.045,
                "reranking_time": 0.234,
                "mmr_time": 0.012,
                "context_preparation_time": 0.008,
                "prompt_building_time": 0.015,
                "llm_generation_time": 1.904,
                "total_time": 2.341,
            }
        }


# ========== RAG ASK MODELS ==========
class AskRequest(BaseModel):
    """Request model for RAG ask (retrieval + LLM generation).
    
    Separates required argument from optional settings (matches run.py ask command).
    """

    # ===== REQUIRED ARGUMENT =====
    query: str = Field(
        ...,
        min_length=1,
        description="Question to answer using RAG (required)",
        examples=["Explain machine learning in simple terms"],
    )
    
    # ===== OPTIONAL SETTINGS (override config defaults) =====
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of context chunks to retrieve (uses config default if None)",
    )
    include_sources: Optional[bool] = Field(
        default=True,
        description="Include source citations in response",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature (uses config default if None)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens in response (uses config default if None)",
    )
    verbose: Optional[bool] = Field(
        default=False,
        description="Include detailed timing breakdown (matches run.py --verbose flag)",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("query cannot be empty")
        return stripped

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain machine learning in simple terms",
                "k": None,  # Uses config default
                "include_sources": True,
                "temperature": None,  # Uses config default
                "max_tokens": None,  # Uses config default
                "verbose": False,
            }
        }


class Source(BaseModel):
    """Model for a source citation."""

    index: int = Field(..., description="Citation index (e.g., [1], [2])")
    content: str = Field(..., description="Source content snippet")
    citation: str = Field(..., description="Formatted citation string")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "index": 1,
                "content": "Machine learning is a subset of artificial intelligence...",
                "citation": "ml_basics.pdf (page 12)",
                "score": 0.8542,
                "metadata": {
                    "file_name": "ml_basics.pdf",
                    "page": 12,
                },
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG ask operation."""

    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(
        default_factory=list,
        description="Source citations used in answer",
    )
    query: str = Field(..., description="Original query")
    query_type: str = Field(
        default="general",
        description="Detected query type (factual, analytical, procedural, etc.)",
    )
    processing_time: float = Field(..., description="Total processing time in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (num_contexts_used, rag_triggered, etc.)",
    )
    timing: Optional[TimingBreakdown] = Field(
        default=None,
        description="Detailed timing breakdown (only included when verbose=True)",
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. [1]",
                "sources": [
                    {
                        "index": 1,
                        "content": "Machine learning is a subset of AI...",
                        "citation": "ml_basics.pdf (page 12)",
                        "score": 0.8542,
                        "metadata": {"file_name": "ml_basics.pdf", "page": 12},
                    }
                ],
                "query": "What is machine learning?",
                "query_type": "factual",
                "processing_time": 2.341,
                "metadata": {
                    "num_contexts_used": 5,
                    "rag_retrieval_triggered": True,
                    "model": "mistral:latest",
                },
                "timestamp": "2026-01-25T14:45:00",
            }
        }


# ========== STREAMING MODELS ==========
class StreamChunk(BaseModel):
    """Model for a single streaming chunk."""

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
        description="Data payload (for source/metadata chunks)",
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
                    "type": "source",
                    "content": None,
                    "data": {
                        "index": 1,
                        "citation": "ml_basics.pdf (page 12)",
                    },
                },
                {
                    "type": "done",
                    "content": None,
                    "data": {"processing_time": 2.34},
                },
            ]
        }


class DetailedAskResponse(AskResponse):
    """Extended response with detailed timing information."""

    timing: TimingBreakdown = Field(
        default_factory=TimingBreakdown,
        description="Detailed timing breakdown",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is...",
                "sources": [],
                "query": "What is ML?",
                "query_type": "factual",
                "processing_time": 2.341,
                "metadata": {},
                "timestamp": "2026-01-25T14:45:00",
                "timing": {
                    "embedding_time": 0.123,
                    "search_time": 0.045,
                    "reranking_time": 0.234,
                    "llm_generation_time": 1.904,
                    "total_time": 2.341,
                },
            }
        }
