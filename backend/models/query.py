"""
Pydantic models for query API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class QueryIntent(str, Enum):
    """Types of query intents"""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    TROUBLESHOOTING = "troubleshooting"
    CREATIVE = "creative"


class QueryRequest(BaseModel):
    """Request model for document queries"""

    query: str = Field(
        ...,
        description="The question or search query",
        min_length=1,
        max_length=2000,
        example="What are the main findings in the research paper?",
    )

    k: int = Field(default=5, description="Number of relevant documents to retrieve", ge=1, le=20, example=5)

    model: str = Field(default="llama3.2:3b", description="LLM model to use for response generation", example="mistral:latest")

    conversation_id: Optional[str] = Field(default=None, description="ID for conversation continuity", example="conv_abc123")

    user_id: Optional[str] = Field(default=None, description="User ID for workspace isolation", example="user123")

    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters for document search", example={"file_type": "pdf", "category": "research"}
    )

    include_sources: bool = Field(default=True, description="Whether to include source documents in response", example=True)

    use_enhancement: bool = Field(
        default=True, description="Whether to use query enhancement (HyDE, expansion, etc.)", example=True
    )

    temperature: float = Field(
        default=0.7, description="Creativity level for LLM response (0.0-1.0)", ge=0.0, le=1.0, example=0.7
    )

    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response", ge=1, le=4096, example=1000)


class SearchResult(BaseModel):
    """Individual search result from document retrieval"""

    content: str = Field(
        ...,
        description="Text content of the document chunk",
        example="This section discusses the implementation of neural networks...",
    )

    metadata: Dict[str, Any] = Field(
        ...,
        description="Document metadata (source, page, etc.)",
        example={"source": "research_paper.pdf", "page": 3, "chunk_id": "chunk_15"},
    )

    score: float = Field(..., description="Relevance score (0.0-1.0, higher is more relevant)", ge=0.0, le=1.0, example=0.8523)

    document_id: str = Field(..., description="Unique identifier for the source document", example="doc_abc123")

    chunk_id: str = Field(..., description="Unique identifier for this text chunk", example="chunk_15")

    highlighted_content: Optional[str] = Field(
        default=None,
        description="Content with search terms highlighted",
        example="This section discusses the <mark>implementation</mark> of neural networks...",
    )


class QueryAnalysis(BaseModel):
    """Analysis of the query for better processing"""

    intent: QueryIntent = Field(..., description="Detected intent of the query")

    keywords: List[str] = Field(
        default=[], description="Extracted keywords from the query", example=["neural networks", "implementation", "training"]
    )

    complexity_score: float = Field(default=0.0, description="Query complexity score (0.0-1.0)", ge=0.0, le=1.0, example=0.6)

    requires_context: bool = Field(default=False, description="Whether query requires conversation context", example=True)

    sub_queries: Optional[List[str]] = Field(
        default=None,
        description="Generated sub-queries for complex questions",
        example=["What are neural networks?", "How to implement neural networks?"],
    )


class QueryResponse(BaseModel):
    """Response model for document queries"""

    query: str = Field(..., description="Original query that was processed", example="What are the main findings?")

    answer: Optional[str] = Field(
        default=None, description="Generated answer from the LLM", example="Based on the documents, the main findings are..."
    )

    results: List[SearchResult] = Field(default=[], description="List of relevant document chunks")

    total_results: int = Field(default=0, description="Total number of results found", ge=0, example=5)

    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for follow-up queries", example="conv_abc123"
    )

    model_used: Optional[str] = Field(
        default=None, description="LLM model that generated the response", example="mistral:latest"
    )

    processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds", ge=0.0, example=2.45)

    analysis: Optional[QueryAnalysis] = Field(default=None, description="Analysis of the query processing")

    confidence_score: Optional[float] = Field(
        default=None, description="Confidence in the generated answer (0.0-1.0)", ge=0.0, le=1.0, example=0.85
    )

    cached: bool = Field(default=False, description="Whether the response was served from cache", example=False)

    warnings: Optional[List[str]] = Field(
        default=None, description="Any warnings during processing", example=["Low confidence in answer due to limited context"]
    )


class ConversationHistory(BaseModel):
    """Conversation metadata and history"""

    conversation_id: str = Field(..., description="Unique conversation identifier", example="conv_abc123")

    user_id: Optional[str] = Field(default=None, description="User who owns this conversation", example="user123")

    created_at: datetime = Field(..., description="When the conversation was started")

    updated_at: datetime = Field(..., description="When the conversation was last updated")

    message_count: int = Field(..., description="Number of messages in conversation", ge=0, example=10)

    title: Optional[str] = Field(
        default=None,
        description="Auto-generated or user-provided title",
        max_length=200,
        example="Discussion about Neural Networks",
    )

    tags: Optional[List[str]] = Field(
        default=None, description="Conversation tags for organization", example=["ai", "research", "implementation"]
    )

    is_active: bool = Field(default=True, description="Whether conversation is active", example=True)


class ChatMessage(BaseModel):
    """Individual message in a conversation"""

    role: str = Field(..., description="Message sender role", pattern="^(user|assistant)$", example="user")

    content: str = Field(
        ..., description="Message content", min_length=1, max_length=10000, example="Can you explain how neural networks work?"
    )

    timestamp: datetime = Field(..., description="When the message was sent")

    sources: Optional[List[SearchResult]] = Field(default=None, description="Source documents used for this message")

    message_id: Optional[str] = Field(default=None, description="Unique message identifier", example="msg_xyz789")

    model_used: Optional[str] = Field(
        default=None, description="LLM model used (for assistant messages)", example="mistral:latest"
    )

    processing_time: Optional[float] = Field(default=None, description="Time taken to generate response", ge=0.0, example=1.23)

    confidence: Optional[float] = Field(
        default=None, description="Confidence in the response (0.0-1.0)", ge=0.0, le=1.0, example=0.92
    )
