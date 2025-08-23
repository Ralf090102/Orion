"""
Pydantic models for query API
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    model: str = "llama3.2:3b"
    conversation_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    include_sources: bool = True


class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str
    chunk_id: str


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    results: List[SearchResult]
    total_results: int
    conversation_id: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None


class ConversationHistory(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    title: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[SearchResult]] = None
