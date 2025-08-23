"""
Pydantic models for ingestion API
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class IngestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    folder_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    async_processing: bool = True
    force_rebuild: bool = False


class IngestResponse(BaseModel):
    task_id: Optional[str] = None
    status: IngestStatus
    message: str
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None
    progress: Optional[float] = None
    error: Optional[str] = None
