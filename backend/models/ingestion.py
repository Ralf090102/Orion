"""
Pydantic models for ingestion-related API endpoints.

Request/response models for document ingestion, knowledge base management,
and file format information.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ========== INGESTION REQUEST/RESPONSE MODELS ==========
class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    path: str = Field(
        ...,
        description="Path to file or directory to ingest",
        examples=["D:/Documents/Books", "./data/knowledge_base"],
    )
    clear_existing: bool = Field(
        default=False,
        description="Clear existing knowledge base before ingestion",
    )
    recursive: bool = Field(
        default=True,
        description="Recursively scan subdirectories",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path is not empty."""
        if not v or not v.strip():
            raise ValueError("path cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "path": "D:/Documents/Books",
                "clear_existing": False,
                "recursive": True,
            }
        }


class IngestionStats(BaseModel):
    """Statistics from ingestion operation."""

    total_files: int = Field(default=0, description="Total files processed")
    successful_files: int = Field(default=0, description="Successfully ingested files")
    failed_files: int = Field(default=0, description="Failed file ingestions")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    total_chunks: int = Field(default=0, description="Total chunks created")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    errors: list[str] = Field(default_factory=list, description="List of error messages")

    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 42,
                "successful_files": 40,
                "failed_files": 2,
                "success_rate": 95.2,
                "total_chunks": 1523,
                "processing_time": 12.45,
                "errors": [
                    "Failed to process file.pdf: Invalid format",
                    "Failed to process doc.docx: File corrupted",
                ],
            }
        }


class IngestResponse(BaseModel):
    """Response model for ingestion operation."""

    status: str = Field(
        ...,
        description="Status of ingestion operation",
        examples=["success", "partial", "failed"],
    )
    message: str = Field(
        ...,
        description="Human-readable status message",
        examples=["Successfully ingested 42 files"],
    )
    stats: IngestionStats = Field(..., description="Detailed ingestion statistics")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of operation completion",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Successfully ingested 42 files into knowledge base",
                "stats": {
                    "total_files": 42,
                    "successful_files": 42,
                    "failed_files": 0,
                    "success_rate": 100.0,
                    "total_chunks": 1523,
                    "processing_time": 12.45,
                    "errors": [],
                },
                "timestamp": "2026-01-25T14:30:00",
            }
        }


# ========== BACKGROUND TASK MODELS ==========
class IngestionTask(BaseModel):
    """Model for tracking background ingestion tasks."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(
        ...,
        description="Current task status",
        examples=["pending", "running", "completed", "failed"],
    )
    path: str = Field(..., description="Path being ingested")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Task start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Task completion timestamp",
    )
    stats: Optional[IngestionStats] = Field(
        default=None,
        description="Statistics (available when completed)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-789",
                "status": "running",
                "path": "D:/Documents/Books",
                "progress": 67.5,
                "started_at": "2026-01-25T14:28:00",
                "completed_at": None,
                "stats": None,
                "error": None,
            }
        }


class IngestTaskResponse(BaseModel):
    """Response when starting a background ingestion task."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Initial task status (usually 'pending')")
    message: str = Field(..., description="Status message")
    check_status_url: str = Field(
        ...,
        description="URL to check task status",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-789",
                "status": "pending",
                "message": "Ingestion task started",
                "check_status_url": "/api/ingest/status/abc123-def456-789",
            }
        }


# ========== CLEAR/DELETE MODELS ==========
class ClearRequest(BaseModel):
    """Request model for clearing knowledge base."""

    confirm: bool = Field(
        default=False,
        description="Confirmation flag (must be true to proceed)",
    )

    @field_validator("confirm")
    @classmethod
    def validate_confirm(cls, v: bool) -> bool:
        """Require explicit confirmation."""
        if not v:
            raise ValueError("confirm must be true to clear knowledge base")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "confirm": True,
            }
        }


class ClearResponse(BaseModel):
    """Response model for clear operation."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Knowledge base cleared successfully",
                "timestamp": "2026-01-25T14:35:00",
            }
        }


# ========== STATUS/INFO MODELS ==========
class KnowledgeBaseStats(BaseModel):
    """Statistics about the knowledge base."""

    total_chunks: int = Field(default=0, description="Total document chunks")
    unique_files: int = Field(default=0, description="Number of unique source files")
    collection_name: str = Field(default="", description="Vector store collection name")
    persist_directory: str = Field(default="", description="Storage directory path")
    file_type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of file types (extension: count)",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_chunks": 1523,
                "unique_files": 42,
                "collection_name": "orion_knowledge_base",
                "persist_directory": "./data/chroma-data",
                "file_type_distribution": {
                    "pdf": 25,
                    "md": 10,
                    "txt": 5,
                    "py": 2,
                },
                "last_updated": "2026-01-25T14:30:00",
            }
        }


class StatusResponse(BaseModel):
    """Response model for system status endpoint."""

    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Orion version")
    knowledge_base: KnowledgeBaseStats = Field(..., description="KB statistics")
    gpu_available: bool = Field(default=False, description="GPU availability")
    gpu_name: Optional[str] = Field(default=None, description="GPU device name")
    ollama_available: bool = Field(default=False, description="Ollama service status")
    embedding_model: str = Field(default="", description="Current embedding model")
    llm_model: str = Field(default="", description="Current LLM model")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "knowledge_base": {
                    "total_chunks": 1523,
                    "unique_files": 42,
                    "collection_name": "orion_knowledge_base",
                    "persist_directory": "./data/chroma-data",
                    "file_type_distribution": {"pdf": 25, "md": 10},
                    "last_updated": "2026-01-25T14:30:00",
                },
                "gpu_available": True,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "ollama_available": True,
                "embedding_model": "all-MiniLM-L12-v2",
                "llm_model": "mistral:latest",
            }
        }


# ========== FILE FORMATS MODELS ==========
class FileFormat(BaseModel):
    """Model for a supported file format."""

    extension: str = Field(..., description="File extension (e.g., '.pdf')")
    category: str = Field(
        ...,
        description="Category (Documents, Code, Data, Web, Config)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the format",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "extension": ".pdf",
                "category": "Documents",
                "description": "Portable Document Format",
            }
        }


class FormatsResponse(BaseModel):
    """Response model for supported formats endpoint."""

    total_formats: int = Field(..., description="Total number of supported formats")
    formats_by_category: dict[str, list[str]] = Field(
        ...,
        description="Formats grouped by category",
    )
    all_formats: list[str] = Field(..., description="Flat list of all extensions")

    class Config:
        json_schema_extra = {
            "example": {
                "total_formats": 35,
                "formats_by_category": {
                    "Documents": [".pdf", ".docx", ".txt", ".md"],
                    "Code": [".py", ".js", ".java", ".cpp"],
                    "Data": [".csv", ".json", ".xml"],
                },
                "all_formats": [".pdf", ".docx", ".txt", ".md", ".py", ".js"],
            }
        }


# ========== ERROR MODELS ==========
class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type or category")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "path cannot be empty",
                "status_code": 422,
                "timestamp": "2026-01-25T14:40:00",
            }
        }
