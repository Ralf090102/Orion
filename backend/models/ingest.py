"""
Pydantic models for ingestion API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class IngestStatus(str, Enum):
    """Status of document ingestion process"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Request model for document ingestion"""

    folder_path: str = Field(..., description="Path to folder containing documents to ingest", example="/path/to/documents")

    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for document splitting",
        ge=100,  # minimum 100 characters
        le=4000,  # maximum 4000 characters
        example=1000,
    )

    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks", ge=0, le=1000, example=200)

    async_processing: bool = Field(
        default=True, description="Whether to process documents asynchronously in background", example=True
    )

    force_rebuild: bool = Field(
        default=False, description="Force rebuild of vectorstore even if documents haven't changed", example=False
    )

    user_id: Optional[str] = Field(default=None, description="User ID for workspace isolation", example="user123")

    include_images: bool = Field(default=True, description="Whether to extract text from images using OCR", example=True)

    extract_tables: bool = Field(default=True, description="Whether to extract tables from PDFs", example=True)


class IngestProgress(BaseModel):
    """Progress information for ingestion process"""

    current_file: Optional[str] = Field(default=None, description="Currently processing file", example="document.pdf")

    files_processed: int = Field(default=0, description="Number of files processed", ge=0, example=5)

    total_files: int = Field(default=0, description="Total number of files to process", ge=0, example=10)

    chunks_created: int = Field(default=0, description="Number of text chunks created", ge=0, example=150)

    percentage_complete: float = Field(
        default=0.0, description="Percentage of completion (0-100)", ge=0.0, le=100.0, example=50.0
    )

    estimated_time_remaining: Optional[float] = Field(
        default=None, description="Estimated time remaining in seconds", ge=0.0, example=120.5
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion"""

    task_id: Optional[str] = Field(
        default=None, description="Unique task identifier for async processing", example="task_abc123"
    )

    status: IngestStatus = Field(..., description="Current status of the ingestion process")

    message: str = Field(..., description="Human-readable status message", example="Successfully ingested 10 documents")

    document_count: Optional[int] = Field(default=None, description="Number of documents processed", ge=0, example=10)

    chunk_count: Optional[int] = Field(default=None, description="Total number of text chunks created", ge=0, example=250)

    progress: Optional[IngestProgress] = Field(default=None, description="Detailed progress information")

    error: Optional[str] = Field(
        default=None, description="Error message if ingestion failed", example="Failed to read file: permission denied"
    )

    processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds", ge=0.0, example=45.2)

    file_types_processed: Optional[List[str]] = Field(
        default=None, description="List of file extensions processed", example=[".pdf", ".docx", ".txt"]
    )

    warnings: Optional[List[str]] = Field(
        default=None, description="Non-fatal warnings during processing", example=["Skipped corrupted file: broken.pdf"]
    )
