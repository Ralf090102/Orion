"""
Ingestion API Endpoints

Endpoints for document ingestion, knowledge base management, and file processing.
"""

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from backend.dependencies import get_config_dependency, reset_retriever
from backend.models.ingestion import (
    ClearRequest,
    ClearResponse,
    IngestRequest,
    IngestResponse,
    IngestTaskResponse,
    IngestionStats,
    IngestionTask,
)
from src.core.ingest import clear_knowledge_base, ingest_documents
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== TASK TRACKER (IN-MEMORY) ==========
# In production, use Redis or database for persistence
_ingestion_tasks: Dict[str, IngestionTask] = {}


def _create_task(path: str) -> str:
    """Create a new ingestion task and return its ID."""
    task_id = str(uuid.uuid4())
    task = IngestionTask(
        task_id=task_id,
        status="pending",
        path=path,
        progress=0.0,
        started_at=None,
        completed_at=None,
        stats=None,
        error=None,
    )
    _ingestion_tasks[task_id] = task
    logger.info(f"Created ingestion task: {task_id}")
    return task_id


def _update_task(task_id: str, **kwargs):
    """Update task fields."""
    if task_id in _ingestion_tasks:
        for key, value in kwargs.items():
            if hasattr(_ingestion_tasks[task_id], key):
                setattr(_ingestion_tasks[task_id], key, value)


def _background_ingestion(
    task_id: str, path: str, clear_existing: bool, config: OrionConfig
):
    """
    Background task for document ingestion.
    
    Args:
        task_id: Task identifier
        path: Path to ingest
        clear_existing: Clear KB before ingestion
        config: Configuration instance
    """
    try:
        # Update task status
        _update_task(
            task_id,
            status="running",
            started_at=datetime.now(),
            progress=10.0,
        )
        
        logger.info(f"Starting ingestion task {task_id}: {path}")
        
        # Perform ingestion
        stats = ingest_documents(
            source_path=path,
            config=config,
            clear_existing=clear_existing,
        )
        
        # Convert stats to IngestionStats model
        ingestion_stats = IngestionStats(
            total_files=stats.total_files,
            successful_files=stats.successful_files,
            failed_files=stats.failed_files,
            success_rate=stats.success_rate,
            total_chunks=stats.total_chunks,
            processing_time=0.0,  # Will be calculated by caller
            errors=stats.errors[:10] if stats.errors else [],  # Limit to 10 errors
        )
        
        # Update task as completed
        _update_task(
            task_id,
            status="completed",
            completed_at=datetime.now(),
            progress=100.0,
            stats=ingestion_stats,
        )
        
        # Reset retriever to pick up new documents
        reset_retriever()
        
        logger.info(f"Completed ingestion task {task_id}: {stats.successful_files}/{stats.total_files} files")
        
    except Exception as e:
        logger.error(f"Ingestion task {task_id} failed: {e}", exc_info=True)
        _update_task(
            task_id,
            status="failed",
            completed_at=datetime.now(),
            progress=0.0,
            error=str(e),
        )


# ========== SYNCHRONOUS INGESTION ==========
@router.post(
    "/api/ingest",
    response_model=IngestResponse,
    summary="Ingest documents",
    description="Ingest documents into knowledge base (synchronous)",
    tags=["Ingestion"],
)
async def ingest_sync(
    request: IngestRequest,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Ingest documents synchronously.
    
    This endpoint blocks until ingestion completes.
    For large datasets, consider using /api/ingest/async instead.
    
    Args:
        request: Ingestion request with path and options
        config: Configuration instance (injected)
        
    Returns:
        IngestResponse with statistics
        
    Raises:
        HTTPException: If path doesn't exist or ingestion fails
    """
    # Validate path exists
    path_obj = Path(request.path)
    if not path_obj.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path not found: {request.path}",
        )
    
    try:
        start_time = time.time()
        
        logger.info(f"Starting synchronous ingestion: {request.path}")
        
        # Perform ingestion
        stats = ingest_documents(
            source_path=request.path,
            config=config,
            clear_existing=request.clear_existing,
        )
        
        processing_time = time.time() - start_time
        
        # Convert to response model
        ingestion_stats = IngestionStats(
            total_files=stats.total_files,
            successful_files=stats.successful_files,
            failed_files=stats.failed_files,
            success_rate=stats.success_rate,
            total_chunks=stats.total_chunks,
            processing_time=processing_time,
            errors=stats.errors[:10] if stats.errors else [],
        )
        
        # Reset retriever to pick up new documents
        reset_retriever()
        
        # Determine status
        if stats.failed_files == 0:
            status_str = "success"
            message = f"Successfully ingested {stats.successful_files} file(s)"
        elif stats.successful_files > 0:
            status_str = "partial"
            message = f"Partially completed: {stats.successful_files} succeeded, {stats.failed_files} failed"
        else:
            status_str = "failed"
            message = f"Ingestion failed: {stats.failed_files} file(s) failed"
        
        logger.info(f"Ingestion complete: {message}")
        
        return IngestResponse(
            status=status_str,
            message=message,
            stats=ingestion_stats,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Synchronous ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


# ========== ASYNCHRONOUS INGESTION ==========
@router.post(
    "/api/ingest/async",
    response_model=IngestTaskResponse,
    summary="Ingest documents (async)",
    description="Ingest documents in background (for large datasets)",
    tags=["Ingestion"],
)
async def ingest_async(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Ingest documents asynchronously.
    
    Returns immediately with a task ID. Use /api/ingest/status/{task_id}
    to check progress.
    
    Args:
        request: Ingestion request with path and options
        background_tasks: FastAPI background tasks
        config: Configuration instance (injected)
        
    Returns:
        IngestTaskResponse with task ID and status URL
        
    Raises:
        HTTPException: If path doesn't exist
    """
    # Validate path exists
    path_obj = Path(request.path)
    if not path_obj.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path not found: {request.path}",
        )
    
    # Create task
    task_id = _create_task(request.path)
    
    # Schedule background ingestion
    background_tasks.add_task(
        _background_ingestion,
        task_id=task_id,
        path=request.path,
        clear_existing=request.clear_existing,
        config=config,
    )
    
    logger.info(f"Scheduled async ingestion task {task_id}: {request.path}")
    
    return IngestTaskResponse(
        task_id=task_id,
        status="pending",
        message="Ingestion task started",
        check_status_url=f"/api/ingest/status/{task_id}",
    )


# ========== TASK STATUS ==========
@router.get(
    "/api/ingest/status/{task_id}",
    response_model=IngestionTask,
    summary="Get ingestion task status",
    description="Check the status of a background ingestion task",
    tags=["Ingestion"],
)
async def get_task_status(task_id: str):
    """
    Get ingestion task status.
    
    Args:
        task_id: Task identifier from /api/ingest/async
        
    Returns:
        IngestionTask with current status and progress
        
    Raises:
        HTTPException: If task not found
    """
    if task_id not in _ingestion_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )
    
    return _ingestion_tasks[task_id]


# ========== LIST TASKS ==========
@router.get(
    "/api/ingest/tasks",
    summary="List ingestion tasks",
    description="List all ingestion tasks (active and completed)",
    tags=["Ingestion"],
)
async def list_tasks():
    """
    List all ingestion tasks.
    
    Returns:
        List of all tasks with their status
    """
    return {
        "total_tasks": len(_ingestion_tasks),
        "tasks": list(_ingestion_tasks.values()),
    }


# ========== CLEAR KNOWLEDGE BASE ==========
@router.post(
    "/api/ingest/clear",
    response_model=ClearResponse,
    summary="Clear knowledge base",
    description="Delete all documents and embeddings from knowledge base",
    tags=["Ingestion"],
)
async def clear_kb(
    request: ClearRequest,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Clear the knowledge base.
    
    **Warning:** This permanently deletes all ingested documents and embeddings!
    
    Args:
        request: Clear request (must have confirm=True)
        config: Configuration instance (injected)
        
    Returns:
        ClearResponse with status
        
    Raises:
        HTTPException: If clear operation fails
    """
    try:
        logger.warning("Clearing knowledge base...")
        
        success = clear_knowledge_base(config=config)
        
        if success:
            # Reset retriever to reflect empty KB
            reset_retriever()
            
            logger.info("Knowledge base cleared successfully")
            return ClearResponse(
                status="success",
                message="Knowledge base cleared successfully",
                timestamp=datetime.now(),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear knowledge base",
            )
            
    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}",
        )


# ========== DELETE TASK ==========
@router.delete(
    "/api/ingest/tasks/{task_id}",
    summary="Delete task record",
    description="Remove a task from the task list",
    tags=["Ingestion"],
)
async def delete_task(task_id: str):
    """
    Delete a task record.
    
    This only removes the task from the tracking list,
    it doesn't affect the ingested documents.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If task not found
    """
    if task_id not in _ingestion_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )
    
    del _ingestion_tasks[task_id]
    
    return {
        "status": "success",
        "message": f"Task {task_id} deleted",
    }
