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
    WatchdogResponse,
    WatchdogStartRequest,
    WatchdogStatusResponse,
    WatchdogStopRequest,
)
from src.core.ingest import clear_knowledge_base, ingest_documents
from src.retrieval.watchdog import FileWatcher
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== TASK TRACKER (IN-MEMORY) ==========
# In production, use Redis or database for persistence
_ingestion_tasks: Dict[str, IngestionTask] = {}

# ========== WATCHDOG MANAGER (SINGLETON) ==========
_file_watcher: Optional[FileWatcher] = None


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


# ========== WATCHDOG ENDPOINTS ==========
@router.post(
    "/api/watchdog/start",
    response_model=WatchdogResponse,
    summary="Start file watcher",
    description="Start watching directories for file changes and auto-ingest",
    tags=["Watchdog"],
)
async def start_watchdog(
    request: WatchdogStartRequest,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Start the file watcher for automatic ingestion.
    
    The watcher monitors specified directories and automatically ingests
    new or modified files into the knowledge base.
    
    Args:
        request: Watchdog start request with paths and options
        config: Configuration instance (injected)
        
    Returns:
        WatchdogResponse with status and watcher info
        
    Raises:
        HTTPException: If watcher is already running or start fails
    """
    global _file_watcher
    
    # Check if already watching
    if _file_watcher and _file_watcher.is_watching():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File watcher is already running. Stop it first before starting a new one.",
        )
    
    # Validate paths exist
    valid_paths = []
    for path_str in request.paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Path does not exist: {path_str}")
            continue
        if not path.is_dir():
            logger.warning(f"Path is not a directory: {path_str}")
            continue
        valid_paths.append(str(path.resolve()))
    
    if not valid_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid directories to watch. All paths must exist and be directories.",
        )
    
    try:
        from src.core.ingest import DocumentIngestor
        from src.retrieval.watchdog import create_file_watcher
        
        logger.info(f"Starting file watcher for {len(valid_paths)} paths")
        
        # Create document ingestor
        ingestor = DocumentIngestor(config=config)
        
        # Create callbacks for file events
        def handle_file_change(file_path: str):
            """Handle file addition or modification"""
            logger.info(f"Ingesting file from watchdog: {file_path}")
            try:
                success, metadata, errors = ingestor.ingest_file(file_path)
                if success:
                    logger.info(f"Watchdog ingestion successful: {file_path}")
                    # Reset retriever to pick up new documents
                    reset_retriever()
                else:
                    logger.error(f"Watchdog ingestion failed: {file_path} - {errors}")
            except Exception as e:
                logger.error(f"Error during watchdog ingestion: {e}", exc_info=True)
        
        # Update watchdog config with request parameters
        config.watchdog.paths = valid_paths
        config.watchdog.recursive = request.recursive
        config.watchdog.debounce_seconds = request.debounce_seconds
        
        # Create and start watcher
        _file_watcher = create_file_watcher(
            vector_store=ingestor.vector_store,
            on_file_added=handle_file_change,
            on_file_modified=handle_file_change,
            config=config,
        )
        
        success = _file_watcher.start(paths=valid_paths)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start file watcher",
            )
        
        # Get watcher status
        watcher_info = _file_watcher.get_status()
        status_response = WatchdogStatusResponse(**watcher_info)
        
        logger.info(f"File watcher started successfully for {len(valid_paths)} paths")
        
        return WatchdogResponse(
            status="success",
            message=f"File watcher started for {len(valid_paths)} directory(ies)",
            watcher_status=status_response,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start watchdog: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start file watcher: {str(e)}",
        )


@router.post(
    "/api/watchdog/stop",
    response_model=WatchdogResponse,
    summary="Stop file watcher",
    description="Stop watching all paths or a specific path",
    tags=["Watchdog"],
)
async def stop_watchdog(
    request: WatchdogStopRequest = WatchdogStopRequest(),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Stop the file watcher.
    
    Can stop watching all paths or just a specific path:
    - No path or path="all": Stop entire watcher
    - path="<specific_path>": Stop watching that path only
    
    Args:
        request: Stop request with optional path
        config: Configuration instance (injected)
        
    Returns:
        WatchdogResponse with status
        
    Raises:
        HTTPException: If watcher is not running or stop fails
    """
    global _file_watcher
    
    if not _file_watcher or not _file_watcher.is_watching():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File watcher is not running",
        )
    
    try:
        stop_path = request.path
        
        # Stop all paths or entire watcher
        if not stop_path or stop_path.lower() == "all":
            logger.info("Stopping file watcher for all paths")
            
            success = _file_watcher.stop()
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to stop file watcher",
                )
            
            logger.info("File watcher stopped successfully")
            
            return WatchdogResponse(
                status="success",
                message="File watcher stopped for all paths",
                watcher_status=None,
            )
        
        # Stop watching a specific path
        else:
            # Normalize the path
            path_to_remove = str(Path(stop_path).resolve())
            
            # Get current watched paths
            current_paths = _file_watcher.get_watched_paths()
            
            # Check if path is being watched
            if path_to_remove not in current_paths:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Path is not being watched: {stop_path}",
                )
            
            # Remove the path from watched paths
            remaining_paths = [p for p in current_paths if p != path_to_remove]
            
            logger.info(f"Stopping watcher for path: {path_to_remove}")
            
            # If no paths remain, stop the entire watcher
            if not remaining_paths:
                success = _file_watcher.stop()
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to stop file watcher",
                    )
                
                logger.info("Stopped watcher for last path - watcher fully stopped")
                
                return WatchdogResponse(
                    status="success",
                    message=f"Stopped watching {stop_path} (last path - watcher stopped)",
                    watcher_status=None,
                )
            
            # Restart watcher with remaining paths
            # Get current config
            current_status = _file_watcher.get_status()
            debounce = current_status["debounce_seconds"]
            recursive = current_status["recursive"]
            
            # Stop current watcher
            _file_watcher.stop()
            
            # Recreate watcher with remaining paths
            from src.core.ingest import DocumentIngestor
            from src.retrieval.watchdog import create_file_watcher
            
            ingestor = DocumentIngestor(config=config)
            
            def handle_file_change(file_path: str):
                """Handle file addition or modification"""
                logger.info(f"Ingesting file from watchdog: {file_path}")
                try:
                    success, metadata, errors = ingestor.ingest_file(file_path)
                    if success:
                        logger.info(f"Watchdog ingestion successful: {file_path}")
                        reset_retriever()
                    else:
                        logger.error(f"Watchdog ingestion failed: {file_path} - {errors}")
                except Exception as e:
                    logger.error(f"Error during watchdog ingestion: {e}", exc_info=True)
            
            # Update config
            config.watchdog.paths = remaining_paths
            config.watchdog.recursive = recursive
            config.watchdog.debounce_seconds = debounce
            
            # Create new watcher
            _file_watcher = create_file_watcher(
                vector_store=ingestor.vector_store,
                on_file_added=handle_file_change,
                on_file_modified=handle_file_change,
                config=config,
            )
            
            # Start with remaining paths
            success = _file_watcher.start(paths=remaining_paths)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to restart watcher with remaining paths",
                )
            
            # Get updated status
            watcher_info = _file_watcher.get_status()
            status_response = WatchdogStatusResponse(**watcher_info)
            
            logger.info(f"Stopped watching {path_to_remove}, now watching {len(remaining_paths)} path(s)")
            
            return WatchdogResponse(
                status="success",
                message=f"Stopped watching {stop_path}. Still watching {len(remaining_paths)} path(s).",
                watcher_status=status_response,
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop watchdog: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop file watcher: {str(e)}",
        )


@router.get(
    "/api/watchdog/status",
    response_model=WatchdogStatusResponse,
    summary="Get watchdog status",
    description="Check if file watcher is running and get configuration",
    tags=["Watchdog"],
)
async def get_watchdog_status():
    """
    Get file watcher status.
    
    Returns:
        WatchdogStatusResponse with current status and configuration
    """
    global _file_watcher
    
    if not _file_watcher:
        # Watcher never created
        return WatchdogStatusResponse(
            is_watching=False,
            watched_paths=[],
            path_count=0,
            debounce_seconds=1.0,
            recursive=True,
            ignore_patterns=[],
            max_workers=2,
        )
    
    # Get status from watcher
    watcher_info = _file_watcher.get_status()
    return WatchdogStatusResponse(**watcher_info)
