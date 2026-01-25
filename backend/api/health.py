"""
Health and Status API Endpoints

Endpoints for system health checks, configuration, and status monitoring.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import (
    get_config_dependency,
    get_database_stats,
    get_retriever_dependency,
    get_session_manager_dependency,
)
from backend.models.ingestion import (
    FormatsResponse,
    KnowledgeBaseStats,
    StatusResponse,
)
from src.core.ingest import get_supported_formats
from src.core.llm import check_ollama_connection
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== BASIC HEALTH CHECK ==========
@router.get(
    "/health",
    summary="Basic health check",
    description="Quick health check endpoint to verify API is running",
    tags=["Health"],
)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Simple status message
    """
    return {
        "status": "healthy",
        "service": "orion-backend-api",
        "version": "1.0.0",
    }


# ========== DETAILED STATUS ==========
@router.get(
    "/api/status",
    response_model=StatusResponse,
    summary="System status",
    description="Comprehensive system status including KB stats, GPU, and Ollama",
    tags=["Health"],
)
async def get_status(
    config: OrionConfig = Depends(get_config_dependency),
    retriever: OrionRetriever = Depends(get_retriever_dependency),
):
    """
    Get comprehensive system status.
    
    Returns system information including:
    - Knowledge base statistics
    - GPU availability and details
    - Ollama service status
    - Current model configuration
    
    Args:
        config: Configuration instance (injected)
        retriever: Retriever instance (injected)
        
    Returns:
        StatusResponse with complete system information
    """
    try:
        # Get knowledge base stats from vector store
        kb_stats_raw = retriever.vector_store.get_collection_stats()
        
        kb_stats = KnowledgeBaseStats(
            total_chunks=kb_stats_raw.get("total_chunks", 0),
            unique_files=kb_stats_raw.get("unique_files", 0),
            collection_name=kb_stats_raw.get("collection_name", ""),
            persist_directory=kb_stats_raw.get("persist_directory", ""),
            file_type_distribution=kb_stats_raw.get("file_type_distribution", {}),
        )
        
    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        # Return empty stats if retriever not initialized
        kb_stats = KnowledgeBaseStats()
    
    # Check GPU availability
    gpu_available = False
    gpu_name = None
    if config.gpu.enabled:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            logger.warning("PyTorch not installed, GPU unavailable")
    
    # Check Ollama connection
    ollama_available = False
    if config.system.ollama_health_check:
        try:
            ollama_available = check_ollama_connection()
        except Exception as e:
            logger.warning(f"Failed to check Ollama connection: {e}")
    
    return StatusResponse(
        status="healthy" if ollama_available else "degraded",
        version=config.version,
        knowledge_base=kb_stats,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        ollama_available=ollama_available,
        embedding_model=config.rag.embedding.model,
        llm_model=config.rag.llm.model,
    )


# ========== CONFIGURATION ==========
@router.get(
    "/api/config",
    summary="Get configuration",
    description="Retrieve current system configuration",
    tags=["Health"],
)
async def get_config(
    config: OrionConfig = Depends(get_config_dependency),
    detailed: bool = False,
):
    """
    Get current system configuration.
    
    Args:
        config: Configuration instance (injected)
        detailed: If True, return full config. If False, return summary
        
    Returns:
        Configuration dictionary
    """
    if detailed:
        # Return full configuration
        return {
            "status": "success",
            "config": config.model_dump(),
        }
    
    # Return configuration summary
    return {
        "status": "success",
        "config": {
            "version": config.version,
            "embedding": {
                "model": config.rag.embedding.model,
                "batch_size": config.rag.embedding.batch_size,
            },
            "chunking": {
                "strategy": config.rag.chunking.strategy.value,
                "chunk_size": config.rag.chunking.chunk_size,
                "chunk_overlap": config.rag.chunking.chunk_overlap,
            },
            "retrieval": {
                "default_k": config.rag.retrieval.default_k,
                "enable_reranking": config.rag.retrieval.enable_reranking,
                "enable_hybrid_search": config.rag.retrieval.enable_hybrid_search,
                "enable_mmr": config.rag.retrieval.enable_mmr,
            },
            "llm": {
                "model": config.rag.llm.model,
                "base_url": config.rag.llm.base_url,
                "temperature": config.rag.llm.temperature,
            },
            "gpu": {
                "enabled": config.gpu.enabled,
                "preferred_device": config.gpu.preferred_device,
            },
            "generation": {
                "mode": config.rag.generation.mode,
                "rag_trigger_mode": config.rag.generation.rag_trigger_mode,
                "max_context_chunks": config.rag.generation.max_context_chunks,
            },
        },
    }


# ========== SUPPORTED FORMATS ==========
@router.get(
    "/api/formats",
    response_model=FormatsResponse,
    summary="Supported file formats",
    description="Get list of all supported file formats for ingestion",
    tags=["Health"],
)
async def get_formats():
    """
    Get supported file formats.
    
    Returns list of file extensions that can be ingested,
    grouped by category (Documents, Code, Data, Web, Config).
    
    Returns:
        FormatsResponse with all supported formats
    """
    try:
        # Get all supported formats from ingestion module
        supported = get_supported_formats()
        
        # Group by category
        categories = {
            "Documents": [".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".txt", ".md", ".rtf"],
            "Data": [".csv", ".json", ".xml", ".yaml", ".yml"],
            "Code": [
                ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
                ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
            ],
            "Web": [".html", ".css", ".scss", ".jsx", ".tsx", ".vue"],
            "Config": [".ini", ".conf", ".toml"],
        }
        
        formats_by_category = {}
        for category, extensions in categories.items():
            category_formats = [ext for ext in extensions if ext in supported]
            if category_formats:
                formats_by_category[category] = category_formats
        
        return FormatsResponse(
            total_formats=len(supported),
            formats_by_category=formats_by_category,
            all_formats=sorted(supported),
        )
        
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve supported formats: {str(e)}",
        )


# ========== DATABASE STATS ==========
@router.get(
    "/api/db/stats",
    summary="Database statistics",
    description="Get SQLite session database statistics",
    tags=["Health"],
)
async def get_db_stats():
    """
    Get session database statistics.
    
    Returns information about the SQLite session database,
    including total sessions, messages, tokens, and file size.
    
    Returns:
        Database statistics dictionary
    """
    try:
        stats = get_database_stats()
        return {
            "status": "success",
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve database stats: {str(e)}",
        )


# ========== READINESS CHECK ==========
@router.get(
    "/api/ready",
    summary="Readiness check",
    description="Check if all services are ready (Ollama, vector store, etc.)",
    tags=["Health"],
)
async def readiness_check(
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Readiness check for all services.
    
    Checks if the system is fully ready to handle requests:
    - Ollama connection
    - Vector store availability
    - Required models loaded
    
    Args:
        config: Configuration instance (injected)
        
    Returns:
        Readiness status and details
    """
    ready = True
    checks = {}
    
    # Check Ollama
    try:
        ollama_ok = check_ollama_connection()
        checks["ollama"] = "ready" if ollama_ok else "unavailable"
        if not ollama_ok and config.system.require_ollama:
            ready = False
    except Exception as e:
        checks["ollama"] = f"error: {str(e)}"
        if config.system.require_ollama:
            ready = False
    
    # Check vector store
    try:
        # Try to get a retriever instance (will lazy-load if needed)
        from backend.dependencies import get_retriever_dependency
        retriever = get_retriever_dependency(config)
        checks["vector_store"] = "ready"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"
        ready = False
    
    # Check session manager
    try:
        from backend.dependencies import get_session_manager_dependency
        session_manager = get_session_manager_dependency()
        checks["session_manager"] = "ready"
    except Exception as e:
        checks["session_manager"] = f"error: {str(e)}"
        # Session manager failure is not critical
    
    return {
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }


# ========== METRICS (OPTIONAL) ==========
@router.get(
    "/api/metrics",
    summary="System metrics",
    description="Get system metrics (requests, latency, etc.)",
    tags=["Health"],
)
async def get_metrics():
    """
    Get system metrics.
    
    Returns basic metrics about API usage and performance.
    Can be extended to support Prometheus format.
    
    Returns:
        Metrics dictionary
    """
    # TODO: Implement proper metrics tracking
    # For now, return placeholder
    return {
        "status": "success",
        "metrics": {
            "uptime_seconds": 0,  # Would track actual uptime
            "total_requests": 0,  # Would track request count
            "average_latency_ms": 0,  # Would track average latency
            "note": "Metrics tracking not yet implemented",
        },
    }
