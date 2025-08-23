"""
System Management API Endpoints
"""

from fastapi import APIRouter, HTTPException

from backend.services.system_service import SystemService

router = APIRouter()
system_service = SystemService()


@router.get("/health")
async def system_health():
    """
    Get system health status
    """
    try:
        health = await system_service.get_health_status()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def system_stats():
    """
    Get system statistics
    """
    try:
        stats = await system_service.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """
    List available LLM models
    """
    try:
        models = await system_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear system cache
    """
    try:
        await system_service.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(lines: int = 100):
    """
    Get recent system logs
    """
    try:
        logs = await system_service.get_recent_logs(lines)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
