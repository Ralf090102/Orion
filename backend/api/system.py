"""
System Management API Endpoints
Enhanced with configuration management, profiles, and GPU acceleration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from backend.services.system_service import SystemService
from backend.services import get_config_service, get_gpu_manager
from backend.models.system import (
    SystemHealth,
    SystemStats,
    SystemConfig,
    SystemConfigUpdateRequest,
    ProfileInfo,
)

router = APIRouter()


def get_system_service() -> SystemService:
    """Dependency to get system service instance"""
    return SystemService()


def get_configuration_service():
    """Dependency to get configuration service instance"""
    return get_config_service()


@router.get("/health", response_model=SystemHealth)
async def system_health(system_service: SystemService = Depends(get_system_service)):
    """Get system health status"""
    try:
        health = await system_service.get_health_status()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=SystemStats)
async def system_stats(system_service: SystemService = Depends(get_system_service)):
    """Get system statistics"""
    try:
        stats = await system_service.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(system_service: SystemService = Depends(get_system_service)):
    """List available LLM models"""
    try:
        models = await system_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(system_service: SystemService = Depends(get_system_service)):
    """Clear system cache"""
    try:
        await system_service.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(lines: int = 100, system_service: SystemService = Depends(get_system_service)):
    """Get recent system logs"""
    try:
        logs = await system_service.get_recent_logs(lines)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New Configuration Management Endpoints


@router.get("/config", response_model=SystemConfig)
async def get_system_config(config_service=Depends(get_configuration_service)):
    """Get current system configuration"""
    try:
        return config_service.get_system_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=SystemConfig)
async def update_system_config(update_request: SystemConfigUpdateRequest, config_service=Depends(get_configuration_service)):
    """Update system configuration"""
    try:
        return config_service.update_system_config(update_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Profile Management Endpoints


@router.get("/profiles", response_model=List[ProfileInfo])
async def list_profiles(config_service=Depends(get_configuration_service)):
    """List all available profiles"""
    try:
        return config_service.list_profiles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/active", response_model=ProfileInfo)
async def get_active_profile(config_service=Depends(get_configuration_service)):
    """Get currently active profile"""
    try:
        active_profile = config_service.get_active_profile()
        if not active_profile:
            raise HTTPException(status_code=404, detail="No active profile found")
        return active_profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles/{profile_name}/activate", response_model=ProfileInfo)
async def activate_profile(profile_name: str, config_service=Depends(get_configuration_service)):
    """Activate a specific profile"""
    try:
        return config_service.activate_profile(profile_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=ProfileInfo)
async def create_profile(name: str, display_name: str, config_service=Depends(get_configuration_service)):
    """Create a new profile"""
    try:
        return config_service.create_profile(name, display_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GPU Management Endpoints


@router.get("/gpu/capabilities")
async def get_gpu_capabilities():
    """Get GPU capabilities and status"""
    try:
        gpu_manager = get_gpu_manager()
        capabilities = gpu_manager.detect_gpu_capabilities()

        return {
            "has_cuda": capabilities.has_cuda,
            "cuda_version": capabilities.cuda_version,
            "device_count": len(capabilities.gpus),
            "total_memory_mb": capabilities.total_gpu_memory_mb,
            "recommended_device": capabilities.recommended_device,
            "gpus": [
                {
                    "device_id": gpu.device_id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memory_total_mb,
                    "memory_used_mb": gpu.memory_used_mb,
                    "memory_free_mb": gpu.memory_free_mb,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_c": gpu.temperature_c,
                }
                for gpu in capabilities.gpus
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu/optimal-memory")
async def get_optimal_gpu_memory(device_id: str = None):
    """Get optimal GPU memory limit for specified device"""
    try:
        gpu_manager = get_gpu_manager()
        optimal_limit = gpu_manager.get_optimal_memory_limit(device_id)

        if optimal_limit is None:
            return {"message": "No GPU available", "optimal_limit_mb": None}

        return {"optimal_limit_mb": optimal_limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
