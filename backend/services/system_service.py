"""
System management service
"""

from typing import Dict, Any, List
import psutil
import platform
from datetime import datetime

from core.rag.llm import get_available_models, check_ollama_connection
from core.utils.caching import get_global_cache_stats, clear_global_cache
from core.utils.orion_utils import log_info


class SystemService:
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status
        """
        try:
            ollama_status = (
                "connected" if await check_ollama_connection() else "disconnected"
            )
        except Exception as e:
            ollama_status = "error"
            log_info(f"Error checking Ollama connection: {e}")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "ollama": ollama_status,
                "vectorstore": "ready",  # TODO: Check vectorstore
                "api": "running",
            },
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": (
                    psutil.disk_usage("/").percent
                    if platform.system() != "Windows"
                    else psutil.disk_usage("C:").percent
                ),
            },
        }

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get detailed system statistics
        """
        cache_stats = get_global_cache_stats()

        return {
            "cache": cache_stats,
            "vectorstore": {
                "document_count": 0,  # TODO: Get actual count
                "chunk_count": 0,
                "index_size_mb": 0,
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 2
                ),
                "disk_free_gb": round(
                    psutil.disk_usage(
                        "C:" if platform.system() == "Windows" else "/"
                    ).free
                    / (1024**3),
                    2,
                ),
            },
        }

    async def get_available_models(self) -> List[str]:
        """
        Get list of available LLM models
        """
        try:
            models = await get_available_models()
            return models
        except Exception as e:
            log_info(f"Error getting models: {e}")
            return []

    async def clear_cache(self):
        """
        Clear system caches
        """
        clear_global_cache()
        log_info("System cache cleared")

    async def get_recent_logs(self, lines: int = 100) -> List[str]:
        """
        Get recent system logs
        """
        # TODO: Implement log reading
        return [
            f"[{datetime.now().isoformat()}] INFO: System running normally",
            f"[{datetime.now().isoformat()}] INFO: Cache cleared",
        ]
