"""
FastAPI Backend Main Application
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.api.system import router as system_router
from backend.services import get_config_service, get_gpu_manager
from backend.services.port_manager import create_port_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 Orion Backend starting up...")

    try:
        # Initialize services
        config_service = get_config_service()
        system_config = config_service.get_system_config()

        # Log system configuration
        logger.info(
            f"System Config - CPU: {system_config.max_cpu_usage_percent}%, Memory: {system_config.max_memory_usage_mb}MB"
        )
        logger.info(f"Active Profile: {system_config.active_profile}")

        # Initialize GPU if available
        if system_config.enable_gpu_acceleration:
            try:
                gpu_manager = get_gpu_manager()
                gpu_capabilities = gpu_manager.detect_gpu_capabilities()
                if gpu_capabilities.has_cuda:
                    logger.info(f"GPU Acceleration: Enabled with {len(gpu_capabilities.gpus)} CUDA device(s)")
                else:
                    logger.info("GPU Acceleration: Requested but no CUDA devices found, running CPU-only")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")

        # Store config in app state for easy access
        app.state.config_service = config_service
        app.state.system_config = system_config

        logger.info("✅ Backend services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize backend services: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Orion Backend shutting down...")


app = FastAPI(
    title="Orion RAG API",
    description="Personal RAG Assistant Backend API",
    version="1.0.0-alpha",
    lifespan=lifespan,
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Svelte dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "tauri://localhost",  # Tauri app
        "https://tauri.localhost",  # Tauri app (secure)
        "http://127.0.0.1:*",  # Local development
        "http://localhost:*",  # Dynamic port support
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers with proper error handling
app.include_router(ingest_router, prefix="/api/ingest", tags=["Document Ingestion"])
app.include_router(query_router, prefix="/api/query", tags=["Query & Search"])
app.include_router(system_router, prefix="/api/system", tags=["System Management"])


# Enhanced error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


@app.get("/")
async def root():
    """Root endpoint with system information"""
    try:
        config_service = app.state.config_service
        system_config = app.state.system_config
        active_profile = config_service.get_active_profile()

        return {
            "message": "Orion Personal RAG Assistant API",
            "version": "1.0.0-alpha",
            "status": "healthy",
            "system": {
                "cpu_limit": f"{system_config.max_cpu_usage_percent}%",
                "memory_limit": f"{system_config.max_memory_usage_mb}MB",
                "gpu_acceleration": system_config.enable_gpu_acceleration,
                "active_profile": active_profile.name if active_profile else None,
            },
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {"message": "Orion Personal RAG Assistant API", "version": "1.0.0-alpha", "status": "error", "error": str(e)}


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        from datetime import datetime

        config_service = app.state.config_service
        system_config = app.state.system_config

        # Basic health info
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "running",
                "configuration": "loaded",
                "profiles": len(config_service.list_profiles()),
            },
            "system": {
                "port": system_config.port,
                "cpu_limit": system_config.max_cpu_usage_percent,
                "memory_limit": system_config.max_memory_usage_mb,
                "gpu_acceleration": system_config.enable_gpu_acceleration,
            },
        }

        # Add GPU info if available
        if system_config.enable_gpu_acceleration:
            try:
                gpu_manager = get_gpu_manager()
                gpu_capabilities = gpu_manager.detect_gpu_capabilities()
                health_data["gpu"] = {
                    "cuda_available": gpu_capabilities.has_cuda,
                    "device_count": len(gpu_capabilities.gpus),
                    "total_memory_mb": gpu_capabilities.total_gpu_memory_mb,
                }
            except Exception as e:
                health_data["gpu"] = {"status": "error", "error": str(e)}

        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


def create_app() -> FastAPI:
    """Factory function to create FastAPI app (useful for testing)"""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the FastAPI server with configuration integration"""
    import uvicorn

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        # Use port manager to find available port
        port_manager = create_port_manager(preferred_port=port)
        available_port = port_manager.find_available_port()

        if available_port != port:
            logger.warning(f"Requested port {port} not available, using port {available_port}")

        logger.info(f"Starting Orion backend on {host}:{available_port}")

        # Run server
        uvicorn.run("backend.main:app", host=host, port=available_port, reload=reload, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    run_server()
