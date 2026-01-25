"""
Orion Backend API - FastAPI Application

Main FastAPI application with CORS, lifespan events, and route registration.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.dependencies import cleanup_resources, initialize_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ========== LIFESPAN EVENTS ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    
    Startup:
        - Initialize configuration
        - Load session manager
        - Warm up retriever and generator (optional)
        - Check Ollama connection
        
    Shutdown:
        - Cleanup resources
        - Close database connections
    """
    logger.info("🚀 Starting Orion Backend API...")
    
    try:
        # Initialize shared resources
        initialize_resources()
        logger.info("✅ Resources initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize resources: {e}")
        raise
    
    # Application is running
    yield
    
    # Shutdown: cleanup resources
    logger.info("🛑 Shutting down Orion Backend API...")
    cleanup_resources()
    logger.info("✅ Cleanup complete")


# ========== FASTAPI APP ==========
app = FastAPI(
    title="Orion RAG Assistant API",
    description="Backend API for Orion - Local RAG Assistant with chat capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ========== CORS MIDDLEWARE ==========
# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Svelte dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Total-Count", "X-Page-Size"],  # Custom headers for pagination
)


# ========== EXCEPTION HANDLERS ==========
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


# ========== ROOT ENDPOINT ==========
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "Orion RAG Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "config": "/api/config",
            "rag": "/api/ask, /api/query",
            "chat": "/api/chat/*",
            "ingestion": "/api/ingest/*",
        },
    }


# ========== ROUTE REGISTRATION ==========
# Import and include routers
from backend.api import health, ingestion, rag, settings

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(rag.router)
app.include_router(settings.router)

# Uncomment as you create each router
# from backend.api import chat
# app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

# WebSocket routes
# from backend.websockets import chat as ws_chat
# app.include_router(ws_chat.router)


# ========== STARTUP MESSAGE ==========
@app.on_event("startup")
async def startup_message():
    """Print startup information."""
    logger.info("=" * 60)
    logger.info("Orion Backend API is ready!")
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m backend.app
    # Or: uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
