"""
Orion Backend API - FastAPI Application

Main FastAPI application with CORS, lifespan events, and route registration.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.dependencies import cleanup_resources, initialize_resources
from backend.metrics import metrics_collector

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
# Orion's frontend only ever runs as one of two origins -- the Vite dev
# server (`npm run dev`, per frontend/vite.config.ts) or the Tauri webview
# in production. A wildcard would let any website open in the user's
# regular browser make requests to a backend that talks to their local
# Ollama instance and knowledge base, so we allowlist explicitly instead.
#
# The production origin is matched by regex rather than one hardcoded
# string: Tauri v2 serves the app from the `tauri.localhost` host, but the
# exact scheme (http/https) is platform- and version-specific and wasn't
# empirically verified against a packaged build in this change. Neither
# `tauri.localhost` nor the `tauri://` scheme can be registered by a real
# website, so matching the pattern is as safe as listing every variant.
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server (frontend/vite.config.ts devUrl)
]
ALLOWED_ORIGIN_REGEX = r"^(https?://tauri\.localhost|tauri://localhost)$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,  # nothing in this app uses cookies/credentialed requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Total-Count", "X-Page-Size"],  # Custom headers for pagination
)


# ========== METRICS MIDDLEWARE ==========
@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Records request count/latency/errors per route, keyed by route template
    (e.g. "GET /api/chat/{session_id}") rather than the raw resolved path, so
    metrics aggregate across different session/file IDs instead of fragmenting."""
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000

    route = request.scope.get("route")
    path = route.path if route is not None else request.url.path
    endpoint = f"{request.method} {path}"

    metrics_collector.record(endpoint=endpoint, latency_ms=latency_ms, is_error=response.status_code >= 500)
    return response


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
            "speech": "/api/speech/*",
        },
    }


# ========== ROUTE REGISTRATION ==========
# Import and include routers
from backend.api import chat, health, ingestion, models, rag, settings, speech

app.include_router(settings.router)
app.include_router(ingestion.router)
app.include_router(rag.router)
app.include_router(chat.router)
app.include_router(speech.router)
app.include_router(models.router)
app.include_router(health.router)


# ========== WEBSOCKET ROUTES ==========
from fastapi import WebSocket

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
    get_session_manager_dependency,
)
from backend.websockets.chat import chat_websocket_endpoint


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.
    
    Compatible with HuggingFace chat-ui and standard WebSocket clients.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session identifier
    
    Example usage (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/chat/abc123');
        
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'message',
                content: 'What is machine learning?',
                data: { rag_mode: 'auto', include_sources: true }
            }));
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'token') {
                console.log(msg.content);
            }
        };
    """
    # Get dependencies manually (WebSocket doesn't support Depends)
    session_manager = get_session_manager_dependency()
    generator = get_generator_dependency(get_config_dependency())
    config = get_config_dependency()
    
    await chat_websocket_endpoint(
        websocket=websocket,
        session_id=session_id,
        session_manager=session_manager,
        generator=generator,
        config=config,
    )


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
