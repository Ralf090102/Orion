"""
FastAPI Backend Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.api.system import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    print("🚀 Orion Backend starting up...")
    yield
    # Shutdown
    print("🛑 Orion Backend shutting down...")


app = FastAPI(
    title="Orion RAG API",
    description="Backend API for Orion RAG Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Svelte dev server
        "http://localhost:5173",  # Vite dev server
        "tauri://localhost",  # Tauri app
        "https://tauri.localhost",  # Tauri app (secure)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ingest_router, prefix="/api/ingest", tags=["Document Ingestion"])
app.include_router(query_router, prefix="/api/query", tags=["Query & Search"])
app.include_router(system_router, prefix="/api/system", tags=["System Management"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Orion RAG API", "version": "0.1.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "ollama": "connected",  # TODO: Check actual ollama connection
            "vectorstore": "ready",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
