"""
Orion Backend Dependencies

Shared dependency instances for FastAPI routes.
Uses singleton pattern for heavy components (retriever, generator).
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from fastapi import Depends, HTTPException, status

from src.generation.generate import AnswerGenerator
from src.generation.session_manager import SessionManager, get_session_manager
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig, get_config

logger = logging.getLogger(__name__)


def _session_storage_dir() -> Optional[Path]:
    """
    Where to persist chat sessions.

    Tauri sets ORION_DATA_DIR to a stable per-user app-data directory that
    survives reinstalls/updates (unlike the NSIS resource dir, which gets
    wiped and re-extracted every time). Falls back to SessionManager's own
    "./data/chat-data" default when unset, e.g. running the backend directly
    for API dev or tests outside of Tauri.
    """
    data_dir = os.environ.get("ORION_DATA_DIR")
    return Path(data_dir) / "chat-data" if data_dir else None


def _build_config() -> OrionConfig:
    """
    Build the app configuration.

    get_config() defaults to from_env=False and otherwise ignores env vars
    entirely, so ORION_VECTORSTORE_PERSIST_DIRECTORY (set by Rust alongside
    ORION_DATA_DIR) would silently do nothing without this explicit override.
    Same "survive reinstalls" motivation as _session_storage_dir() above.

    The TTS audio cache (config.tts.cache_dir) had the identical CWD-relative
    default ("./data/tts/cache") and was missed in the original sweep -- it's
    lower stakes than the vector store (a regenerable cache, not primary
    data), but still gets wiped on every reinstall/update without this.
    """
    config = get_config()
    data_dir = os.environ.get("ORION_DATA_DIR")
    if data_dir:
        config.rag.vectorstore.persist_directory = str(Path(data_dir) / "chroma-data")
        config.tts.cache_dir = Path(data_dir) / "tts-cache"
    return config


# ========== GLOBAL INSTANCES (SINGLETONS) ==========
_config: Optional[OrionConfig] = None
_session_manager: Optional[SessionManager] = None
_retriever: Optional[OrionRetriever] = None
_generator: Optional[AnswerGenerator] = None
_tts_manager: Optional["UnifiedTTSManager"] = None  # Forward reference for lazy import

if TYPE_CHECKING:
    # Import for type checkers only (avoids runtime import side-effects)
    from src.utilities.tts.tts_manager import UnifiedTTSManager


# ========== INITIALIZATION & CLEANUP ==========
def initialize_resources():
    """
    Initialize shared resources on application startup.
    
    Called by FastAPI lifespan event.
    """
    global _config, _session_manager, _retriever, _generator
    
    logger.info("Initializing shared resources...")
    
    # 1. Load configuration
    _config = _build_config()
    logger.info(f"✓ Configuration loaded (version: {_config.version})")
    
    # 2. Initialize session manager with persistence
    _session_manager = get_session_manager(
        persist_to_disk=True,
        storage_dir=_session_storage_dir(),
        session_expiry_days=7,
        auto_cleanup=True,
    )
    logger.info("✓ Session manager initialized")
    
    # 3. Pre-warm retriever (optional - can be lazy loaded)
    try:
        _retriever = OrionRetriever(config=_config)
        logger.info("✓ Retriever initialized")
    except Exception as e:
        logger.warning(f"⚠ Retriever initialization failed (will lazy-load): {e}")
        _retriever = None
    
    # 4. Pre-warm generator (optional - can be lazy loaded)
    try:
        _generator = AnswerGenerator(config=_config)
        logger.info("✓ Answer generator initialized")
    except Exception as e:
        logger.warning(f"⚠ Generator initialization failed (will lazy-load): {e}")
        _generator = None
    
    logger.info("✅ All resources initialized")


def warm_up_retriever_background() -> None:
    """
    Pre-load the retriever's heavy ML components (embedding model, reranker,
    Chroma client) on a background thread, right after startup.

    initialize_resources() already constructs the `_retriever` singleton
    itself, but cheaply -- OrionRetriever.__init__ just sets fields to None
    (see src/retrieval/retriever.py); the actual embedding/reranker/vector-
    store loading is deferred to _initialize_components(), normally only
    triggered by the first real query. Calling that here, on a daemon thread,
    means it usually finishes warming before the user's first query arrives
    instead of the query itself paying that cost -- without blocking Uvicorn
    from binding the port or answering /health in the meantime, and without
    duplicating work: get_retriever_dependency() below returns this same
    _retriever instance, and _initialize_components() is idempotent/lock-
    guarded, so a real query racing this thread just waits on the same
    in-progress load rather than starting a second one.

    Best-effort: any failure here is logged and swallowed, since the normal
    lazy-load path in get_retriever_dependency() (or the retriever's own
    _initialize_components() call from .query()) is still there as a
    fallback -- a failed warm-up just means the first real query pays the
    cost it would have paid anyway before this existed.
    """

    def _warm() -> None:
        if _retriever is None:
            logger.info("Skipping retriever warm-up: retriever was not constructed at startup")
            return
        try:
            logger.info("Warming up retriever in background (embedding model, reranker, vector store)...")
            _retriever._initialize_components()
            logger.info("✅ Retriever warm-up complete")
        except Exception as e:
            logger.warning(f"Retriever warm-up failed, will lazy-load on first query instead: {e}")

    threading.Thread(target=_warm, name="orion-retriever-warmup", daemon=True).start()


def cleanup_resources():
    """
    Cleanup resources on application shutdown.
    
    Called by FastAPI lifespan event.
    """
    global _config, _session_manager, _retriever, _generator
    
    logger.info("Cleaning up resources...")
    
    # Reset all singletons
    _config = None
    _session_manager = None
    _retriever = None
    _generator = None
    
    logger.info("✅ Resources cleaned up")


# ========== DEPENDENCY FUNCTIONS ==========
def get_config_dependency() -> OrionConfig:
    """
    Dependency: Get configuration instance.
    
    Returns:
        OrionConfig instance
        
    Raises:
        HTTPException: If config not initialized
    """
    global _config
    
    if _config is None:
        # Lazy initialization
        _config = _build_config()

    return _config


def get_session_manager_dependency() -> SessionManager:
    """
    Dependency: Get session manager instance.
    
    Returns:
        SessionManager instance
        
    Raises:
        HTTPException: If session manager not initialized
    """
    global _session_manager
    
    if _session_manager is None:
        # Lazy initialization
        _session_manager = get_session_manager(
            persist_to_disk=True,
            storage_dir=_session_storage_dir(),
            session_expiry_days=7,
            auto_cleanup=True,
        )
    
    return _session_manager


def get_retriever_dependency(
    config: OrionConfig = Depends(get_config_dependency)
) -> OrionRetriever:
    """
    Dependency: Get retriever instance.
    
    Args:
        config: Configuration instance (injected)
        
    Returns:
        OrionRetriever instance
        
    Raises:
        HTTPException: If retriever initialization fails
    """
    global _retriever
    
    if _retriever is None:
        try:
            _retriever = OrionRetriever(config=config)
            logger.info("Retriever lazy-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Retriever service unavailable: {str(e)}",
            )
    
    return _retriever


def get_generator_dependency(
    config: OrionConfig = Depends(get_config_dependency)
) -> AnswerGenerator:
    """
    Dependency: Get answer generator instance.
    
    Args:
        config: Configuration instance (injected)
        
    Returns:
        AnswerGenerator instance
        
    Raises:
        HTTPException: If generator initialization fails
    """
    global _generator
    
    if _generator is None:
        try:
            _generator = AnswerGenerator(config=config)
            logger.info("Answer generator lazy-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Generator service unavailable: {str(e)}",
            )
    
    return _generator


# ========== OPTIONAL: API KEY AUTHENTICATION ==========
# Uncomment to enable API key authentication

# from fastapi.security import APIKeyHeader
# import os
# 
# API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
# 
# async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
#     """
#     Dependency: Verify API key for protected endpoints.
#     
#     Set environment variable ORION_API_KEY to enable authentication.
#     
#     Args:
#         api_key: API key from request header
#         
#     Raises:
#         HTTPException: If API key invalid or missing
#     """
#     expected_key = os.getenv("ORION_API_KEY")
#     
#     # If no key configured, skip authentication
#     if not expected_key:
#         return
#     
#     if not api_key or api_key != expected_key:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing API key",
#             headers={"WWW-Authenticate": "ApiKey"},
#         )
#     
#     return api_key


# ========== UTILITY FUNCTIONS ==========
def reset_generator():
    """
    Reset generator instance (useful for config changes).
    
    Call this after updating configuration to force re-initialization.
    """
    global _generator
    _generator = None
    logger.info("Generator instance reset")


def reset_retriever():
    """
    Reset retriever instance (useful for config changes).
    
    Call this after updating configuration to force re-initialization.
    """
    global _retriever
    _retriever = None
    logger.info("Retriever instance reset")


def get_database_stats() -> dict:
    """
    Get database statistics from session manager.
    
    Returns:
        Dictionary with database stats
    """
    session_manager = get_session_manager_dependency()
    return session_manager.get_database_stats()


def get_tts_manager() -> "UnifiedTTSManager":
    """
    Dependency: Get UnifiedTTSManager instance.
    
    Returns:
        UnifiedTTSManager instance
        
    Raises:
        HTTPException: If TTS not enabled or initialization fails
    """
    global _tts_manager
    
    if _tts_manager is None:
        config = get_config_dependency()
        
        if not config.tts.enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS service is disabled in configuration",
            )
        
        try:
            # Lazy import to avoid dependency issues
            from src.utilities.tts.tts_manager import UnifiedTTSManager
            
            _tts_manager = UnifiedTTSManager(config)
            logger.info("✓ TTS manager lazy-loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import UnifiedTTSManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS service unavailable. Piper TTS may not be installed.",
            )
        except Exception as e:
            logger.error(f"Failed to initialize TTS manager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS service unavailable: {str(e)}",
            )
    
    return _tts_manager


def reset_tts_manager():
    """
    Reset TTS manager instance (useful for config changes).
    
    Call this after updating TTS configuration to force re-initialization.
    """
    global _tts_manager
    if _tts_manager is not None:
        _tts_manager.unload()  # Unload current voice
    _tts_manager = None
    logger.info("TTS manager instance reset")
