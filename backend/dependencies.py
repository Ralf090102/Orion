"""
Orion Backend Dependencies

Shared dependency instances for FastAPI routes.
Uses singleton pattern for heavy components (retriever, generator).
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status

from src.generation.generate import AnswerGenerator
from src.generation.session_manager import SessionManager, get_session_manager
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig, get_config

logger = logging.getLogger(__name__)


# ========== GLOBAL INSTANCES (SINGLETONS) ==========
_config: Optional[OrionConfig] = None
_session_manager: Optional[SessionManager] = None
_retriever: Optional[OrionRetriever] = None
_generator: Optional[AnswerGenerator] = None


# ========== INITIALIZATION & CLEANUP ==========
def initialize_resources():
    """
    Initialize shared resources on application startup.
    
    Called by FastAPI lifespan event.
    """
    global _config, _session_manager, _retriever, _generator
    
    logger.info("Initializing shared resources...")
    
    # 1. Load configuration
    _config = get_config()
    logger.info(f"✓ Configuration loaded (version: {_config.version})")
    
    # 2. Initialize session manager with persistence
    _session_manager = get_session_manager(
        persist_to_disk=True,
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
        _config = get_config()
    
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
