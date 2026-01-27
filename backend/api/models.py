"""
Models API endpoints for managing LLM models
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
import ollama

from backend.models.models import (
    ModelConfig,
    UpdateModelRequest,
    ModelsListResponse,
    OllamaModelInfo,
)
from backend.dependencies import get_config_dependency
from src.utilities.config import OrionConfig
from src.core.llm import check_ollama_connection

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== GET CURRENT MODEL CONFIGURATION ==========
@router.get(
    "/api/models/config",
    summary="Get current LLM configuration",
    description="Returns the current active LLM model configuration",
    tags=["Models"],
    response_model=ModelConfig,
)
async def get_model_config(config: OrionConfig = Depends(get_config_dependency)):
    """
    Get the current LLM model configuration.
    
    Args:
        config: Singleton config instance (injected)
    
    Returns:
        ModelConfig with current model settings from OrionConfig
        
    Raises:
        HTTPException: If configuration cannot be loaded
    """
    try:
        return ModelConfig(
            model=config.rag.llm.model,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            top_p=config.rag.llm.top_p,
            max_tokens=config.rag.llm.max_tokens,
            timeout=config.rag.llm.timeout,
        )
        
    except Exception as e:
        logger.error(f"Failed to get model configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model configuration: {str(e)}",
        )


# ========== UPDATE CURRENT MODEL ==========
@router.patch(
    "/api/models/config",
    summary="Update active LLM model",
    description="Update the active LLM model used for generation",
    tags=["Models"],
    response_model=ModelConfig,
)
async def update_model_config(
    request: UpdateModelRequest,
    config: OrionConfig = Depends(get_config_dependency)
):
    """
    Update the active LLM model.
    
    Args:
        request: UpdateModelRequest with new model name
        config: Singleton config instance (injected)
        
    Returns:
        Updated ModelConfig
        
    Raises:
        HTTPException: If update fails or Ollama is not available
    """
    try:
        # Check if Ollama is running
        if not check_ollama_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service is not running. Please start Ollama first.",
            )
        
        # Verify the model exists in Ollama
        try:
            models_response = ollama.list()
            available_models = []
            
            if hasattr(models_response, 'models'):
                available_models = [
                    getattr(model, 'model', getattr(model, 'name', ''))
                    for model in models_response.models
                ]
            
            if request.model not in available_models:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{request.model}' not found in Ollama. Available models: {', '.join(available_models)}",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not verify model existence: {e}")
            # Continue anyway - user might know better
        
        # Update the singleton config instance
        config.rag.llm.model = request.model
        
        logger.info(f"✅ Updated active LLM model to: {request.model}")
        
        return ModelConfig(
            model=config.rag.llm.model,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            top_p=config.rag.llm.top_p,
            max_tokens=config.rag.llm.max_tokens,
            timeout=config.rag.llm.timeout,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model configuration: {str(e)}",
        )


# ========== LIST OLLAMA MODELS ==========
@router.get(
    "/api/models",
    summary="List available Ollama models",
    description="Get list of all available Ollama models with current active model",
    tags=["Models"],
    response_model=ModelsListResponse,
)
async def list_models(config: OrionConfig = Depends(get_config_dependency)):
    """
    List all available Ollama models and show which one is currently active.
    
    Args:
        config: Singleton config instance (injected)
    
    Returns detailed information about each installed model including:
    - Model name
    - Model ID (digest)
    - Size
    - Modified date
    - Current active model indicator
    
    Returns:
        ModelsListResponse with models list and current active model
        
    Raises:
        HTTPException: If Ollama is not available or request fails
    """
    try:
        # Check if Ollama is running
        if not check_ollama_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service is not running. Please start Ollama first.",
            )
        
        # Get current active model from singleton config
        current_model = config.rag.llm.model
        
        # Get models from Ollama
        models_response = ollama.list()
        
        models_list = []
        
        # Parse the response
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                # Get modified_at and convert datetime to string if needed
                modified_at = getattr(model, 'modified_at', None)
                if modified_at and hasattr(modified_at, 'isoformat'):
                    # It's a datetime object, convert to ISO string
                    modified_str = modified_at.isoformat()
                else:
                    # Already a string or None
                    modified_str = modified_at
                
                model_info = OllamaModelInfo(
                    name=getattr(model, 'model', getattr(model, 'name', 'unknown')),
                    id=getattr(model, 'digest', 'unknown')[:12],  # Short digest like CLI
                    size=_format_size(getattr(model, 'size', 0)),
                    size_bytes=getattr(model, 'size', 0),
                    modified=modified_str,
                    details={
                        "format": getattr(model, 'details', {}).get('format') if hasattr(model, 'details') else None,
                        "family": getattr(model, 'details', {}).get('family') if hasattr(model, 'details') else None,
                        "parameter_size": getattr(model, 'details', {}).get('parameter_size') if hasattr(model, 'details') else None,
                    }
                )
                models_list.append(model_info)
        
        return ModelsListResponse(
            status="success",
            current_model=current_model,
            total_models=len(models_list),
            models=models_list,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Ollama models: {str(e)}",
        )


def _format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable format (matches Ollama CLI output).
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "4.4 GB", "274 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.0f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
