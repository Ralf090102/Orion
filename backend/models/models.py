"""
Models endpoint request/response models
"""

from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    """Current LLM model configuration"""
    
    model: str
    base_url: str
    temperature: float
    top_p: float
    max_tokens: Optional[int] = None
    timeout: int


class UpdateModelRequest(BaseModel):
    """Request to update the active LLM model"""
    
    model: str


class OllamaModelInfo(BaseModel):
    """Information about an Ollama model"""
    
    name: str
    id: str
    size: str
    size_bytes: int
    modified: str
    details: Optional[dict] = None


class ModelsListResponse(BaseModel):
    """Response containing list of available models and current active model"""
    
    status: str
    current_model: str
    total_models: int
    models: list[OllamaModelInfo]
