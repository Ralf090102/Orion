"""
Models endpoint request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelConfig(BaseModel):
    """Current LLM model configuration"""
    
    model: str
    base_url: str
    temperature: float
    top_p: float
    max_tokens: Optional[int] = None
    timeout: int
    system_prompt: str


class UpdateModelRequest(BaseModel):
    """Request to update the active LLM model"""
    
    model: str


class UpdateLLMConfigRequest(BaseModel):
    """Request to update LLM configuration parameters (excluding model name)"""
    
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for text generation (0.0-2.0)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate (null for unlimited)")
    timeout: Optional[int] = Field(None, ge=1, description="Request timeout in seconds")
    system_prompt: Optional[str] = Field(None, min_length=1, description="System prompt for the LLM")


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

