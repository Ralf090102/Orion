"""
Pydantic models for system API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SystemStatus(str, Enum):
    """System health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class AutoIndexMode(str, Enum):
    """Auto-indexing behavior for file monitoring"""

    OFF = "off"
    SMART_DEFAULTS = "smart_defaults"  # Auto-index files < X MB
    ASK_ALWAYS = "ask_always"
    AUTO_ALL = "auto_all"


class ServiceStatus(BaseModel):
    """Status of individual system service"""

    name: str = Field(..., description="Service name", example="ollama")

    status: SystemStatus = Field(..., description="Current service status")

    response_time: Optional[float] = Field(default=None, description="Service response time in seconds", ge=0.0, example=0.123)

    last_check: datetime = Field(..., description="When the service was last checked")

    error: Optional[str] = Field(
        default=None, description="Error message if service is unhealthy", example="Connection refused"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional service details", example={"version": "1.0.0", "uptime": "5 days"}
    )


class ModelInfo(BaseModel):
    """Information about available LLM models"""

    name: str = Field(..., description="Model name", example="mistral:latest")

    size: Optional[str] = Field(default=None, description="Model size", example="7B parameters")

    modified: Optional[datetime] = Field(default=None, description="When the model was last modified")

    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional model details", example={"family": "mistral", "parameter_size": "7B"}
    )

    is_available: bool = Field(default=True, description="Whether the model is currently available", example=True)


class SystemHealth(BaseModel):
    """Overall system health response"""

    status: SystemStatus = Field(..., description="Overall system status")

    services: List[ServiceStatus] = Field(default=[], description="Status of individual services")

    uptime: Optional[float] = Field(default=None, description="System uptime in seconds", ge=0.0, example=3600.5)

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the health check was performed")

    version: Optional[str] = Field(default=None, description="Application version", example="0.1.0")


class SystemStats(BaseModel):
    """System usage statistics"""

    vectorstore_size: Optional[int] = Field(default=None, description="Number of documents in vectorstore", ge=0, example=1500)

    total_chunks: Optional[int] = Field(default=None, description="Total number of text chunks", ge=0, example=25000)

    active_conversations: Optional[int] = Field(default=None, description="Number of active conversations", ge=0, example=10)

    total_queries: Optional[int] = Field(default=None, description="Total queries processed since startup", ge=0, example=5000)

    cache_hit_rate: Optional[float] = Field(
        default=None, description="Cache hit rate percentage (0-100)", ge=0.0, le=100.0, example=85.5
    )

    cache_size: Optional[int] = Field(default=None, description="Number of items in cache", ge=0, example=500)

    memory_usage: Optional[Dict[str, float]] = Field(
        default=None,
        description="Memory usage statistics in MB",
        example={"used": 512.5, "available": 1024.0, "percentage": 50.0},
    )

    disk_usage: Optional[Dict[str, float]] = Field(
        default=None, description="Disk usage statistics in GB", example={"used": 2.5, "available": 10.0, "percentage": 25.0}
    )

    processing_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document processing statistics",
        example={"documents_processed_today": 50, "average_processing_time": 2.3, "failed_documents": 2},
    )


class ModelsResponse(BaseModel):
    """Response containing available models"""

    models: List[ModelInfo] = Field(default=[], description="List of available models")

    default_model: Optional[str] = Field(default=None, description="Default model used for queries", example="mistral:latest")

    total_models: int = Field(default=0, description="Total number of models available", ge=0, example=5)


class ConfigUpdate(BaseModel):
    """Request to update system configuration"""

    chunk_size: Optional[int] = Field(
        default=None, description="Default chunk size for document processing", ge=100, le=4000, example=1000
    )

    chunk_overlap: Optional[int] = Field(default=None, description="Default chunk overlap", ge=0, le=1000, example=200)

    retrieval_k: Optional[int] = Field(
        default=None, description="Default number of documents to retrieve", ge=1, le=20, example=5
    )

    default_model: Optional[str] = Field(default=None, description="Default LLM model", example="mistral:latest")

    temperature: Optional[float] = Field(default=None, description="Default LLM temperature", ge=0.0, le=1.0, example=0.7)

    enable_caching: Optional[bool] = Field(default=None, description="Whether to enable caching", example=True)

    cache_size: Optional[int] = Field(default=None, description="Maximum cache size", ge=100, le=10000, example=1000)


class ConfigResponse(BaseModel):
    """Current system configuration"""

    chunk_size: int = Field(..., description="Current chunk size setting", example=1000)

    chunk_overlap: int = Field(..., description="Current chunk overlap setting", example=200)

    retrieval_k: int = Field(..., description="Current retrieval K setting", example=5)

    default_model: str = Field(..., description="Current default model", example="mistral:latest")

    temperature: float = Field(..., description="Current temperature setting", example=0.7)

    enable_caching: bool = Field(..., description="Whether caching is enabled", example=True)

    cache_size: int = Field(..., description="Current cache size", example=1000)

    features: Dict[str, bool] = Field(
        default={},
        description="Enabled features",
        example={"ocr_processing": True, "table_extraction": True, "async_processing": True, "query_enhancement": True},
    )

    last_updated: Optional[datetime] = Field(default=None, description="When configuration was last updated")


class AutoIndexingConfig(BaseModel):
    """Auto-indexing configuration for file monitoring"""

    mode: AutoIndexMode = Field(default=AutoIndexMode.SMART_DEFAULTS, description="Auto-indexing behavior mode")

    max_file_size_mb: int = Field(
        default=10, description="Maximum file size in MB for smart auto-indexing", ge=1, le=1000, example=10
    )

    excluded_extensions: List[str] = Field(
        default=[".tmp", ".log", ".cache", ".lock", ".db", ".sqlite"],
        description="File extensions to exclude from auto-indexing",
        example=[".tmp", ".log", ".cache", ".flac", ".mp3"],
    )

    included_extensions: List[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx", ".py", ".js", ".ts"],
        description="File extensions to prioritize for auto-indexing",
        example=[".txt", ".md", ".pdf", ".docx", ".py"],
    )

    schedule_enabled: bool = Field(default=False, description="Enable scheduled indexing", example=True)

    schedule_time: Optional[str] = Field(
        default=None,
        description="Scheduled indexing time (HH:MM format)",
        pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$",
        example="02:00",
    )


class SystemConfig(BaseModel):
    """Complete system configuration"""

    # Server Configuration
    port: int = Field(default=8000, description="FastAPI server port", ge=1024, le=65535, example=8000)

    auto_start: bool = Field(default=False, description="Start Orion automatically on system boot", example=False)

    # Resource Management (Based on Ryzen 5 5600 + 16GB RAM)
    max_memory_usage_mb: int = Field(
        default=4096,  # ~25% of 16GB
        description="Maximum memory usage in MB",
        ge=512,
        le=12288,  # Max ~75% of 16GB
        example=4096,
    )

    max_cpu_usage_percent: int = Field(
        default=75, description="Maximum CPU usage percentage", ge=10, le=90, example=75  # ~75% of 6 cores (4.5 cores)
    )

    # Profile Management
    active_profile: str = Field(
        default="default", description="Currently active user profile", min_length=1, max_length=50, example="work_profile"
    )

    load_multiple_vectorstores: bool = Field(
        default=False, description="Allow loading multiple vectorstores simultaneously", example=False
    )

    # File Monitoring
    auto_indexing: AutoIndexingConfig = Field(
        default_factory=AutoIndexingConfig, description="Auto-indexing configuration for file monitoring"
    )

    # Data Management
    data_directory: str = Field(
        default="./orion-data",
        description="Directory for Orion data storage",
        example="C:\\Users\\Username\\AppData\\Local\\Orion",
    )

    config_directory: str = Field(
        default="./orion-config",
        description="Directory for Orion configuration files",
        example="C:\\Users\\Username\\AppData\\Local\\Orion\\config",
    )

    # Integration Settings
    require_ollama_setup: bool = Field(default=True, description="Require separate Ollama installation/setup", example=True)

    # GPU Acceleration Settings
    enable_gpu_acceleration: bool = Field(
        default=True, description="Enable GPU acceleration if available (NVIDIA/CUDA)", example=True
    )

    gpu_memory_limit_mb: Optional[int] = Field(
        default=None, description="GPU memory limit in MB (None = auto-detect)", ge=512, le=32768, example=8192
    )

    preferred_gpu_device: Optional[str] = Field(
        default="auto", description="Preferred GPU device (auto, cuda:0, cuda:1, etc.)", example="cuda:0"
    )


class ProfileInfo(BaseModel):
    """User profile information"""

    name: str = Field(..., description="Profile name", min_length=1, max_length=50, example="work_profile")

    display_name: str = Field(
        ..., description="Human-readable profile name", min_length=1, max_length=100, example="Work Documents"
    )

    vectorstore_path: str = Field(
        ..., description="Path to profile's vectorstore", example="./orion-data/profiles/work_profile/vectorstore"
    )

    document_count: int = Field(default=0, description="Number of documents in this profile", ge=0, example=1250)

    total_size_mb: float = Field(default=0.0, description="Total size of indexed documents in MB", ge=0.0, example=45.7)

    last_indexed: Optional[datetime] = Field(default=None, description="When documents were last indexed for this profile")

    created_at: datetime = Field(..., description="When this profile was created")

    is_active: bool = Field(default=False, description="Whether this profile is currently active", example=True)


class SystemConfigUpdateRequest(BaseModel):
    """Request to update system configuration"""

    port: Optional[int] = Field(default=None, description="Update server port", ge=1024, le=65535)

    auto_start: Optional[bool] = Field(default=None, description="Update auto-start setting")

    max_memory_usage_mb: Optional[int] = Field(default=None, description="Update maximum memory usage", ge=512, le=12288)

    max_cpu_usage_percent: Optional[int] = Field(default=None, description="Update maximum CPU usage", ge=10, le=90)

    active_profile: Optional[str] = Field(default=None, description="Switch to different profile", min_length=1, max_length=50)

    load_multiple_vectorstores: Optional[bool] = Field(default=None, description="Update multiple vectorstore loading setting")

    auto_indexing: Optional[AutoIndexingConfig] = Field(default=None, description="Update auto-indexing configuration")

    enable_gpu_acceleration: Optional[bool] = Field(default=None, description="Update GPU acceleration setting")

    gpu_memory_limit_mb: Optional[int] = Field(default=None, description="Update GPU memory limit", ge=512, le=32768)

    preferred_gpu_device: Optional[str] = Field(default=None, description="Update preferred GPU device")
