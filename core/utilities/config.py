"""
Configuration settings for Orion RAG pipeline.
Simplified configuration system for local personal RAG assistant.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


# ========== ENUMS ==========

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ChunkerType(str, Enum):
    """Text chunking strategies"""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic" 
    SMART = "smart"


# ========== BASE CONFIGURATION CLASS ==========

@dataclass
class BaseConfig:
    """Base configuration class"""
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ========== CORE RAG CONFIGURATION ==========

@dataclass
class EmbeddingConfig(BaseConfig):
    """Embedding model configuration"""
    model: str = "nomic-embed-text"
    batch_size: int = 32
    timeout: int = 30
    cache_embeddings: bool = True


@dataclass
class ChunkingConfig(BaseConfig):
    """Document chunking configuration"""
    strategy: ChunkerType = ChunkerType.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100


@dataclass
class RetrievalConfig(BaseConfig):
    """Document retrieval configuration"""
    default_k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    
    # Hybrid search
    enable_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3


@dataclass
class LLMConfig(BaseConfig):
    """Large Language Model configuration"""
    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    
    # RAG prompt
    system_prompt: str = (
        "You are Orion, a helpful AI assistant with access to a knowledge base. "
        "Use the provided context to answer questions accurately and cite sources when appropriate."
    )


@dataclass
class VectorStoreConfig(BaseConfig):
    """Vector store configuration"""
    index_type: str = "IndexFlatIP"  # FAISS index type
    persist_immediately: bool = True


@dataclass
class RAGConfig(BaseConfig):
    """Complete RAG pipeline configuration"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    
    # Processing
    enable_deduplication: bool = True
    deduplication_threshold: float = 0.9


# ========== SYSTEM CONFIGURATION ==========

@dataclass
class StorageConfig(BaseConfig):
    """Data storage configuration"""
    data_directory: str = "./orion-data"
    temp_directory: str = "./temp"
    
    # Profile management (simplified)
    default_profile: str = "default" 
    active_profile: str = "default"


@dataclass
class ServerConfig(BaseConfig):
    """Web server configuration (for API mode)"""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
        "tauri://localhost"
    ])


@dataclass
class SystemConfig(BaseConfig):
    """System configuration"""
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Ollama integration
    require_ollama: bool = True
    ollama_health_check: bool = True


# ========== GPU CONFIGURATION ==========

@dataclass
class GPUConfig(BaseConfig):
    """GPU acceleration configuration"""
    enabled: bool = True
    auto_detect: bool = True
    preferred_device: str = "auto"  # "auto", "cpu", "cuda:0"
    fallback_to_cpu: bool = True


# ========== LOGGING CONFIGURATION ==========

@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "./logs/orion.log"
    log_to_console: bool = True
    verbose: bool = False  # Detailed output mode


# ========== MAIN CONFIGURATION CLASS ==========

@dataclass
class OrionConfig(BaseConfig):
    """Complete Orion configuration"""
    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    version: str = "1.0.0"
    
    @classmethod
    def from_env(cls) -> "OrionConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # RAG settings
        if model := os.getenv("ORION_EMBEDDING_MODEL"):
            config.rag.embedding.model = model
        if model := os.getenv("ORION_LLM_MODEL"):
            config.rag.llm.model = model
        if size := os.getenv("ORION_CHUNK_SIZE"):
            config.rag.chunking.chunk_size = int(size)
        if overlap := os.getenv("ORION_CHUNK_OVERLAP"):
            config.rag.chunking.chunk_overlap = int(overlap)
        if k := os.getenv("ORION_RETRIEVAL_K"):
            config.rag.retrieval.default_k = int(k)
        if temp := os.getenv("ORION_TEMPERATURE"):
            config.rag.llm.temperature = float(temp)
        
        # System settings
        if port := os.getenv("ORION_PORT"):
            config.system.server.port = int(port)
        if host := os.getenv("ORION_HOST"):
            config.system.server.host = host
        if data_dir := os.getenv("ORION_DATA_DIR"):
            config.system.storage.data_directory = data_dir
        
        # GPU settings
        if gpu_enabled := os.getenv("ORION_GPU_ENABLED"):
            config.gpu.enabled = gpu_enabled.lower() == "true"
        if device := os.getenv("ORION_GPU_DEVICE"):
            config.gpu.preferred_device = device
        
        # Logging
        if log_level := os.getenv("ORION_LOG_LEVEL"):
            config.logging.level = LogLevel(log_level.lower())
        if verbose := os.getenv("ORION_VERBOSE"):
            config.logging.verbose = verbose.lower() == "true"
            if config.logging.verbose:
                config.logging.level = LogLevel.DEBUG
        
        return config
    
    @property
    def vectorstore_path(self) -> str:
        """Get the vectorstore path for active profile"""
        return str(Path(self.system.storage.data_directory) / "profiles" / self.system.storage.active_profile / "vectorstore")


# ========== SUPPORTED FILE EXTENSIONS ==========

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".txt",
    ".csv",
    ".md",
    ".rtf",
    # Images (for future OCR/vision)
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
    # Code files
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".html",
    ".css",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".r",
    ".m",
    ".swift",
    ".kt",
    ".dart",
    # Text/Documentation
    ".org",
    ".rst",
    ".tex",
    ".log",
    ".conf",
    ".properties",
    # Email (common formats)
    ".eml",
    ".msg",
    ".mbox",
    # Archives (we can extract and process contents)
    ".zip",
    ".tar",
    ".gz",
    ".7z"
}
