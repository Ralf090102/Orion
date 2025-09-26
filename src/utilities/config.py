import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ========== ENVIRONMENT VARIABLE HELPERS ==========
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float value from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_str(key: str, default: str = "") -> str:
    """Get string value from environment variable"""
    return os.getenv(key, default)


def get_env_enum(key: str, enum_class: type, default: Any) -> Any:
    """Get enum value from environment variable"""
    value = os.getenv(key, "").lower()
    for enum_val in enum_class:
        if enum_val.value.lower() == value:
            return enum_val
    return default


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

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration values"""
        pass


# ========== RAG CONFIGURATION ==========
@dataclass
class EmbeddingConfig(BaseConfig):
    """Embedding model configuration"""

    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    timeout: int = 30
    cache_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding configuration from environment variables"""
        return cls(
            model=get_env_str("ORION_EMBEDDING_MODEL", "BAAI/bge-m3"),
            batch_size=get_env_int("ORION_EMBEDDING_BATCH_SIZE", 32),
            timeout=get_env_int("ORION_EMBEDDING_TIMEOUT", 30),
            cache_embeddings=get_env_bool("ORION_EMBEDDING_CACHE", True),
        )


@dataclass
class ChunkingConfig(BaseConfig):
    """Document chunking configuration"""

    strategy: ChunkerType = ChunkerType.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 256
    max_chunk_size: int = 512
    min_chunk_size: int = 256

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """Load chunking configuration from environment variables"""
        return cls(
            strategy=get_env_enum("ORION_CHUNKING_STRATEGY", ChunkerType, ChunkerType.RECURSIVE),
            chunk_size=get_env_int("ORION_CHUNK_SIZE", 512),
            chunk_overlap=get_env_int("ORION_CHUNK_OVERLAP", 256),
            max_chunk_size=get_env_int("ORION_MAX_CHUNK_SIZE", 512),
            min_chunk_size=get_env_int("ORION_MIN_CHUNK_SIZE", 256),
        )

    def validate(self) -> None:
        """Validate chunking configuration"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if self.chunk_size > self.max_chunk_size:
            raise ValueError("chunk_size must be <= max_chunk_size")


@dataclass
class PreprocessingConfig(BaseConfig):
    """Text preprocessing configuration"""

    similarity_threshold: float = 0.85
    min_text_length: int = 50
    enable_deduplication: bool = True
    enable_normalization: bool = True

    @classmethod
    def from_env(cls) -> "PreprocessingConfig":
        """Load preprocessing configuration from environment variables"""
        return cls(
            similarity_threshold=get_env_float("ORION_PREPROCESSING_SIMILARITY_THRESHOLD", 0.85),
            min_text_length=get_env_int("ORION_PREPROCESSING_MIN_TEXT_LENGTH", 50),
            enable_deduplication=get_env_bool("ORION_PREPROCESSING_ENABLE_DEDUPLICATION", True),
            enable_normalization=get_env_bool("ORION_PREPROCESSING_ENABLE_NORMALIZATION", True),
        )

    def validate(self) -> None:
        """Validate preprocessing configuration"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.min_text_length <= 0:
            raise ValueError("min_text_length must be positive")


@dataclass
class RetrievalConfig(BaseConfig):
    """Document retrieval configuration"""

    default_k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.2
    enable_reranking: bool = True

    enable_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # MMR (Maximal Marginal Relevance)
    enable_mmr: bool = True
    mmr_diversity_bias: float = 0.5
    mmr_fetch_k: int = 20
    mmr_threshold: float = 0.475

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Load retrieval configuration from environment variables"""
        return cls(
            default_k=get_env_int("ORION_RETRIEVAL_DEFAULT_K", 5),
            max_k=get_env_int("ORION_RETRIEVAL_MAX_K", 20),
            similarity_threshold=get_env_float("ORION_RETRIEVAL_SIMILARITY_THRESHOLD", 0.2),
            enable_reranking=get_env_bool("ORION_RETRIEVAL_ENABLE_RERANKING", True),
            enable_hybrid_search=get_env_bool("ORION_RETRIEVAL_ENABLE_HYBRID_SEARCH", True),
            semantic_weight=get_env_float("ORION_RETRIEVAL_SEMANTIC_WEIGHT", 0.7),
            keyword_weight=get_env_float("ORION_RETRIEVAL_KEYWORD_WEIGHT", 0.3),
            enable_mmr=get_env_bool("ORION_RETRIEVAL_ENABLE_MMR", True),
            mmr_diversity_bias=get_env_float("ORION_RETRIEVAL_MMR_DIVERSITY_BIAS", 0.5),
            mmr_fetch_k=get_env_int("ORION_RETRIEVAL_MMR_FETCH_K", 20),
            mmr_threshold=get_env_float("ORION_RETRIEVAL_MMR_THRESHOLD", 0.6),
        )

    def validate(self) -> None:
        """Validate retrieval configuration"""
        if self.default_k <= 0:
            raise ValueError("default_k must be positive")
        if self.max_k < self.default_k:
            raise ValueError("max_k must be >= default_k")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.enable_hybrid_search:
            if not 0.0 <= self.semantic_weight <= 1.0:
                raise ValueError("semantic_weight must be between 0.0 and 1.0")
            if not 0.0 <= self.keyword_weight <= 1.0:
                raise ValueError("keyword_weight must be between 0.0 and 1.0")
            if abs(self.semantic_weight + self.keyword_weight - 1.0) > 0.001:
                raise ValueError("semantic_weight + keyword_weight must equal 1.0")
        if self.enable_mmr:
            if not 0.0 <= self.mmr_diversity_bias <= 1.0:
                raise ValueError("mmr_diversity_bias must be between 0.0 and 1.0")
            if self.mmr_fetch_k < self.default_k:
                raise ValueError("mmr_fetch_k must be >= default_k")


@dataclass
class RerankerConfig(BaseConfig):
    """Document reranking configuration"""

    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    timeout: int = 30
    top_k: int = 10
    score_threshold: float = 0.5

    # Caching and batch processing
    batch_analysis_size: int = 10
    enable_batch_processing: bool = True

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        """Load reranker configuration from environment variables"""
        return cls(
            model=get_env_str("ORION_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            batch_size=get_env_int("ORION_RERANKER_BATCH_SIZE", 16),
            timeout=get_env_int("ORION_RERANKER_TIMEOUT", 30),
            top_k=get_env_int("ORION_RERANKER_TOP_K", 10),
            score_threshold=get_env_float("ORION_RERANKER_SCORE_THRESHOLD", 0.5),
            batch_analysis_size=get_env_int("ORION_RERANKER_BATCH_ANALYSIS_SIZE", 10),
            enable_batch_processing=get_env_bool("ORION_RERANKER_ENABLE_BATCH_PROCESSING", True),
        )

    def validate(self) -> None:
        """Validate reranker configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")

@dataclass
class LLMConfig(BaseConfig):
    """Large Language Model configuration"""

    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    timeout: int = 90

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None

    # RAG prompt
    system_prompt: str = (
        "You are Orion, a helpful AI assistant with access to a knowledge base. "
        "Use the provided context to answer questions accurately and cite sources when appropriate."
    )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables"""
        max_tokens_str = get_env_str("ORION_LLM_MAX_TOKENS", "")
        max_tokens = int(max_tokens_str) if max_tokens_str.isdigit() else None

        return cls(
            model=get_env_str("ORION_LLM_MODEL", "mistral:latest"),
            base_url=get_env_str("ORION_LLM_BASE_URL", "http://localhost:11434"),
            timeout=get_env_int("ORION_LLM_TIMEOUT", 90),
            temperature=get_env_float("ORION_LLM_TEMPERATURE", 0.7),
            top_p=get_env_float("ORION_LLM_TOP_P", 0.9),
            max_tokens=max_tokens,
            system_prompt=get_env_str(
                "ORION_LLM_SYSTEM_PROMPT",
                "You are Orion, a helpful AI assistant with access to a knowledge base. "
                "Use the provided context to answer questions accurately and cite sources when appropriate.",
            ),
        )


@dataclass
class VectorStoreConfig(BaseConfig):
    """Vector store configuration"""

    index_type: str = "chroma"
    collection_name: str = "orion_knowledge_base"
    persist_immediately: bool = True
    persist_directory: str = "./data/chroma-data"
    distance_metric: str = "cosine"
    use_gpu: bool = False
    batch_size: int = 64

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Load vector store configuration from environment variables"""
        return cls(
            index_type=get_env_str("ORION_VECTORSTORE_INDEX_TYPE", "chroma"),
            collection_name=get_env_str("ORION_VECTORSTORE_COLLECTION_NAME", "orion_knowledge_base"),
            persist_immediately=get_env_bool("ORION_VECTORSTORE_PERSIST_IMMEDIATELY", True),
            persist_directory=get_env_str("ORION_VECTORSTORE_PERSIST_DIRECTORY", "./data/chroma-data"),
            distance_metric=get_env_str("ORION_VECTORSTORE_DISTANCE_METRIC", "cosine"),
            use_gpu=get_env_bool("ORION_VECTORSTORE_USE_GPU", False),
            batch_size=get_env_int("ORION_VECTORSTORE_BATCH_SIZE", 64),
        )


# ========== MAIN RAG CONFIGURATION ==========
@dataclass
class RAGConfig(BaseConfig):
    """Complete RAG pipeline configuration"""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load RAG configuration from environment variables"""
        return cls(
            embedding=EmbeddingConfig.from_env(),
            preprocessing=PreprocessingConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            llm=LLMConfig.from_env(),
            vectorstore=VectorStoreConfig.from_env(),
        )


# ========== SYSTEM CONFIGURATION ==========
@dataclass
class StorageConfig(BaseConfig):
    """Data storage configuration"""

    data_directory: str = "./orion-data"
    temp_directory: str = "./temp"
    knowledge_base_directory: str = "./data/knowledge_base"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load storage configuration from environment variables"""
        return cls(
            data_directory=get_env_str("ORION_STORAGE_DATA_DIRECTORY", "./orion-data"),
            temp_directory=get_env_str("ORION_STORAGE_TEMP_DIRECTORY", "./temp"),
            knowledge_base_directory=get_env_str("ORION_STORAGE_KNOWLEDGE_BASE_DIRECTORY", "./data/knowledge_base"),
        )


@dataclass
class SystemConfig(BaseConfig):
    """System configuration"""

    storage: StorageConfig = field(default_factory=StorageConfig)

    # Ollama integration
    require_ollama: bool = True
    ollama_health_check: bool = True

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load system configuration from environment variables"""
        return cls(
            storage=StorageConfig.from_env(),
            require_ollama=get_env_bool("ORION_SYSTEM_REQUIRE_OLLAMA", True),
            ollama_health_check=get_env_bool("ORION_SYSTEM_OLLAMA_HEALTH_CHECK", True),
        )


# ========== GPU CONFIGURATION ==========
@dataclass
class GPUConfig(BaseConfig):
    """GPU acceleration configuration"""

    enabled: bool = True
    auto_detect: bool = True
    preferred_device: str = "auto"  # "auto", "cpu", "cuda:0"
    fallback_to_cpu: bool = True

    @classmethod
    def from_env(cls) -> "GPUConfig":
        """Load GPU configuration from environment variables"""
        return cls(
            enabled=get_env_bool("ORION_GPU_ENABLED", True),
            auto_detect=get_env_bool("ORION_GPU_AUTO_DETECT", True),
            preferred_device=get_env_str("ORION_GPU_PREFERRED_DEVICE", "auto"),
            fallback_to_cpu=get_env_bool("ORION_GPU_FALLBACK_TO_CPU", True),
        )


# ========== LOGGING CONFIGURATION ==========
@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration"""

    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "./logs/orion.log"
    log_to_console: bool = True
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load logging configuration from environment variables"""
        return cls(
            level=get_env_enum("ORION_LOGGING_LEVEL", LogLevel, LogLevel.INFO),
            log_to_file=get_env_bool("ORION_LOGGING_LOG_TO_FILE", True),
            log_file_path=get_env_str("ORION_LOGGING_LOG_FILE_PATH", "./logs/orion.log"),
            log_to_console=get_env_bool("ORION_LOGGING_LOG_TO_CONSOLE", True),
            verbose=get_env_bool("ORION_LOGGING_VERBOSE", False),
        )


# ========== MAIN CONFIGURATION CLASS ==========
@dataclass
class OrionConfig(BaseConfig):
    """Complete ORION configuration"""

    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "OrionConfig":
        """Load configuration from environment variables"""
        config = cls(
            rag=RAGConfig.from_env(),
            system=SystemConfig.from_env(),
            gpu=GPUConfig.from_env(),
            logging=LoggingConfig.from_env(),
            version=get_env_str("ORION_VERSION", "1.0.0"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate entire configuration"""
        self.rag.preprocessing.validate()
        self.rag.chunking.validate()
        self.rag.retrieval.validate()
        # Add more validations as needed


def get_config(from_env: bool = False) -> OrionConfig:
    """
    Get a configuration instance.

    Args:
        from_env: If True, load configuration from environment variables.
                 If False, use default values.

    Returns:
        OrionConfig instance with specified settings

    Example:
        # Use default configuration
        config = get_config()

        # Load from environment variables
        config = get_config(from_env=True)

        # Or directly
        config = OrionConfig.from_env()

        Environment Variables:
        RAG Configuration:
            ORION_EMBEDDING_MODEL="nomic-embed-text"
            ORION_EMBEDDING_BATCH_SIZE=32
            ORION_CHUNK_SIZE=128
            ORION_CHUNK_OVERLAP=0
            ORION_CHUNKING_STRATEGY="recursive"
            ORION_LLM_MODEL="mistral:latest"
            ORION_LLM_BASE_URL="http://localhost:11434"
            ORION_LLM_TEMPERATURE=0.7
            ORION_VECTORSTORE_COLLECTION_NAME="oRION_knowledge_base"
            ORION_VECTORSTORE_PERSIST_DIRECTORY="./data/chroma-data"
            ORION_RETRIEVAL_DEFAULT_K=5        System Configuration:
            ORION_STORAGE_DATA_DIRECTORY="./oRION-data"
            ORION_SYSTEM_REQUIRE_OLLAMA=true
            ORION_GPU_ENABLED=true
            ORION_LOGGING_LEVEL="info"
    """
    if from_env:
        return OrionConfig.from_env()
    return OrionConfig()