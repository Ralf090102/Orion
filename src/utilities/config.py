import os
from src.utilities.utils import log_error
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# Disable HuggingFace symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


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

    model: str = "all-MiniLM-L12-v2"
    batch_size: int = 64
    timeout: int = 30
    cache_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding configuration from environment variables"""
        return cls(
            model=get_env_str("ORION_EMBEDDING_MODEL", "all-MiniLM-L12-v2"),
            batch_size=get_env_int("ORION_EMBEDDING_BATCH_SIZE", 64),
            timeout=get_env_int("ORION_EMBEDDING_TIMEOUT", 30),
            cache_embeddings=get_env_bool("ORION_EMBEDDING_CACHE", True),
        )


@dataclass
class ChunkingConfig(BaseConfig):
    """Document chunking configuration"""

    strategy: ChunkerType = ChunkerType.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 128
    max_chunk_size: int = 512
    min_chunk_size: int = 256

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """Load chunking configuration from environment variables"""
        return cls(
            strategy=get_env_enum("ORION_CHUNKING_STRATEGY", ChunkerType, ChunkerType.RECURSIVE),
            chunk_size=get_env_int("ORION_CHUNK_SIZE", 512),
            chunk_overlap=get_env_int("ORION_CHUNK_OVERLAP", 128),
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
    semantic_weight: float = 0.8
    keyword_weight: float = 0.2

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
            semantic_weight=get_env_float("ORION_RETRIEVAL_SEMANTIC_WEIGHT", 0.8),
            keyword_weight=get_env_float("ORION_RETRIEVAL_KEYWORD_WEIGHT", 0.2),
            enable_mmr=get_env_bool("ORION_RETRIEVAL_ENABLE_MMR", True),
            mmr_diversity_bias=get_env_float("ORION_RETRIEVAL_MMR_DIVERSITY_BIAS", 0.5),
            mmr_fetch_k=get_env_int("ORION_RETRIEVAL_MMR_FETCH_K", 20),
            mmr_threshold=get_env_float("ORION_RETRIEVAL_MMR_THRESHOLD", 0.475),
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

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
            model=get_env_str("ORION_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
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


@dataclass
class GenerationConfig(BaseConfig):
    """Text generation and prompt configuration"""

    # Generation mode
    mode: str = "rag"  # "rag" or "chat"

    # RAG mode settings
    enable_citations: bool = True
    citation_format: str = "[{index}]"  # or "({source})" or "[{source}]"
    max_context_chunks: int = 5

    # Chat mode settings
    max_history_messages: int = 10  # Number of conversation turns to keep
    enable_rag_augmentation: bool = True  # Use RAG in chat mode
    rag_trigger_mode: str = "auto"  # "always", "auto", "manual", "never"

    # Context window management
    max_total_tokens: int = 4096  # Model context limit
    reserve_tokens_for_response: int = 1024  # Reserve for LLM response
    chars_per_token: int = 4  # Rough estimation: 4 chars ≈ 1 token

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        """Load generation configuration from environment variables"""
        return cls(
            mode=get_env_str("ORION_GENERATION_MODE", "rag"),
            enable_citations=get_env_bool("ORION_GENERATION_CITATIONS", True),
            citation_format=get_env_str("ORION_GENERATION_CITATION_FORMAT", "[{index}]"),
            max_context_chunks=get_env_int("ORION_GENERATION_MAX_CONTEXT_CHUNKS", 5),
            max_history_messages=get_env_int("ORION_GENERATION_MAX_HISTORY", 10),
            enable_rag_augmentation=get_env_bool("ORION_GENERATION_RAG_AUGMENTATION", True),
            rag_trigger_mode=get_env_str("ORION_GENERATION_RAG_TRIGGER", "auto"),
            max_total_tokens=get_env_int("ORION_GENERATION_MAX_TOKENS", 4096),
            reserve_tokens_for_response=get_env_int("ORION_GENERATION_RESERVE_TOKENS", 1024),
            chars_per_token=get_env_int("ORION_GENERATION_CHARS_PER_TOKEN", 4),
        )

    def validate(self) -> None:
        """Validate generation configuration"""
        if self.mode not in ("rag", "chat"):
            raise ValueError("mode must be 'rag' or 'chat'")
        if self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        if self.max_history_messages <= 0:
            raise ValueError("max_history_messages must be positive")
        if self.rag_trigger_mode not in ("always", "auto", "manual", "never"):
            raise ValueError("rag_trigger_mode must be 'always', 'auto', 'manual', or 'never'")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.reserve_tokens_for_response <= 0:
            raise ValueError("reserve_tokens_for_response must be positive")
        if self.reserve_tokens_for_response >= self.max_total_tokens:
            raise ValueError("reserve_tokens_for_response must be less than max_total_tokens")


# ========== WATCHDOG / FILE-WATCHER CONFIGURATION ==========
@dataclass
class WatchdogConfig(BaseConfig):
    """Configuration for filesystem watchers (watchdog)."""

    enabled: bool = False
    paths: list[str] = field(default_factory=lambda: ["./data/knowledge_base"])
    recursive: bool = True
    debounce_seconds: float = 1.0
    ignore_patterns: list[str] = field(default_factory=lambda: ["*.tmp", "*.swp", "__pycache__/*"])
    max_workers: int = 2

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        """Load watchdog configuration from environment variables."""
        raw_paths = get_env_str("ORION_WATCHDOG_PATHS", "")
        paths = [p.strip() for p in raw_paths.split(os.pathsep) if p.strip()] if raw_paths else []
        if not paths:
            paths = [get_env_str("ORION_WATCHDOG_DEFAULT_PATH", "./data/knowledge_base")]

        return cls(
            enabled=get_env_bool("ORION_WATCHDOG_ENABLED", False),
            paths=paths,
            recursive=get_env_bool("ORION_WATCHDOG_RECURSIVE", True),
            debounce_seconds=get_env_float("ORION_WATCHDOG_DEBOUNCE_SECONDS", 1.0),
            ignore_patterns=[p.strip() for p in get_env_str("ORION_WATCHDOG_IGNORE_PATTERNS", "*.tmp;*.swp;__pycache__/*").split(";") if p.strip()],
            max_workers=get_env_int("ORION_WATCHDOG_MAX_WORKERS", 2),
        )

    def validate(self) -> None:
        """Validate watchdog configuration values."""
        if self.debounce_seconds < 0:
            raise ValueError("debounce_seconds must be >= 0")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not isinstance(self.paths, list) or not all(isinstance(p, str) for p in self.paths):
            raise ValueError("paths must be a list of strings")



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
    generation: GenerationConfig = field(default_factory=GenerationConfig)

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
            generation=GenerationConfig.from_env(),
        )


# ========== SYSTEM CONFIGURATION ==========
@dataclass
class StorageConfig(BaseConfig):
    """Data storage configuration"""

    data_directory: str = "./orion-data"
    temp_directory: str = "./temp"
    knowledge_base_directory: str = "D:/Database"

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

    enabled: bool = False
    auto_detect: bool = True
    preferred_device: str = "auto"  # "auto", "cpu", "cuda:0"
    fallback_to_cpu: bool = True

    @classmethod
    def from_env(cls) -> "GPUConfig":
        """Load GPU configuration from environment variables"""
        return cls(
            enabled=get_env_bool("ORION_GPU_ENABLED", False),
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

# ========== TIMING BREAKDOWN ==========
@dataclass
class TimingBreakdown:
    """Container for component timing information."""

    # Retrieval components
    embedding_time: float = 0.0
    search_time: float = 0.0
    reranking_time: float = 0.0
    mmr_time: float = 0.0

    # Generation components
    context_preparation_time: float = 0.0
    prompt_building_time: float = 0.0
    llm_generation_time: float = 0.0

    total_time: float = 0.0

    def get_percentages(self) -> dict[str, float]:
        """Calculate percentage breakdown of timing."""
        if self.total_time == 0:
            return {
                "embedding_percent": 0.0,
                "search_percent": 0.0,
                "reranking_percent": 0.0,
                "mmr_percent": 0.0,
                "context_preparation_percent": 0.0,
                "prompt_building_percent": 0.0,
                "llm_generation_percent": 0.0,
            }

        return {
            "embedding_percent": (self.embedding_time / self.total_time) * 100,
            "search_percent": (self.search_time / self.total_time) * 100,
            "reranking_percent": (self.reranking_time / self.total_time) * 100,
            "mmr_percent": (self.mmr_time / self.total_time) * 100,
            "context_preparation_percent": (self.context_preparation_time / self.total_time) * 100,
            "prompt_building_percent": (self.prompt_building_time / self.total_time) * 100,
            "llm_generation_percent": (self.llm_generation_time / self.total_time) * 100,
        }

    def format_timing_summary(self) -> str:
        """Format timing breakdown as a readable string."""
        retrieval_sum = self.embedding_time + self.search_time + self.reranking_time + self.mmr_time
        generation_sum = self.context_preparation_time + self.prompt_building_time + self.llm_generation_time
        component_sum = retrieval_sum + generation_sum

        percentages = self.get_percentages_from_components()

        lines = [
            "Timing Breakdown",
            f"embedding time = {self.embedding_time:.2f} s",
            f"search time = {self.search_time:.2f} s",
            f"reranking time = {self.reranking_time:.2f} s",
        ]

        if self.mmr_time > 0:
            lines.append(f"mmr time = {self.mmr_time:.2f} s")

        if generation_sum > 0:
            lines.extend(
                [
                    f"context preparation time = {self.context_preparation_time:.2f} s",
                    f"prompt building time = {self.prompt_building_time:.2f} s",
                    f"llm generation time = {self.llm_generation_time:.2f} s",
                ]
            )

        lines.extend(
            [
                f"total time = {component_sum:.2f} s",
                "",
                "Time % Breakdown",
                f"embedding time = {percentages['embedding_percent']:.2f}%",
                f"search time = {percentages['search_percent']:.2f}%",
                f"reranking time = {percentages['reranking_percent']:.2f}%",
            ]
        )

        if self.mmr_time > 0:
            lines.append(f"mmr time = {percentages['mmr_percent']:.2f}%")

        if generation_sum > 0:
            lines.extend(
                [
                    f"context preparation time = {percentages['context_preparation_percent']:.2f}%",
                    f"prompt building time = {percentages['prompt_building_percent']:.2f}%",
                    f"llm generation time = {percentages['llm_generation_percent']:.2f}%",
                ]
            )

        return "\n".join(lines)

    def get_percentages_from_components(self) -> dict[str, float]:
        """Calculate percentage breakdown based on component times only."""
        retrieval_sum = self.embedding_time + self.search_time + self.reranking_time + self.mmr_time
        generation_sum = self.context_preparation_time + self.prompt_building_time + self.llm_generation_time
        component_sum = retrieval_sum + generation_sum

        if component_sum == 0:
            return {
                "embedding_percent": 0.0,
                "search_percent": 0.0,
                "reranking_percent": 0.0,
                "mmr_percent": 0.0,
                "context_preparation_percent": 0.0,
                "prompt_building_percent": 0.0,
                "llm_generation_percent": 0.0,
            }

        return {
            "embedding_percent": (self.embedding_time / component_sum) * 100,
            "search_percent": (self.search_time / component_sum) * 100,
            "reranking_percent": (self.reranking_time / component_sum) * 100,
            "mmr_percent": (self.mmr_time / component_sum) * 100,
            "context_preparation_percent": (self.context_preparation_time / component_sum) * 100,
            "prompt_building_percent": (self.prompt_building_time / component_sum) * 100,
            "llm_generation_percent": (self.llm_generation_time / component_sum) * 100,
        }

# ========== MAIN CONFIGURATION CLASS ==========
@dataclass
class OrionConfig(BaseConfig):
    """Complete ORION configuration"""

    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "OrionConfig":
        """Load configuration from environment variables"""
        config = cls(
            rag=RAGConfig.from_env(),
            system=SystemConfig.from_env(),
            gpu=GPUConfig.from_env(),
            watchdog=WatchdogConfig.from_env(),
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
        try:
            self.rag.embedding.validate()
        except Exception as e:
            log_error(f"Embedding config validation failed: {e}")
            raise

        try:
            self.rag.reranker.validate()
        except Exception as e:
            log_error(f"Reranker config validation failed: {e}")
            raise
        try:
            self.rag.vectorstore.validate()
        except Exception as e:
            log_error(f"Vector store config validation failed: {e}")
            raise
        try:
            self.rag.generation.validate()
        except Exception as e:
            log_error(f"Generation config validation failed: {e}")
            raise
        try:    
            self.watchdog.validate()
        except Exception as e:
            log_error(f"Watchdog config validation failed: {e}")
            raise
        

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
            ORION_EMBEDDING_BATCH_SIZE=64
            ORION_CHUNK_SIZE=128
            ORION_CHUNK_OVERLAP=0
            ORION_CHUNKING_STRATEGY="recursive"
            ORION_LLM_MODEL="mistral:latest"
            ORION_LLM_BASE_URL="http://localhost:11434"
            ORION_LLM_TEMPERATURE=0.7
            ORION_VECTORSTORE_COLLECTION_NAME="Orion_knowledge_base"
            ORION_VECTORSTORE_PERSIST_DIRECTORY="./data/chroma-data"
            ORION_RETRIEVAL_DEFAULT_K=5        System Configuration:
            ORION_STORAGE_DATA_DIRECTORY="./Orion-data"
            ORION_SYSTEM_REQUIRE_OLLAMA=true
            ORION_GPU_ENABLED=true
            ORION_LOGGING_LEVEL="info"
    """
    if from_env:
        return OrionConfig.from_env()
    return OrionConfig()