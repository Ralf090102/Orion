"""
Configuration settings for Orion RAG pipeline.
"""

import os
from dataclasses import dataclass

# ---------- Defaults ----------
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "mistral"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 4000

DEFAULT_RETRIEVAL_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.7

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv", ".md", ".rtf",
    # Images (for future OCR/vision)
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
    # Code files
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".php",
    ".html", ".css", ".xml", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sql", ".sh", ".bat", ".ps1", ".r", ".m", ".swift", ".kt", ".dart",
    # Text/Documentation
    ".org", ".rst", ".tex", ".log", ".conf", ".properties",
    # Email (common formats)
    ".eml", ".msg", ".mbox",
    # Archives (we can extract and process contents)
    ".zip", ".tar", ".gz",
}

DEFAULT_VECTORSTORE_PATH = "vectorstore"
DEFAULT_DATA_PATH = "data"
DEFAULT_USER_ID = "default"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 30

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = None

DEFAULT_ENABLE_DEDUPLICATION = True
DEFAULT_ENABLE_NORMALIZATION = True
DEFAULT_DEDUPLICATION_THRESHOLD = 0.9


@dataclass
class Config:
    """Runtime configuration for Orion."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    persist_path: str = DEFAULT_VECTORSTORE_PATH
    user_id: str = DEFAULT_USER_ID
    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_timeout: int = OLLAMA_TIMEOUT
    
    enable_deduplication: bool = DEFAULT_ENABLE_DEDUPLICATION
    enable_normalization: bool = DEFAULT_ENABLE_NORMALIZATION
    deduplication_threshold: float = DEFAULT_DEDUPLICATION_THRESHOLD

    @property
    def user_persist_path(self) -> str:
        """Get the user-specific vectorstore path."""
        return f"{self.persist_path}/{self.user_id}"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            embedding_model=os.getenv("ORION_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            llm_model=os.getenv("ORION_LLM_MODEL", DEFAULT_LLM_MODEL),
            chunk_size=int(os.getenv("ORION_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(os.getenv("ORION_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
            retrieval_k=int(os.getenv("ORION_RETRIEVAL_K", DEFAULT_RETRIEVAL_K)),
            temperature=float(os.getenv("ORION_TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_tokens=(
                int(v) if (v := os.getenv("ORION_MAX_TOKENS")) else DEFAULT_MAX_TOKENS
            ),
            persist_path=os.getenv("ORION_PERSIST_PATH", DEFAULT_VECTORSTORE_PATH),
            user_id=os.getenv("ORION_USER_ID", DEFAULT_USER_ID),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT)),
            enable_deduplication=bool(os.getenv("ORION_ENABLE_DEDUPLICATION", DEFAULT_ENABLE_DEDUPLICATION)),
            enable_normalization=bool(os.getenv("ORION_ENABLE_NORMALIZATION", DEFAULT_ENABLE_NORMALIZATION)),
            deduplication_threshold=float(os.getenv("ORION_DEDUPLICATION_THRESHOLD", DEFAULT_DEDUPLICATION_THRESHOLD)),
        )
