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

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt"}

DEFAULT_VECTORSTORE_PATH = "vectorstore"
DEFAULT_DATA_PATH = "data"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 30

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = None


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
    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_timeout: int = OLLAMA_TIMEOUT

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
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT)),
        )
