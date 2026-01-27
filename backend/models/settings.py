"""
Pydantic models for Settings API endpoints.

Models for reading and updating system configuration dynamically.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ========== EMBEDDING SETTINGS ==========
class EmbeddingSettings(BaseModel):
    """Embedding model settings (read-only)."""

    model: str = Field(..., description="Embedding model name")
    batch_size: int = Field(..., description="Batch size for embedding generation")
    timeout: int = Field(..., description="Timeout in seconds")
    cache_embeddings: bool = Field(..., description="Whether to cache embeddings")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "all-MiniLM-L12-v2",
                "batch_size": 64,
                "timeout": 30,
                "cache_embeddings": True,
            }
        }


class EmbeddingSettingsUpdate(BaseModel):
    """Embedding settings update (partial)."""

    model: Optional[str] = Field(None, description="Embedding model name")
    batch_size: Optional[int] = Field(None, ge=1, le=256, description="Batch size (1-256)")
    timeout: Optional[int] = Field(None, ge=10, description="Timeout in seconds (min 10)")
    cache_embeddings: Optional[bool] = Field(None, description="Enable embedding cache")


# ========== CHUNKING SETTINGS ==========
class ChunkingSettings(BaseModel):
    """Document chunking settings (read-only)."""

    strategy: str = Field(..., description="Chunking strategy (recursive/semantic/smart)")
    chunk_size: int = Field(..., description="Target chunk size in characters")
    chunk_overlap: int = Field(..., description="Overlap between chunks")
    max_chunk_size: int = Field(..., description="Maximum chunk size")
    min_chunk_size: int = Field(..., description="Minimum chunk size")

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 128,
                "max_chunk_size": 512,
                "min_chunk_size": 256,
            }
        }


class ChunkingSettingsUpdate(BaseModel):
    """Chunking settings update (partial)."""

    strategy: Optional[str] = Field(None, description="Chunking strategy")
    chunk_size: Optional[int] = Field(None, ge=100, le=2048, description="Chunk size (100-2048)")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=512, description="Chunk overlap (0-512)")
    max_chunk_size: Optional[int] = Field(None, ge=100, le=4096, description="Max chunk size")
    min_chunk_size: Optional[int] = Field(None, ge=50, le=1024, description="Min chunk size")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: Optional[str]) -> Optional[str]:
        """Validate chunking strategy."""
        if v is not None and v not in ("recursive", "semantic", "smart"):
            raise ValueError("strategy must be 'recursive', 'semantic', or 'smart'")
        return v


# ========== RETRIEVAL SETTINGS ==========
class RetrievalSettings(BaseModel):
    """Document retrieval settings (read-only)."""

    default_k: int = Field(..., description="Default number of documents to retrieve")
    max_k: int = Field(..., description="Maximum allowed k value")
    similarity_threshold: float = Field(..., description="Minimum similarity score")
    enable_reranking: bool = Field(..., description="Enable cross-encoder reranking")
    enable_hybrid_search: bool = Field(..., description="Enable hybrid search (semantic + keyword)")
    semantic_weight: float = Field(..., description="Weight for semantic search (0-1)")
    keyword_weight: float = Field(..., description="Weight for keyword search (0-1)")
    enable_mmr: bool = Field(..., description="Enable MMR diversity")
    mmr_diversity_bias: float = Field(..., description="MMR diversity bias (0=relevance, 1=diversity)")
    mmr_fetch_k: int = Field(..., description="Number of candidates for MMR")
    mmr_threshold: float = Field(..., description="MMR similarity threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "default_k": 5,
                "max_k": 20,
                "similarity_threshold": 0.2,
                "enable_reranking": True,
                "enable_hybrid_search": True,
                "semantic_weight": 0.8,
                "keyword_weight": 0.2,
                "enable_mmr": True,
                "mmr_diversity_bias": 0.5,
                "mmr_fetch_k": 20,
                "mmr_threshold": 0.475,
            }
        }


class RetrievalSettingsUpdate(BaseModel):
    """Retrieval settings update (partial)."""

    default_k: Optional[int] = Field(None, ge=1, le=50, description="Default k (1-50)")
    max_k: Optional[int] = Field(None, ge=1, le=100, description="Max k (1-100)")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")
    enable_reranking: Optional[bool] = Field(None, description="Enable reranking")
    enable_hybrid_search: Optional[bool] = Field(None, description="Enable hybrid search")
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic weight (0-1)")
    keyword_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Keyword weight (0-1)")
    enable_mmr: Optional[bool] = Field(None, description="Enable MMR")
    mmr_diversity_bias: Optional[float] = Field(None, ge=0.0, le=1.0, description="MMR diversity (0-1)")
    mmr_fetch_k: Optional[int] = Field(None, ge=1, le=200, description="MMR fetch k (1-200)")
    mmr_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="MMR threshold (0-1)")


# ========== RERANKER SETTINGS ==========
class RerankerSettings(BaseModel):
    """Reranker settings (read-only)."""

    model: str = Field(..., description="Cross-encoder reranker model")
    batch_size: int = Field(..., description="Reranking batch size")
    timeout: int = Field(..., description="Timeout in seconds")
    top_k: int = Field(..., description="Top k results after reranking")
    score_threshold: float = Field(..., description="Minimum reranker score")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "batch_size": 16,
                "timeout": 30,
                "top_k": 10,
                "score_threshold": 0.5,
            }
        }


class RerankerSettingsUpdate(BaseModel):
    """Reranker settings update (partial)."""

    model: Optional[str] = Field(None, description="Reranker model name")
    batch_size: Optional[int] = Field(None, ge=1, le=128, description="Batch size (1-128)")
    timeout: Optional[int] = Field(None, ge=10, description="Timeout (min 10s)")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Top k (1-50)")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score threshold (0-1)")


# ========== GENERATION SETTINGS ==========
class GenerationSettings(BaseModel):
    """Text generation settings (read-only)."""

    mode: str = Field(..., description="Generation mode (rag/chat)")
    enable_citations: bool = Field(..., description="Include source citations")
    citation_format: str = Field(..., description="Citation format string")
    max_context_chunks: int = Field(..., description="Max context chunks to use")
    validate_citations: bool = Field(..., description="Validate and remove hallucinated citations")
    expand_citations: bool = Field(..., description="Expand citations to full text")
    max_history_messages: int = Field(..., description="Max conversation history (chat mode)")
    enable_rag_augmentation: bool = Field(..., description="Use RAG in chat mode")
    rag_trigger_mode: str = Field(..., description="RAG trigger mode (always/auto/manual/never)")
    max_total_tokens: int = Field(..., description="Model context window size")
    reserve_tokens_for_response: int = Field(..., description="Tokens reserved for response")

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "rag",
                "enable_citations": True,
                "citation_format": "[{index}]",
                "max_context_chunks": 5,
                "validate_citations": True,
                "expand_citations": False,
                "max_history_messages": 10,
                "enable_rag_augmentation": True,
                "rag_trigger_mode": "auto",
                "max_total_tokens": 4096,
                "reserve_tokens_for_response": 1024,
            }
        }


class GenerationSettingsUpdate(BaseModel):
    """Generation settings update (partial)."""

    mode: Optional[str] = Field(None, description="Generation mode")
    enable_citations: Optional[bool] = Field(None, description="Enable citations")
    citation_format: Optional[str] = Field(None, description="Citation format")
    max_context_chunks: Optional[int] = Field(None, ge=1, le=20, description="Max context chunks (1-20)")
    validate_citations: Optional[bool] = Field(None, description="Validate citations")
    expand_citations: Optional[bool] = Field(None, description="Expand citations")
    max_history_messages: Optional[int] = Field(None, ge=1, le=50, description="Max history (1-50)")
    enable_rag_augmentation: Optional[bool] = Field(None, description="Enable RAG in chat")
    rag_trigger_mode: Optional[str] = Field(None, description="RAG trigger mode")
    max_total_tokens: Optional[int] = Field(None, ge=512, le=32768, description="Context window (512-32768)")
    reserve_tokens_for_response: Optional[int] = Field(None, ge=128, le=4096, description="Reserve tokens (128-4096)")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        """Validate generation mode."""
        if v is not None and v not in ("rag", "chat"):
            raise ValueError("mode must be 'rag' or 'chat'")
        return v

    @field_validator("rag_trigger_mode")
    @classmethod
    def validate_rag_trigger_mode(cls, v: Optional[str]) -> Optional[str]:
        """Validate RAG trigger mode."""
        if v is not None and v not in ("always", "auto", "manual", "never"):
            raise ValueError("rag_trigger_mode must be 'always', 'auto', 'manual', or 'never'")
        return v


# ========== VECTOR STORE SETTINGS ==========
class VectorStoreSettings(BaseModel):
    """Vector store settings (read-only)."""

    collection_name: str = Field(..., description="ChromaDB collection name")
    persist_directory: str = Field(..., description="Persistence directory")
    distance_metric: str = Field(..., description="Distance metric (cosine/l2/ip)")
    batch_size: int = Field(..., description="Batch size for operations")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "orion_knowledge_base",
                "persist_directory": "./data/chroma-data",
                "distance_metric": "cosine",
                "batch_size": 64,
            }
        }


class VectorStoreSettingsUpdate(BaseModel):
    """Vector store settings update (partial)."""

    collection_name: Optional[str] = Field(None, min_length=1, description="Collection name")
    distance_metric: Optional[str] = Field(None, description="Distance metric")
    batch_size: Optional[int] = Field(None, ge=1, le=256, description="Batch size (1-256)")

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: Optional[str]) -> Optional[str]:
        """Validate distance metric."""
        if v is not None and v not in ("cosine", "l2", "ip"):
            raise ValueError("distance_metric must be 'cosine', 'l2', or 'ip'")
        return v


# ========== GPU SETTINGS ==========
class GPUSettings(BaseModel):
    """GPU acceleration settings (read-only)."""

    enabled: bool = Field(..., description="GPU acceleration enabled")
    auto_detect: bool = Field(..., description="Auto-detect GPU availability")
    preferred_device: str = Field(..., description="Preferred device (auto/cpu/cuda:0)")
    fallback_to_cpu: bool = Field(..., description="Fallback to CPU if GPU unavailable")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": False,
                "auto_detect": True,
                "preferred_device": "auto",
                "fallback_to_cpu": True,
            }
        }


class GPUSettingsUpdate(BaseModel):
    """GPU settings update (partial)."""

    enabled: Optional[bool] = Field(None, description="Enable GPU")
    auto_detect: Optional[bool] = Field(None, description="Auto-detect GPU")
    preferred_device: Optional[str] = Field(None, description="Preferred device")
    fallback_to_cpu: Optional[bool] = Field(None, description="Fallback to CPU")


# ========== COMPLETE SETTINGS RESPONSE ==========
class SettingsResponse(BaseModel):
    """Complete system settings response."""

    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings
    generation: GenerationSettings
    vectorstore: VectorStoreSettings
    gpu: GPUSettings
    last_updated: Optional[datetime] = Field(None, description="Last settings update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "embedding": {"model": "all-MiniLM-L12-v2", "batch_size": 64},
                "retrieval": {"default_k": 5, "enable_reranking": True},
                "generation": {"mode": "rag", "enable_citations": True},
                "last_updated": "2026-01-25T10:30:00Z",
            }
        }


# ========== SETTINGS UPDATE REQUEST ==========
class SettingsUpdateRequest(BaseModel):
    """Request to update multiple settings categories."""

    embedding: Optional[EmbeddingSettingsUpdate] = None
    chunking: Optional[ChunkingSettingsUpdate] = None
    retrieval: Optional[RetrievalSettingsUpdate] = None
    reranker: Optional[RerankerSettingsUpdate] = None
    generation: Optional[GenerationSettingsUpdate] = None
    vectorstore: Optional[VectorStoreSettingsUpdate] = None
    gpu: Optional[GPUSettingsUpdate] = None

    class Config:
        json_schema_extra = {
            "example": {
                "retrieval": {
                    "default_k": 10,
                    "enable_reranking": False,
                },
            }
        }


# ========== SETTINGS UPDATE RESPONSE ==========
class SettingsUpdateResponse(BaseModel):
    """Response after updating settings."""

    status: str = Field(..., description="Update status")
    message: str = Field(..., description="Human-readable message")
    updated_categories: list[str] = Field(..., description="Categories that were updated")
    requires_restart: list[str] = Field(
        default_factory=list,
        description="Components that require restart/re-initialization",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about settings changes",
    )
    settings: SettingsResponse = Field(..., description="Updated settings")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Settings updated successfully",
                "updated_categories": ["retrieval", "llm"],
                "requires_restart": ["retriever"],
                "warnings": ["Changing chunking settings requires re-ingestion"],
                "settings": {"embedding": {}, "retrieval": {}},
            }
        }


# ========== RESET REQUEST/RESPONSE ==========
class SettingsResetRequest(BaseModel):
    """Request to reset settings."""

    category: str = Field(
        default="all",
        description="Category to reset (all resets everything)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category": "all",
            }
        }


class SettingsResetResponse(BaseModel):
    """Response after resetting settings."""

    status: str = Field(..., description="Reset status")
    message: str = Field(..., description="Human-readable message")
    reset_categories: list[str] = Field(..., description="Categories that were reset")
    settings: SettingsResponse = Field(..., description="Reset settings")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Settings reset to defaults",
                "reset_categories": ["all"],
                "settings": {},
            }
        }


# ========== SAVE/LOAD RESPONSES ==========
class SettingsSaveResponse(BaseModel):
    """Response after saving settings to file."""

    status: str = Field(..., description="Save status")
    message: str = Field(..., description="Human-readable message")
    file_path: str = Field(..., description="Path where settings were saved")
    timestamp: datetime = Field(..., description="Save timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Settings saved successfully",
                "file_path": "./data/settings.json",
                "timestamp": "2026-01-25T10:30:00Z",
            }
        }


class SettingsLoadResponse(BaseModel):
    """Response after loading settings from file."""

    status: str = Field(..., description="Load status")
    message: str = Field(..., description="Human-readable message")
    file_path: str = Field(..., description="Path from which settings were loaded")
    loaded_at: datetime = Field(..., description="Load timestamp")
    settings: SettingsResponse = Field(..., description="Loaded settings")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Settings loaded successfully",
                "file_path": "./data/settings.json",
                "loaded_at": "2026-01-25T10:30:00Z",
                "settings": {},
            }
        }
