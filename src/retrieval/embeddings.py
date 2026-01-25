"""
Embedding generation and management for Orion using sentence-transformers.
Supports any SentenceTransformer model with configurable batching and caching.
"""

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utilities.utils import (
    ensure_config,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
    timer,
)

if TYPE_CHECKING:
    from src.utilities.config import OrionConfig


class EmbeddingManager:
    """
    Manages embedding generation using sentence-transformers models.
    Supports caching, batching, and GPU acceleration.
    """

    def __init__(self, config: Optional["OrionConfig"] = None):
        """
        Initialize embedding manager.

        Args:
            config: Orion configuration object
        """
        self.config = ensure_config(config)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.cache_dir = self._setup_cache_directory()
        self.embedding_cache = {}

        self._load_model()

    def _get_device(self) -> str:
        """Determine device for embedding computation."""
        if self.config.gpu.enabled and torch.cuda.is_available():
            device = "cuda"
            log_info(f"Using GPU for embeddings: {torch.cuda.get_device_name()}", config=self.config)
        else:
            device = "cpu"
            if self.config.gpu.enabled and not torch.cuda.is_available():
                log_warning(
                    "GPU acceleration requested but CUDA not available. "
                    "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu128",
                    config=self.config
                )
            log_info("Using CPU for embeddings", config=self.config)

        return device

    def _setup_cache_directory(self) -> Path | None:
        """Setup embedding cache directory if caching is enabled."""
        if not self.config.rag.embedding.cache_embeddings:
            return None

        try:
            cache_dir = Path(self.config.system.storage.temp_directory) / "embedding_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Embedding cache directory: {cache_dir}", self.config)
            return cache_dir
        except Exception as e:
            log_warning(f"Failed to create cache directory: {e}", config=self.config)
            return None

    def _load_model(self) -> None:
        """Load the embedding model based on configuration."""
        model_name = self.config.rag.embedding.model

        try:
            log_info(f"Loading embedding model: {model_name}", config=self.config)

            # Load any sentence-transformers model
            self.model = SentenceTransformer(model_name, device=self.device)

            # Set to evaluation mode
            self.model.eval()

            # Configure max sequence length if model supports it
            if hasattr(self.model, "max_seq_length"):
                # Use model's default max length or set reasonable limit
                max_length = getattr(self.model, "max_seq_length", 512)
                log_debug(f"Model max sequence length: {max_length}", self.config)

            log_success(f"Successfully loaded model: {model_name} (dim={self.get_embedding_dimension()})", config=self.config)

        except Exception as e:
            log_error(f"Failed to load embedding model {model_name}: {e}", config=self.config)
            raise RuntimeError(f"Embedding model failed to load: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_from_cache(self, cache_key: str) -> list[float] | None:
        """Load embedding from cache if available."""
        if not self.config.rag.embedding.cache_embeddings or not self.cache_dir:
            return None

        # Check memory cache
        if cache_key in self.embedding_cache:
            log_debug("Embedding found in memory cache", self.config)
            return self.embedding_cache[cache_key]

        # Check disk cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)

                    # Store in memory cache
                    self.embedding_cache[cache_key] = embedding

                    log_debug("Embedding found in disk cache", self.config)
                    return embedding
        except Exception as e:
            log_warning(f"Failed to load from cache: {e}", config=self.config)

        return None

    def _save_to_cache(self, cache_key: str, embedding: list[float]) -> None:
        """Save embedding to cache."""
        if not self.config.rag.embedding.cache_embeddings:
            return

        # Save to memory cache
        self.embedding_cache[cache_key] = embedding

        # Save to disk cache
        if self.cache_dir:
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(embedding, f)
                log_debug("Embedding saved to cache", self.config)
            except Exception as e:
                log_warning(f"Failed to save to cache: {e}", config=self.config)

    @timer
    def encode_single(self, text: str) -> list[float]:
        """
        Encode a single text into embedding.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            log_warning("Empty text provided for encoding", config=self.config)
            return []

        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        try:
            # Generate embedding using sentence-transformers
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

            self._save_to_cache(cache_key, embedding)

            log_debug(f"Generated embedding of dimension {len(embedding)}", self.config)
            return embedding

        except Exception as e:
            log_error(f"Failed to encode text: {e}", config=self.config)
            return []

    @timer
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts into embeddings efficiently.

        Args:
            texts: List of input texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            log_warning("Empty text list provided for batch encoding", config=self.config)
            return []

        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            log_warning("No valid texts found for encoding", config=self.config)
            return [[] for _ in texts]

        try:
            batch_size = self.config.rag.embedding.batch_size
            all_embeddings = []

            # Batch Processing
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i : i + batch_size]

                log_debug(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts", self.config)

                # Generate embeddings using sentence-transformers
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                    )

                    if isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = batch_embeddings.tolist()

                    all_embeddings.extend(batch_embeddings)

            result_embeddings = [[] for _ in texts]
            for i, embedding in enumerate(all_embeddings):
                original_index = valid_indices[i]
                result_embeddings[original_index] = embedding

            return result_embeddings

        except Exception as e:
            log_error(f"Failed to encode batch: {e}", config=self.config)
            return [[] for _ in texts]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.

        Returns:
            Embedding dimension
        """
        try:
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                return self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, "encode"):
                # Fallback: encode a test string to determine dimension
                test_embedding = self.model.encode("test", convert_to_tensor=False)
                return len(test_embedding) if isinstance(test_embedding, (list, np.ndarray)) else 768
            else:
                return 768  # Common default for many models

        except Exception as e:
            log_error(f"Failed to get embedding dimension: {e}", config=self.config)
            return 768  # Fallback to common dimension

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        try:
            self.embedding_cache.clear()

            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

            log_info("Embedding cache cleared", config=self.config)

        except Exception as e:
            log_error(f"Failed to clear cache: {e}", config=self.config)

    def get_model_info(self) -> dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.rag.embedding.model,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_enabled": self.config.rag.embedding.cache_embeddings,
            "batch_size": self.config.rag.embedding.batch_size,
        }


def create_embedding_manager(config: Optional["OrionConfig"] = None) -> EmbeddingManager:
    """
    Factory function to create an EmbeddingManager instance.

    Args:
        config: Orion configuration object

    Returns:
        Initialized EmbeddingManager instance
    """
    return EmbeddingManager(config)
