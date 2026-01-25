"""
Reranker module for Orion.

Provides document reranking using cross-encoder models for improved
relevance ranking after initial retrieval.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from sentence_transformers import CrossEncoder

from src.utilities.utils import ensure_config, log_debug, log_error, log_info, log_warning, timer

if TYPE_CHECKING:
    from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)


class Document:
    """Document class for reranking operations."""

    def __init__(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        score: float = 0.0,
        doc_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
        self.doc_id = doc_id or ""

    def __repr__(self) -> str:
        return f"Document(content='{self.content[:50]}...', score={self.score:.3f}, id={self.doc_id})"


class RerankerManager:
    """
    Manages document reranking using cross-encoder models.
    Supports any sentence-transformers compatible cross-encoder.
    """

    def __init__(self, config: Optional["OrionConfig"] = None):
        """
        Initialize the reranker manager.

        Args:
            config: Orion configuration. If None, uses default configuration.
        """
        self.config = ensure_config(config)
        self.reranker_config = self.config.rag.reranker
        self.model: Optional[CrossEncoder] = None
        self._device = "cuda" if (self.config.gpu.enabled and torch.cuda.is_available()) else "cpu"

        log_info(f"Initializing reranker with model: {self.reranker_config.model}", config=self.config)
        log_info(f"Using device: {self._device}", config=self.config)

    def _initialize_model(self) -> None:
        """Initialize the reranker model if not already loaded."""
        if self.model is None:
            try:
                log_info(f"Loading cross-encoder model: {self.reranker_config.model}", config=self.config)
                self.model = CrossEncoder(self.reranker_config.model, device=self._device, max_length=512)
                log_info("Cross-encoder model loaded successfully", config=self.config)
            except Exception as e:
                log_error(f"Failed to load cross-encoder model: {e}", config=self.config)
                raise RuntimeError(f"Could not initialize reranker model: {e}")

    @timer
    def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Rerank documents based on relevance to query using cross-encoder.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return. If None, uses config.rag.reranker.top_k

        Returns:
            List of reranked documents sorted by relevance score (highest first)
        """
        if not documents:
            log_warning("No documents provided for reranking", config=self.config)
            return []

        if not query.strip():
            log_warning("Empty query provided for reranking", config=self.config)
            return documents

        self._initialize_model()

        if self.model is None:
            log_error("Reranker model not available", config=self.config)
            return documents

        top_k = top_k or self.reranker_config.top_k
        log_info(f"Reranking {len(documents)} documents with top_k={top_k}", config=self.config)

        try:
            # Prepare query-document pairs
            pairs = [(query, doc.content) for doc in documents]

            # Batch processing for efficiency
            batch_size = self.reranker_config.batch_size
            all_scores = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                batch_scores = self.model.predict(batch_pairs)
                
                # Ensure scores are floats
                if hasattr(batch_scores, "__iter__") and not isinstance(batch_scores, str):
                    all_scores.extend([float(score) for score in batch_scores])
                else:
                    all_scores.append(float(batch_scores))

            # Create reranked documents with updated scores
            reranked_docs = []
            for doc, score in zip(documents, all_scores):
                reranked_doc = Document(
                    content=doc.content,
                    metadata=doc.metadata.copy() if doc.metadata else {},
                    score=float(score),
                    doc_id=doc.doc_id,
                )
                
                # Add reranking metadata
                if reranked_doc.metadata is not None:
                    reranked_doc.metadata["rerank_score"] = float(score)
                    reranked_doc.metadata["original_score"] = doc.score
                
                reranked_docs.append(reranked_doc)

            # Filter by score threshold
            filtered_docs = [
                doc for doc in reranked_docs
                if doc.score >= self.reranker_config.score_threshold
            ]

            # Sort by score (highest first)
            filtered_docs.sort(key=lambda x: x.score, reverse=True)

            # Return top_k results
            result = filtered_docs[:top_k]

            log_info(
                f"Reranking complete: {len(result)} documents returned "
                f"(filtered {len(reranked_docs) - len(filtered_docs)} below threshold {self.reranker_config.score_threshold})",
                config=self.config
            )

            if result:
                log_debug(
                    f"Top result score: {result[0].score:.4f}, Bottom result score: {result[-1].score:.4f}",
                    self.config
                )

            return result

        except Exception as e:
            log_error(f"Error during reranking: {e}", config=self.config)
            # Fallback: return original documents sorted by original score
            return sorted(documents, key=lambda x: x.score, reverse=True)[:top_k]

    def rerank_search_results(
        self,
        query: str,
        search_results: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank search results from the retrieval pipeline.

        Args:
            query: Search query
            search_results: List of search result dictionaries
            top_k: Number of top results to return

        Returns:
            List of reranked search results
        """
        if not search_results:
            return []

        # Convert search results to Document objects
        documents = []
        for result in search_results:
            doc = Document(
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score", 0.0),
                doc_id=result.get("id", ""),
            )
            documents.append(doc)

        # Rerank documents
        reranked_docs = self.rerank_documents(query, documents, top_k)

        # Convert back to search result format
        reranked_results = []
        for doc in reranked_docs:
            result = {
                "id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            reranked_results.append(result)

        return reranked_results

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the reranker model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_name": self.reranker_config.model,
            "device": self._device,
            "batch_size": self.reranker_config.batch_size,
            "score_threshold": self.reranker_config.score_threshold,
            "top_k": self.reranker_config.top_k,
            "model_loaded": self.model is not None,
        }

        if self.model is not None:
            info["max_length"] = getattr(self.model, "max_length", "unknown")

        return info


def create_reranker(config: Optional["OrionConfig"] = None) -> RerankerManager:
    """
    Factory function to create a reranker manager.

    Args:
        config: Orion configuration. If None, uses default configuration.

    Returns:
        Configured RerankerManager instance
    """
    return RerankerManager(config)


def rerank_documents(
    query: str,
    documents: list[Document],
    config: Optional["OrionConfig"] = None,
    top_k: Optional[int] = None,
) -> list[Document]:
    """
    Convenience function to rerank documents.

    Args:
        query: Search query
        documents: List of documents to rerank
        config: Orion configuration
        top_k: Number of top documents to return

    Returns:
        List of reranked documents
    """
    reranker = create_reranker(config)
    return reranker.rerank_documents(query, documents, top_k)


def rerank_search_results(
    query: str,
    search_results: list[dict[str, Any]],
    config: Optional["OrionConfig"] = None,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        search_results: List of search result dictionaries
        config: Orion configuration
        top_k: Number of top results to return

    Returns:
        List of reranked search results
    """
    reranker = create_reranker(config)
    return reranker.rerank_search_results(query, search_results, top_k)
