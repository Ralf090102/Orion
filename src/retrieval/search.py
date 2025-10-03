"""
Advanced search strategies for Orion retrieval system.
Implements semantic, hybrid, and MMR search with Filipino cultural content optimization.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.utilities.utils import (
    ensure_config,
    log_debug,
    log_info,
    log_warning,
    timer,
)

if TYPE_CHECKING:
    from src.retrieval.embeddings import EmbeddingManager
    from src.retrieval.vector_store import ChromaVectorStore
    from src.utilities.config import OrionConfig


class SearchResult:
    """Represents a single search result."""

    def __init__(self, document_id: str, content: str, metadata: dict[str, Any], score: float, search_type: str = "semantic"):
        self.document_id = document_id
        self.content = content
        self.metadata = metadata
        self.score = score
        self.search_type = search_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "search_type": self.search_type,
        }


class SemanticSearcher:
    """
    Pure vector similarity search using embeddings.
    """

    def __init__(
        self, embedding_manager: "EmbeddingManager", vector_store: "ChromaVectorStore", config: Optional["OrionConfig"] = None
    ):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.config = ensure_config(config)

    @timer
    def search(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Search query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        if not query.strip():
            log_warning("Empty query provided for semantic search", config=self.config)
            return []

        try:
            query_embedding = self.embedding_manager.encode_single(query)
            if not query_embedding:
                log_warning("Failed to generate query embedding", config=self.config)
                return []

            if similarity_threshold is None:
                similarity_threshold = self.config.rag.retrieval.similarity_threshold

            results = self.vector_store.query(query_embeddings=[query_embedding], n_results=k, where=metadata_filter)

            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Lower distance = higher similarity
                    distance = results["distances"][0][i] if results["distances"] else float("inf")

                    # Convert distance to similarity score (0-1 range, higher is better)
                    # Using exponential decay: similarity = exp(-distance)
                    # This naturally handles any distance range and keeps scores in 0-1
                    import math

                    if distance == 0:
                        similarity_score = 1.0  # Perfect match
                    elif distance == float("inf"):
                        similarity_score = 0.0  # No match
                    else:
                        # Exponential decay provides smooth similarity scores
                        similarity_score = math.exp(-distance)

                    # Apply threshold filter
                    if similarity_score >= similarity_threshold:
                        search_result = SearchResult(
                            document_id=results["ids"][0][i],
                            content=results["documents"][0][i] if results["documents"] else "",
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                            score=similarity_score,
                            search_type="semantic",
                        )
                        search_results.append(search_result)

            log_debug(
                f"Semantic search returned {len(search_results)} results above threshold {similarity_threshold}", self.config
            )
            return search_results

        except Exception as e:
            log_warning(f"Semantic search failed: {e}", config=self.config)
            return []


class KeywordSearcher:
    """
    BM25-based keyword search for exact term matching.
    Handles both English and Filipino text effectively.
    """

    def __init__(self, config: Optional["OrionConfig"] = None):
        self.config = ensure_config(config)
        self.bm25 = None
        self.documents = []
        self.document_metadata = []
        self.document_ids = []

    def index_documents(self, documents: list[str], document_ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        """
        Index documents for keyword search.

        Args:
            documents: List of document texts
            document_ids: List of document IDs
            metadatas: List of metadata dictionaries
        """
        if not documents:
            log_warning("No documents provided for keyword indexing", config=self.config)
            return

        try:
            self.documents = documents
            self.document_ids = document_ids
            self.document_metadata = metadatas

            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                tokens = self._tokenize(doc)
                tokenized_docs.append(tokens)

            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)

            log_info(f"Indexed {len(documents)} documents for keyword search", config=self.config)

        except Exception as e:
            log_warning(f"Failed to index documents for keyword search: {e}", config=self.config)

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.
        Handles Filipino and English text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        import re

        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)

        # Filter out short tokens
        tokens = [token for token in tokens if len(token) >= 2]

        return tokens

    @timer
    def search(
        self, query: str, k: int = 5, score_threshold: float = 0.0, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Perform keyword search using BM25.

        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum BM25 score threshold
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects sorted by BM25 score (highest first)
        """
        if not query.strip() or self.bm25 is None:
            log_warning("Empty query or BM25 not initialized", config=self.config)
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                log_warning("No valid tokens in query", config=self.config)
                return []

            scores = self.bm25.get_scores(query_tokens)

            results_with_scores = []
            for i, score in enumerate(scores):
                if score >= score_threshold:
                    # Apply metadata filter
                    if metadata_filter:
                        doc_metadata = self.document_metadata[i]
                        if not self._matches_filter(doc_metadata, metadata_filter):
                            continue

                    search_result = SearchResult(
                        document_id=self.document_ids[i],
                        content=self.documents[i],
                        metadata=self.document_metadata[i],
                        score=float(score),
                        search_type="keyword",
                    )
                    results_with_scores.append(search_result)

            # Sort by score
            results_with_scores.sort(key=lambda x: x.score, reverse=True)
            results = results_with_scores[:k]

            log_debug(f"Keyword search returned {len(results)} results above threshold {score_threshold}", self.config)
            return results

        except Exception as e:
            log_warning(f"Keyword search failed: {e}", config=self.config)
            return []

    def _matches_filter(self, metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class HybridSearcher:
    """
    Combines semantic and keyword search with configurable weights.
    Provides comprehensive retrieval for Filipino cultural content.
    """

    def __init__(
        self, semantic_searcher: SemanticSearcher, keyword_searcher: KeywordSearcher, config: Optional["OrionConfig"] = None
    ):
        self.semantic_searcher = semantic_searcher
        self.keyword_searcher = keyword_searcher
        self.config = ensure_config(config)

    @timer
    def search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword approaches.

        Args:
            query: Search query text
            k: Number of results to return
            semantic_weight: Weight for semantic scores (0.0-1.0)
            keyword_weight: Weight for keyword scores (0.0-1.0)
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects sorted by combined score (highest first)
        """
        if not query.strip():
            log_warning("Empty query provided for hybrid search", config=self.config)
            return []

        if semantic_weight is None:
            semantic_weight = self.config.rag.retrieval.semantic_weight
        if keyword_weight is None:
            keyword_weight = self.config.rag.retrieval.keyword_weight

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            keyword_weight /= total_weight
        else:
            semantic_weight = keyword_weight = 0.5

        try:
            semantic_results = self.semantic_searcher.search(query=query, k=k * 2, metadata_filter=metadata_filter)

            keyword_results = self.keyword_searcher.search(query=query, k=k * 2, metadata_filter=metadata_filter)

            combined_results = self._fuse_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

            final_results = combined_results[:k]

            log_debug(
                f"Hybrid search combined {len(semantic_results)} semantic + {len(keyword_results)} keyword results", self.config
            )
            return final_results

        except Exception as e:
            log_warning(f"Hybrid search failed: {e}", config=self.config)
            return []

    def _fuse_results(
        self,
        semantic_results: list[SearchResult],
        keyword_results: list[SearchResult],
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        """
        Fuse semantic and keyword results using weighted score combination.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores

        Returns:
            Fused and ranked results
        """
        semantic_results = self._normalize_scores(semantic_results)
        keyword_results = self._normalize_scores(keyword_results)

        # Create document ID to result mapping
        combined_docs = {}

        for result in semantic_results:
            doc_id = result.document_id
            combined_docs[doc_id] = {
                "document_id": doc_id,
                "content": result.content,
                "metadata": result.metadata,
                "semantic_score": result.score,
                "keyword_score": 0.0,
                "semantic_result": result,
            }

        for result in keyword_results:
            doc_id = result.document_id
            if doc_id in combined_docs:
                combined_docs[doc_id]["keyword_score"] = result.score
            else:
                combined_docs[doc_id] = {
                    "document_id": doc_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "semantic_score": 0.0,
                    "keyword_score": result.score,
                    "keyword_result": result,
                }

        # Calculate combined scores
        fused_results = []
        for doc_data in combined_docs.values():
            combined_score = semantic_weight * doc_data["semantic_score"] + keyword_weight * doc_data["keyword_score"]

            search_result = SearchResult(
                document_id=doc_data["document_id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                score=combined_score,
                search_type="hybrid",
            )
            fused_results.append(search_result)

        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)

        return results


class MMRSearcher:
    """
    Maximal Marginal Relevance (MMR) search for diverse results.
    """

    def __init__(self, embedding_manager: "EmbeddingManager", config: Optional["OrionConfig"] = None):
        self.embedding_manager = embedding_manager
        self.config = ensure_config(config)

    @timer
    def search(
        self,
        query: str,
        candidate_results: list[SearchResult],
        k: int = 5,
        diversity_bias: float | None = None,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Apply MMR selection to candidate results for diverse retrieval.

        Args:
            query: Original search query
            candidate_results: Initial candidate results to diversify
            k: Number of diverse results to return
            diversity_bias: Balance between relevance and diversity (0.0-1.0)
            similarity_threshold: Minimum similarity threshold

        Returns:
            Diversified list of SearchResult objects
        """
        if not candidate_results or not query.strip():
            log_warning("No candidates or empty query for MMR", config=self.config)
            return []

        if diversity_bias is None:
            diversity_bias = self.config.rag.retrieval.mmr_diversity_bias
        if similarity_threshold is None:
            similarity_threshold = self.config.rag.retrieval.mmr_threshold

        try:
            query_embedding = self.embedding_manager.encode_single(query)
            if not query_embedding:
                log_warning("Failed to generate query embedding for MMR", config=self.config)
                return candidate_results[:k]

            # Generate embeddings for candidate documents
            candidate_texts = [result.content for result in candidate_results]
            candidate_embeddings = self.embedding_manager.encode_batch(candidate_texts)

            # Filter out candidates without embeddings
            valid_candidates = []
            valid_embeddings = []
            for result, embedding in zip(candidate_results, candidate_embeddings):
                if embedding:
                    valid_candidates.append(result)
                    valid_embeddings.append(embedding)

            if not valid_candidates:
                log_warning("No valid embeddings for MMR", config=self.config)
                return candidate_results[:k]

            # MMR algorithm
            selected_results = self._mmr_selection(
                query_embedding=query_embedding,
                candidates=valid_candidates,
                candidate_embeddings=valid_embeddings,
                k=k,
                lambda_param=diversity_bias,
                similarity_threshold=similarity_threshold,
            )

            log_debug(
                f"MMR selected {len(selected_results)} diverse results from {len(valid_candidates)} candidates", self.config
            )
            return selected_results

        except Exception as e:
            log_warning(f"MMR search failed: {e}", config=self.config)
            return candidate_results[:k]

    def _mmr_selection(
        self,
        query_embedding: list[float],
        candidates: list[SearchResult],
        candidate_embeddings: list[list[float]],
        k: int,
        lambda_param: float,
        similarity_threshold: float,
    ) -> list[SearchResult]:
        """
        Core MMR selection algorithm.

        Args:
            query_embedding: Query embedding vector
            candidates: Candidate search results
            candidate_embeddings: Corresponding embeddings
            k: Number of results to select
            lambda_param: Balance parameter (1.0 = pure relevance, 0.0 = pure diversity)
            similarity_threshold: Minimum similarity threshold

        Returns:
            Selected diverse results
        """
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array(candidate_embeddings)

        # Calculate relevance scores (cosine similarity with query)
        relevance_scores = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
        )

        # Filter by similarity threshold
        valid_indices = np.where(relevance_scores >= similarity_threshold)[0]
        if len(valid_indices) == 0:
            log_warning(f"No candidates above similarity threshold {similarity_threshold}", config=self.config)
            return []

        selected_indices = []
        remaining_indices = valid_indices.tolist()
        selected_embeddings = []

        for _ in range(min(k, len(remaining_indices))):
            if not remaining_indices:
                break

            if not selected_indices:
                # First selection: highest relevance
                best_idx = remaining_indices[np.argmax(relevance_scores[remaining_indices])]
            else:
                # Subsequent selections: MMR formula
                mmr_scores = []
                for idx in remaining_indices:
                    relevance = relevance_scores[idx]

                    # Diversity component (max similarity to already selected)
                    if selected_embeddings:
                        similarities = []
                        for selected_emb in selected_embeddings:
                            sim = np.dot(candidate_vecs[idx], selected_emb) / (
                                np.linalg.norm(candidate_vecs[idx]) * np.linalg.norm(selected_emb)
                            )
                            similarities.append(sim)
                        max_similarity = max(similarities)
                    else:
                        max_similarity = 0.0

                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    mmr_scores.append(mmr_score)

                best_relative_idx = np.argmax(mmr_scores)
                best_idx = remaining_indices[best_relative_idx]

            selected_indices.append(best_idx)
            selected_embeddings.append(candidate_vecs[best_idx])
            remaining_indices.remove(best_idx)

        # Return selected results with updated scores
        selected_results = []
        for idx in selected_indices:
            result = candidates[idx]
            # Update score
            result.score = relevance_scores[idx]
            result.search_type = "mmr"
            selected_results.append(result)

        return selected_results


def create_semantic_searcher(
    embedding_manager: "EmbeddingManager", vector_store: "ChromaVectorStore", config: Optional["OrionConfig"] = None
) -> SemanticSearcher:
    """Factory function for SemanticSearcher."""
    return SemanticSearcher(embedding_manager, vector_store, config)


def create_keyword_searcher(config: Optional["OrionConfig"] = None) -> KeywordSearcher:
    """Factory function for KeywordSearcher."""
    return KeywordSearcher(config)


def create_hybrid_searcher(
    semantic_searcher: SemanticSearcher, keyword_searcher: KeywordSearcher, config: Optional["OrionConfig"] = None
) -> HybridSearcher:
    """Factory function for HybridSearcher."""
    return HybridSearcher(semantic_searcher, keyword_searcher, config)


def create_mmr_searcher(embedding_manager: "EmbeddingManager", config: Optional["OrionConfig"] = None) -> MMRSearcher:
    """Factory function for MMRSearcher."""
    return MMRSearcher(embedding_manager, config)
