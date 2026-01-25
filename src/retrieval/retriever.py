"""
Orion Document Retriever

High-level interface for the complete Orion retrieval pipeline:
semantic/hybrid search → reranking → MMR diversity → formatted results

Usage:
    from src.retrieval.retriever import OrionRetriever

    retriever = OrionRetriever()
    results = retriever.query("What is machine learning?")
    print(results)
"""

import logging
from typing import Optional

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.reranker import Document, RerankerManager
from src.retrieval.search import HybridSearcher, KeywordSearcher, MMRSearcher, SearchResult, SemanticSearcher
from src.retrieval.vector_store import ChromaVectorStore
from src.utilities.config import OrionConfig, TimingBreakdown
from src.utilities.utils import ensure_config, log_error, log_info, log_warning

# Set up logging
logger = logging.getLogger(__name__)


class OrionRetriever:
    """
    High-level document retriever for Orion.

    Provides a simple interface to query the knowledge base using the full
    Orion retrieval pipeline: search → rerank → MMR → format results.
    """

    def __init__(self, config: Optional[OrionConfig] = None):
        """
        Initialize the Orion retriever.

        Args:
            config: Optional Orion configuration. If None, loads default config.
        """
        self.config = ensure_config(config)
        self._vector_store = None
        self._embedding_manager = None
        self._reranker = None
        self._mmr_searcher = None
        self._initialized = False

    def _initialize_components(self):
        """Lazy initialization of retrieval components."""
        if self._initialized:
            return

        try:
            log_info("Initializing Orion retriever components...", config=self.config)

            # Initialize core components
            self._vector_store = ChromaVectorStore(self.config)
            self._embedding_manager = EmbeddingManager(self.config)

            # Initialize reranker
            self._reranker = RerankerManager(self.config)

            # Initialize MMR searcher
            self._mmr_searcher = MMRSearcher(self._embedding_manager, self.config)

            self._initialized = True
            log_info("Orion retriever components initialized successfully", config=self.config)

        except Exception as e:
            log_error(f"Failed to initialize retriever components: {e}", config=self.config)
            raise

    def _check_knowledge_base(self) -> int:
        """
        Check if the knowledge base has documents.

        Returns:
            Number of documents in the knowledge base.

        Raises:
            ValueError: If no documents are found.
        """
        stats = self._vector_store.get_collection_stats()
        doc_count = stats.get("document_count", 0)

        if doc_count == 0:
            raise ValueError(
                "No documents found in knowledge base. "
                "Please run ingestion first."
            )

        log_info(f"Found {doc_count} documents in knowledge base", config=self.config)
        return doc_count

    def _perform_search(self, query: str, k: int, search_type: str) -> list[SearchResult]:
        """
        Perform the initial search (semantic or hybrid).

        Args:
            query: Search query
            k: Number of results to retrieve
            search_type: Type of search ('semantic' or 'hybrid')

        Returns:
            List of search results
        """
        if search_type == "semantic":
            searcher = SemanticSearcher(self._embedding_manager, self._vector_store, self.config)
            return searcher.search(query, k=k)

        elif search_type == "hybrid":
            # Create semantic and keyword searchers
            semantic_searcher = SemanticSearcher(self._embedding_manager, self._vector_store, self.config)
            
            # Keyword searcher with auto-sync
            keyword_searcher = KeywordSearcher(self._vector_store, self.config)
            
            # Create hybrid searcher with RRF fusion (default)
            hybrid_searcher = HybridSearcher(semantic_searcher, keyword_searcher, self.config)
            
            # Use RRF fusion by default (more robust than weighted)
            return hybrid_searcher.search(query, k=k, fusion_method="rrf")

        else:
            raise ValueError(f"Unsupported search type: {search_type}. Use 'semantic' or 'hybrid'.")

    def _apply_reranking(
        self, query: str, results: list[SearchResult], k: int
    ) -> list[SearchResult]:
        """
        Apply reranking to search results.

        Args:
            query: Original search query
            results: Initial search results
            k: Number of top results to return

        Returns:
            Reranked search results
        """
        if not results:
            return results

        # Convert SearchResult objects to Document objects for reranking
        docs_to_rerank = []
        for result in results:
            doc = Document(
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                doc_id=result.document_id
            )
            docs_to_rerank.append(doc)

        # Rerank documents
        reranked_docs = self._reranker.rerank_documents(query, docs_to_rerank, top_k=k)

        # Convert back to SearchResult objects
        reranked_results = []
        for doc in reranked_docs:
            result = SearchResult(
                document_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
                search_type=f"{results[0].search_type}_reranked" if results else "reranked",
            )
            reranked_results.append(result)

        return reranked_results

    def _apply_mmr(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        """
        Apply MMR diversity search to results.

        Args:
            query: Original search query
            results: Search results to diversify
            k: Number of diverse results to return

        Returns:
            Diversified search results
        """
        if not results or len(results) <= 1:
            return results

        # Apply MMR to get diverse subset
        mmr_results = self._mmr_searcher.search(query, candidate_results=results, k=k)

        # Update search_type to indicate MMR was applied
        for result in mmr_results:
            if result.search_type and not result.search_type.endswith("_mmr"):
                result.search_type += "_mmr"

        return mmr_results

    def _format_results(self, results: list[SearchResult]) -> str:
        """
        Format search results into a readable string.

        Args:
            results: Search results to format

        Returns:
            Formatted results string
        """
        if not results:
            return "No results found for your query."

        formatted_parts = [f"Found {len(results)} relevant results:\n"]

        for i, result in enumerate(results, 1):
            # Basic result info
            source = result.metadata.get("source_file", "Unknown")
            file_type = result.metadata.get("file_type", "Unknown")
            score = result.score

            # Chunk information
            chunk_info = ""
            if "chunk_index" in result.metadata:
                chunk_idx = result.metadata["chunk_index"] + 1
                chunk_total = result.metadata.get("chunk_total", "Unknown")
                chunk_info = f" (Chunk {chunk_idx}/{chunk_total})"

            # Format header
            formatted_parts.append(f"\n--- Result {i} (Score: {score:.4f}) ---")
            formatted_parts.append(f"Source: {source}{chunk_info}")
            formatted_parts.append(f"Type: {file_type}")

            # Content preview (limit to 300 characters for readability)
            content = result.content.strip()
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_parts.append(f"Content: {content}")
            formatted_parts.append("")

        return "\n".join(formatted_parts)

    def query(
        self,
        query_text: str,
        k: int = 5,
        search_type: str = "hybrid",
        enable_reranking: bool = True,
        enable_mmr: bool = True,
        formatted: bool = True,
        return_timing: bool = False,
    ) -> str | tuple[str | list[SearchResult], TimingBreakdown]:
        """
        Query the knowledge base and return formatted results.

        Args:
            query_text: The search query string
            k: Number of results to return (default: 5)
            search_type: Type of search - 'semantic' or 'hybrid' (default: 'hybrid')
            enable_reranking: Whether to apply reranking (default: True)
            enable_mmr: Whether to apply MMR diversity (default: True)
            formatted: Whether to return formatted string or raw SearchResult objects
            return_timing: Whether to return timing breakdown

        Returns:
            Formatted string containing search results, or tuple of (results, timing)

        Raises:
            ValueError: If knowledge base is empty or search type is invalid
            RuntimeError: If retrieval components fail to initialize
        """
        import time
        
        # Initialize timing
        timing = TimingBreakdown()
        overall_start = time.time()
        
        try:
            # Initialize components if needed
            self._initialize_components()

            # Check knowledge base
            doc_count = self._check_knowledge_base()

            log_info(f"Knowledge base contains {doc_count} documents", config=self.config)
            log_info(f"Querying: '{query_text}' (type: {search_type}, k: {k})", config=self.config)

            # Perform initial search (includes embedding and search time)
            search_start = time.time()
            results = self._perform_search(query_text, k=k, search_type=search_type)
            search_elapsed = time.time() - search_start
            
            # Estimate embedding took ~30% of search time, search ~70%
            timing.embedding_time = search_elapsed * 0.3
            timing.search_time = search_elapsed * 0.7

            if not results:
                log_warning(f"No initial results found for query: {query_text}", config=self.config)
                result = "No results found for your query." if formatted else []
                if return_timing:
                    timing.total_time = time.time() - overall_start
                    return result, timing
                return result

            log_info(f"Initial search returned {len(results)} results", config=self.config)

            # Apply reranking if enabled
            if enable_reranking and results:
                log_info("Applying reranking...", config=self.config)
                rerank_start = time.time()
                results = self._apply_reranking(query_text, results, k=k)
                timing.reranking_time = time.time() - rerank_start
                log_info(f"Reranking returned {len(results)} results", config=self.config)

            # Apply MMR diversity if enabled
            if enable_mmr and results and len(results) > 1:
                log_info("Applying MMR diversity...", config=self.config)
                mmr_start = time.time()
                results = self._apply_mmr(query_text, results, k=k)
                timing.mmr_time = time.time() - mmr_start
                log_info(f"MMR returned {len(results)} diverse results", config=self.config)

            # Format results if requested
            if formatted:
                result = self._format_results(results)
            else:
                result = results
            
            log_info(f"Query completed successfully, returning {len(results)} results", config=self.config)
            
            # Calculate total time
            timing.total_time = time.time() - overall_start
            
            if return_timing:
                return result, timing
            return result

        except ValueError as e:
            # User-facing errors (empty knowledge base, invalid search type)
            log_warning(f"Query failed: {e}", config=self.config)
            result = f"Error: {e}"
            if return_timing:
                timing.total_time = time.time() - overall_start
                return result, timing
            return result

        except Exception as e:
            # System errors
            log_error(f"Unexpected error during query: {e}", config=self.config)
            result = f"An error occurred while processing your query: {e}"
            if return_timing:
                timing.total_time = time.time() - overall_start
                return result, timing
            return result

    def get_status(self) -> dict:
        """
        Get the current status of the retriever and knowledge base.

        Returns:
            Dictionary containing status information
        """
        try:
            self._initialize_components()
            stats = self._vector_store.get_collection_stats()

            return {
                "initialized": self._initialized,
                "document_count": stats.get("document_count", 0),
                "collection_stats": stats,
                "config": {
                    "embedding_model": self.config.rag.embedding.model,
                    "llm_model": self.config.rag.llm.model,
                    "reranking_enabled": self.config.rag.retrieval.enable_reranking,
                    "chunk_size": self.config.rag.chunking.chunk_size,
                },
            }
        except Exception as e:
            return {"initialized": False, "error": str(e), "document_count": 0}


# Convenience function for simple usage
def query_knowledge_base(query: str, **kwargs) -> str:
    """
    Convenience function to query the knowledge base with default settings.

    Args:
        query: Search query string
        **kwargs: Additional arguments passed to OrionRetriever.query()

    Returns:
        Formatted search results string
    """
    retriever = OrionRetriever()
    return retriever.query(query, **kwargs)
