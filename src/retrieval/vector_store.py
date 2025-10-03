"""
Chroma vector database operations for Orion.
Handles document storage, retrieval, and index management.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import chromadb
from chromadb.config import Settings

from src.utilities.utils import (
    ensure_config,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)

if TYPE_CHECKING:
    from src.utilities.config import OrionConfig


class ChromaVectorStore:
    """
    Manages Chroma vector database operations for Orion.

    Handles document storage, retrieval, and persistence with
    Filipino cultural content optimization.
    """

    def __init__(self, config: Optional["OrionConfig"] = None):
        """
        Initialize Chroma vector store.

        Args:
            config: Orion configuration object
        """
        self.config = ensure_config(config)
        self.vectorstore_config = self.config.rag.vectorstore
        self.client = None
        self.collection = None
        self._collection_name = self.vectorstore_config.collection_name

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Chroma client and ensure persistence directory exists."""
        try:
            # Create persistence directory if it doesn't exist
            persist_dir = Path(self.vectorstore_config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            log_debug(f"Initializing Chroma client at: {persist_dir}", self.config)

            # Configure Chroma settings
            settings = Settings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False,
            )

            # Create persistent client
            self.client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)

            log_success("Chroma client initialized successfully", config=self.config)

        except Exception as e:
            log_error(f"Failed to initialize Chroma client: {e}", config=self.config)
            raise

    def _get_or_create_collection(self) -> None:
        """Get existing collection or create a new one."""
        if self.client is None:
            raise RuntimeError("Chroma client not initialized")

        try:
            self.collection = self.client.get_collection(name=self._collection_name)
            log_info(f"Retrieved existing collection: {self._collection_name}", config=self.config)

        except Exception:
            try:
                self.collection = self.client.create_collection(
                    name=self._collection_name,
                    metadata={
                        "description": "Orion Filipino cultural knowledge base",
                        "created_by": "Orion",
                        "distance_metric": self.vectorstore_config.distance_metric,
                    },
                )
                log_success(f"Created new collection: {self._collection_name}", config=self.config)

            except Exception as e:
                log_error(f"Failed to create collection: {e}", config=self.config)
                raise

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs

        Returns:
            True if successful, False otherwise
        """
        if not documents or not embeddings or not metadatas:
            log_warning("Empty documents, embeddings, or metadatas provided", config=self.config)
            return False

        if len(documents) != len(embeddings) or len(documents) != len(metadatas):
            log_error("Mismatch in lengths of documents, embeddings, and metadatas", config=self.config)
            return False

        if self.collection is None:
            self._get_or_create_collection()

        try:
            if ids is None:
                ids = [f"doc_{i}_{hash(doc[:50])}" for i, doc in enumerate(documents)]

            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            log_success(f"Added {len(documents)} documents to vector store", config=self.config)

            # Persist if configured
            if self.vectorstore_config.persist_immediately:
                self.persist()

            return True

        except Exception as e:
            log_error(f"Failed to add documents: {e}", config=self.config)
            return False

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filter
            include: Fields to include in results

        Returns:
            Query results dictionary
        """
        if self.collection is None:
            self._get_or_create_collection()

        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            n_results = min(n_results, self.config.rag.retrieval.max_k)

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include,
            )

            log_debug(f"Query returned {len(results.get('ids', [[]])[0])} results", self.config)
            return results

        except Exception as e:
            log_error(f"Failed to query vector store: {e}", config=self.config)
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    def get_document_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Document data or None if not found
        """
        if self.collection is None:
            self._get_or_create_collection()

        try:
            results = self.collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])

            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0] if results["documents"] else None,
                    "metadata": results["metadatas"][0] if results["metadatas"] else None,
                    "embedding": results["embeddings"][0] if results["embeddings"] else None,
                }

            log_debug(f"Document not found: {doc_id}", self.config)
            return None

        except Exception as e:
            log_error(f"Failed to get document {doc_id}: {e}", config=self.config)
            return None

    def delete_documents(self, ids: list[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            log_warning("No collection available for deletion", config=self.config)
            return False

        try:
            self.collection.delete(ids=ids)
            log_info(f"Deleted {len(ids)} documents from vector store", config=self.config)

            if self.vectorstore_config.persist_immediately:
                self.persist()

            return True

        except Exception as e:
            log_error(f"Failed to delete documents: {e}", config=self.config)
            return False

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            self._get_or_create_collection()

        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self._collection_name,
                "persist_directory": self.vectorstore_config.persist_directory,
                "distance_metric": self.vectorstore_config.distance_metric,
            }

        except Exception as e:
            log_error(f"Failed to get collection stats: {e}", config=self.config)
            return {"document_count": 0, "error": str(e)}

    def list_all_sources(self) -> list[dict[str, Any]]:
        """
        List all unique sources in the vector store with statistics.

        Returns:
            List of dictionaries containing source information:
            - source_file: Original file path
            - file_name: Just the filename
            - file_type: Type of file (PDF, Text, etc.)
            - chunk_count: Number of chunks from this source
            - first_chunk_id: ID of the first chunk (for reference)
        """
        if self.collection is None:
            self._get_or_create_collection()

        try:
            # Get all documents with metadata
            all_data = self.collection.get(include=["metadatas"])

            if not all_data["metadatas"] or not all_data.get("ids"):
                log_info("No documents found in collection", config=self.config)
                return []

            # Group by source file
            sources = {}
            for i, metadata in enumerate(all_data["metadatas"]):
                if not metadata:
                    continue

                source_file = metadata.get("source_file", "unknown")
                file_name = metadata.get("file_name", "unknown")
                file_type = metadata.get("file_type", "unknown")
                doc_id = all_data["ids"][i] if i < len(all_data["ids"]) else f"doc_{i}"

                if source_file not in sources:
                    sources[source_file] = {
                        "source_file": source_file,
                        "file_name": file_name,
                        "file_type": file_type,
                        "chunk_count": 0,
                        "first_chunk_id": doc_id,
                    }

                sources[source_file]["chunk_count"] += 1

            # Convert to sorted list
            source_list = list(sources.values())
            source_list.sort(key=lambda x: x["file_name"].lower())

            log_info(f"Found {len(source_list)} unique sources", config=self.config)
            return source_list

        except Exception as e:
            log_error(f"Failed to list sources: {e}", config=self.config)
            return []

    def get_chunks_by_source(self, source_file: str) -> list[dict[str, Any]]:
        """
        Get all chunks from a specific source file.

        Args:
            source_file: Path to the source file

        Returns:
            List of chunks with their content and metadata
        """
        if self.collection is None:
            self._get_or_create_collection()

        try:
            # Query for documents from specific source
            results = self.collection.get(where={"source_file": source_file}, include=["documents", "metadatas"])

            chunks = []
            if results.get("ids"):
                for i in range(len(results["ids"])):
                    chunk = {
                        "id": results["ids"][i],
                        "content": results["documents"][i] if i < len(results["documents"]) else "",
                        "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {},
                    }
                    chunks.append(chunk)

            # Sort by chunk_index if available
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            log_info(f"Found {len(chunks)} chunks for source: {source_file}", config=self.config)
            return chunks

        except Exception as e:
            log_error(f"Failed to get chunks for source {source_file}: {e}", config=self.config)
            return []

    def persist(self) -> bool:
        """
        Manually persist the collection to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.client:
                # Chroma automatically persists with PersistentClient
                log_debug("Vector store persisted successfully", self.config)
                return True
            else:
                log_warning("No client available for persistence", config=self.config)
                return False

        except Exception as e:
            log_error(f"Failed to persist vector store: {e}", config=self.config)
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            log_info("No collection to clear", config=self.config)
            return True

        try:
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
                log_info(f"Cleared {len(all_data['ids'])} documents from collection", config=self.config)
            else:
                log_info("Collection already empty", config=self.config)

            return True

        except Exception as e:
            log_error(f"Failed to clear collection: {e}", config=self.config)
            return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "vectorstore_config") and self.vectorstore_config.persist_immediately:
            try:
                self.persist()
            except Exception:
                pass  # Ignore errors during cleanup


def create_vector_store(config: Optional["OrionConfig"] = None) -> ChromaVectorStore:
    """
    Factory function to create a ChromaVectorStore instance.

    Args:
        config: Orion configuration object

    Returns:
        Initialized ChromaVectorStore instance
    """
    return ChromaVectorStore(config)
