"""
Async Document Processing Module

This module provides asynchronous document processing capabilities for better performance.
Key concepts:
- asyncio: Python's built-in async/await framework
- Concurrent processing: Multiple documents processed simultaneously
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document

from app.utils import log_info, log_success, log_warning, log_error


class AsyncDocumentProcessor:
    """
    Async document processor for parallel document loading and processing.

    Key Features:
    - Parallel document loading
    - Async file I/O operations
    - Progress tracking
    - Error handling with partial success
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize async processor.

        Args:
            max_workers: Maximum number of concurrent document processing tasks
                        (Default: 4 - good balance for most systems)
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def load_document_async(self, file_path: Path) -> Optional[List[Document]]:
        """
        Load a single document asynchronously.

        This function:
        1. Checks if file is supported
        2. Loads document using appropriate loader
        3. Adds metadata
        4. Returns loaded documents

        Args:
            file_path: Path to document to load

        Returns:
            List of Document objects or None if failed
        """
        try:
            # Import here to avoid circular imports
            from app.ingest import (
                get_loader_for_file,
                extract_metadata,
                SUPPORTED_EXTENSIONS,
            )

            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                return None

            # Extract metadata first (this is I/O intensive)
            metadata = await asyncio.get_event_loop().run_in_executor(
                self.executor, extract_metadata, file_path, suffix
            )

            # Get appropriate loader
            loader = get_loader_for_file(file_path)

            if loader == "metadata_only":
                return [Document(page_content="", metadata=metadata)]
            elif loader in ["email_loader", "archive_loader"]:
                # Skip unimplemented loaders
                return None
            elif loader:
                # Load document content in thread executor (I/O bound)
                docs = await asyncio.get_event_loop().run_in_executor(
                    self.executor, loader.load
                )

                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update(metadata)

                return docs
            else:
                return None

        except Exception as e:
            log_error(f"Failed to load {file_path.name}: {e}")
            return None

    async def load_documents_async(self, folder_path: str) -> List[Document]:
        """
        Load all documents from folder asynchronously.

        This is the main optimization - instead of loading documents one by one,
        we load multiple documents in parallel using asyncio.gather().

        Performance Improvement:
        - Old way: Document 1 (2s) → Document 2 (2s) → Document 3 (2s) = 6 seconds
        - New way: Document 1, 2, 3 simultaneously (2s max) = 2 seconds

        Args:
            folder_path: Path to folder containing documents

        Returns:
            List of all loaded documents
        """
        folder = Path(folder_path)
        if not folder.exists():
            log_error(f"Folder not found: {folder_path}")
            return []

        # Import here to avoid circular imports
        from app.ingest import SUPPORTED_EXTENSIONS

        # Find all supported files
        supported_files = [
            f
            for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not supported_files:
            log_warning(f"No supported files found in '{folder_path}'")
            return []

        log_info(f"Loading {len(supported_files)} documents asynchronously...")

        # Record start time for performance measurement
        start_time = time.time()

        # Create async tasks for all documents
        # This is the key optimization - all documents load simultaneously!
        tasks = [self.load_document_async(file_path) for file_path in supported_files]

        # Execute all tasks concurrently and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and collect successful document loads
        all_docs = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log_error(f"Exception loading {supported_files[i].name}: {result}")
                failed_count += 1
            elif result is not None:
                all_docs.extend(result)
            else:
                failed_count += 1

        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time

        log_success(
            f"Async loading completed: {len(all_docs)} documents loaded "
            f"in {total_time:.2f}s (failed: {failed_count})"
        )

        return all_docs

    async def process_documents_in_batches(
        self, documents: List[Document], batch_size: int = 10
    ) -> List[Document]:
        """
        Process documents in batches to avoid overwhelming the system.

        This prevents memory issues when processing thousands of documents.

        Args:
            documents: List of documents to process
            batch_size: Number of documents to process simultaneously

        Returns:
            List of processed documents
        """
        processed_docs = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            log_info(
                f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}"
            )

            # Process batch asynchronously
            batch_tasks = [self.process_single_document(doc) for doc in batch]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect successful results
            for result in batch_results:
                if not isinstance(result, Exception) and result is not None:
                    processed_docs.append(result)

        return processed_docs

    async def process_single_document(self, document: Document) -> Optional[Document]:
        """
        Process a single document (placeholder for future enhancements).

        Could include:
        - Text cleaning
        - Language detection
        - Content analysis
        - Format normalization

        Args:
            document: Document to process

        Returns:
            Processed document
        """
        # For now, just return the document as-is
        # In future, could add async processing steps here
        return document

    def close(self):
        """Clean up thread pool executor."""
        self.executor.shutdown(wait=True)


# Convenience function for easy async document loading
async def load_documents_fast(folder_path: str, max_workers: int = 4) -> List[Document]:
    """
    Fast async document loading - convenience function.

    Usage:
        import asyncio
        docs = asyncio.run(load_documents_fast("./documents"))

    Args:
        folder_path: Path to documents folder
        max_workers: Number of concurrent workers

    Returns:
        List of loaded documents
    """
    processor = AsyncDocumentProcessor(max_workers=max_workers)
    try:
        return await processor.load_documents_async(folder_path)
    finally:
        processor.close()


def compare_performance(folder_path: str):
    """
    Compare sync vs async performance - useful for testing.

    This function demonstrates the performance difference between
    synchronous and asynchronous document loading.
    """
    from app.ingest import load_documents  # Original sync function

    print("🔄 Performance Comparison: Sync vs Async Document Loading")
    print("=" * 60)

    # Test synchronous loading
    print("Testing synchronous loading...")
    sync_start = time.time()
    sync_docs = load_documents(folder_path)
    sync_time = time.time() - sync_start

    # Test asynchronous loading
    print("Testing asynchronous loading...")
    async_start = time.time()
    async_docs = asyncio.run(load_documents_fast(folder_path))
    async_time = time.time() - async_start

    # Results
    print("\n📊 Results:")
    print(f"Sync:  {len(sync_docs)} docs in {sync_time:.2f}s")
    print(f"Async: {len(async_docs)} docs in {async_time:.2f}s")

    if async_time < sync_time:
        speedup = sync_time / async_time
        print(f"🚀 Async is {speedup:.1f}x faster!")
    else:
        print("ℹ️  No significant speedup (may need more documents to see benefit)")
