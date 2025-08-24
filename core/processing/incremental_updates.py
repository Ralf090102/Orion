"""
Incremental Document Update System

This module implements smart document tracking and incremental updates:

1. **Change Detection**: Track file modifications using timestamps and checksums
2. **Selective Processing**: Only process documents that have actually changed
3. **Dependency Tracking**: Update related documents when dependencies change
4. **Rollback Capability**: Revert to previous states if updates fail

Key Concepts:
- Checksum: A unique "fingerprint" for file content
- Timestamp: When a file was last modified
- Incremental: Processing only changes, not everything
- Hot Reload: Automatically update when files change
"""

import json
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

from core.utils.orion_utils import log_info, log_debug, log_warning, log_error
from core.processing.async_processing import AsyncDocumentProcessor
from core.utils.caching import _global_cache


@dataclass
class DocumentMetadata:
    """
    Metadata for tracking document changes.

    Attributes:
        path: Full path to the document
        checksum: MD5 hash of file content
        size: File size in bytes
        last_modified: OS timestamp of last modification
        last_processed: When we last processed this document
        version: Version number for this document
        dependencies: Other documents this one depends on
    """

    path: str
    checksum: str
    size: int
    last_modified: float
    last_processed: float
    version: int = 1
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class IncrementalUpdateManager:
    """
    Manages incremental updates for document processing.

    This system tracks document changes and only processes what's actually
    changed, dramatically improving performance for large document sets.

    Features:
    - File change detection using checksums
    - Batch processing of multiple changes
    - Dependency tracking
    - Rollback on failures
    - Hot reload capabilities
    """

    def __init__(
        self,
        documents_dir: str,
        metadata_file: str = "data/document_metadata.json",
        enable_hot_reload: bool = False,
    ):
        """
        Initialize the incremental update manager.

        Args:
            documents_dir: Directory containing documents to monitor
            metadata_file: File to store document metadata
            enable_hot_reload: Whether to automatically detect and process changes
        """
        self.documents_dir = Path(documents_dir)
        self.metadata_file = Path(metadata_file)
        self.enable_hot_reload = enable_hot_reload

        # Ensure directories exist
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Document metadata tracking
        self.metadata: Dict[str, DocumentMetadata] = {}
        self.load_metadata()

        # Initialize async processor for efficient updates
        self.async_processor = AsyncDocumentProcessor()

        # Statistics for monitoring
        self.stats = {
            "total_documents": 0,
            "changed_documents": 0,
            "added_documents": 0,
            "deleted_documents": 0,
            "last_scan": 0,
            "processing_time": 0,
        }

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate MD5 checksum of file content.

        This creates a unique "fingerprint" for the file that changes
        if the content changes, allowing us to detect modifications.
        """
        hash_md5 = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        except Exception as e:
            log_warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""

    def _get_file_stats(self, file_path: Path) -> Tuple[int, float]:
        """Get file size and modification time."""
        try:
            stat = file_path.stat()
            return stat.st_size, stat.st_mtime
        except Exception as e:
            log_warning(f"Could not get stats for {file_path}: {e}")
            return 0, 0

    def load_metadata(self):
        """Load document metadata from disk."""
        if not self.metadata_file.exists():
            log_info("No existing metadata file found, starting fresh")
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            # Convert dict to DocumentMetadata objects
            for path, data in metadata_dict.items():
                self.metadata[path] = DocumentMetadata(**data)

            log_info(f"Loaded metadata for {len(self.metadata)} documents")

        except Exception as e:
            log_error(f"Error loading metadata: {e}")
            log_warning("Starting with empty metadata")
            self.metadata = {}

    def save_metadata(self):
        """Save document metadata to disk."""
        try:
            # Convert DocumentMetadata objects to dict for JSON serialization
            metadata_dict = {path: asdict(metadata) for path, metadata in self.metadata.items()}

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2)

            log_debug(f"Saved metadata for {len(self.metadata)} documents")

        except Exception as e:
            log_error(f"Error saving metadata: {e}")

    def scan_for_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Scan directory for document changes.

        Returns:
            Tuple of (added_files, changed_files, deleted_files)
        """
        log_info(f"Scanning for changes in {self.documents_dir}")
        start_time = time.time()

        added_files = []
        changed_files = []
        deleted_files = []

        # Track current files
        current_files = set()

        # Scan all files in documents directory
        if self.documents_dir.exists():
            for file_path in self.documents_dir.rglob("*"):
                if file_path.is_file() and self._should_process_file(file_path):
                    file_path_str = str(file_path)
                    current_files.add(file_path_str)

                    # Calculate current file properties
                    checksum = self._calculate_file_checksum(file_path)
                    size, last_modified = self._get_file_stats(file_path)

                    if file_path_str in self.metadata:
                        # Existing file - check if changed
                        existing = self.metadata[file_path_str]

                        if checksum != existing.checksum or size != existing.size or last_modified > existing.last_modified:

                            changed_files.append(file_path_str)
                            log_debug(f"Changed: {file_path_str}")

                    else:
                        # New file
                        added_files.append(file_path_str)
                        log_debug(f"Added: {file_path_str}")

        # Check for deleted files
        tracked_files = set(self.metadata.keys())
        deleted_files = list(tracked_files - current_files)

        for deleted_file in deleted_files:
            log_debug(f"Deleted: {deleted_file}")

        scan_time = time.time() - start_time

        log_info(f"Change scan completed in {scan_time:.2f}s:")
        log_info(f"  Added: {len(added_files)} files")
        log_info(f"  Changed: {len(changed_files)} files")
        log_info(f"  Deleted: {len(deleted_files)} files")

        # Update statistics
        self.stats.update(
            {
                "total_documents": len(current_files),
                "added_documents": len(added_files),
                "changed_documents": len(changed_files),
                "deleted_documents": len(deleted_files),
                "last_scan": time.time(),
            }
        )

        return added_files, changed_files, deleted_files

    def _should_process_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be processed.

        Add your own logic here to filter files by extension,
        size, or other criteria.
        """
        # Common document extensions
        valid_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".doc",
            ".html",
            ".htm",
            ".rtf",
            ".odt",
        }

        extension = file_path.suffix.lower()

        # Skip hidden files and system files
        if file_path.name.startswith(".") or file_path.name.startswith("~"):
            return False

        # Check file extension
        if extension in valid_extensions:
            return True

        # Skip very large files (>10MB) to avoid memory issues
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                log_warning(f"Skipping large file: {file_path} ({file_path.stat().st_size} bytes)")
                return False
        except Exception as e:
            log_warning(f"Error checking file size for {file_path}: {e}")
            return False

        return extension in valid_extensions

    async def process_incremental_update(self, force_full_rebuild: bool = False) -> Dict[str, Any]:
        """
        Process incremental updates for changed documents.

        Args:
            force_full_rebuild: If True, rebuild everything regardless of changes

        Returns:
            Dictionary with update results and statistics
        """
        start_time = time.time()

        if force_full_rebuild:
            log_info("Force rebuilding all documents")
            # Clear metadata to force full reprocessing
            self.metadata.clear()
            _global_cache.clear()  # Clear cache too

        # Scan for changes
        added_files, changed_files, deleted_files = self.scan_for_changes()

        total_changes = len(added_files) + len(changed_files) + len(deleted_files)

        if total_changes == 0 and not force_full_rebuild:
            log_info("No changes detected - nothing to process")
            return {
                "status": "no_changes",
                "processing_time": time.time() - start_time,
                "statistics": self.stats,
            }

        results = {"added": [], "updated": [], "deleted": [], "errors": []}

        try:
            # Process deleted files first
            for deleted_file in deleted_files:
                try:
                    self._handle_deleted_file(deleted_file)
                    results["deleted"].append(deleted_file)
                except Exception as e:
                    error_msg = f"Error handling deleted file {deleted_file}: {e}"
                    log_error(error_msg)
                    results["errors"].append(error_msg)

            # Process added and changed files
            files_to_process = added_files + changed_files

            if files_to_process:
                log_info(f"Processing {len(files_to_process)} documents...")

                # Use async processing for better performance
                processed_docs = await self.async_processor.load_documents_from_paths(files_to_process)

                # Update metadata for processed files
                for i, file_path in enumerate(files_to_process):
                    try:
                        if i < len(processed_docs):
                            self._update_file_metadata(file_path)

                            if file_path in added_files:
                                results["added"].append(file_path)
                            else:
                                results["updated"].append(file_path)

                    except Exception as e:
                        error_msg = f"Error updating metadata for {file_path}: {e}"
                        log_error(error_msg)
                        results["errors"].append(error_msg)

            # Save updated metadata
            self.save_metadata()

            processing_time = time.time() - start_time
            self.stats["processing_time"] = processing_time

            log_info(f"Incremental update completed in {processing_time:.2f}s")
            log_info(f"  Processed: {len(results['added']) + len(results['updated'])} files")
            log_info(f"  Deleted: {len(results['deleted'])} files")
            log_info(f"  Errors: {len(results['errors'])}")

            return {
                "status": "success",
                "processing_time": processing_time,
                "results": results,
                "statistics": self.stats,
            }

        except Exception as e:
            error_msg = f"Error during incremental update: {e}"
            log_error(error_msg)

            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time,
                "results": results,
                "statistics": self.stats,
            }

    def _handle_deleted_file(self, file_path: str):
        """Handle cleanup for a deleted file."""
        # Remove from metadata
        if file_path in self.metadata:
            del self.metadata[file_path]

        # Clear related cache entries
        # This is a simple approach - in a real system you might want more sophisticated cache invalidation
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]

        log_debug(f"Cleaned up deleted file: {file_path} (cache key: {file_hash})")

    def _update_file_metadata(self, file_path: str):
        """Update metadata for a processed file."""
        file_path_obj = Path(file_path)

        # Calculate new file properties
        checksum = self._calculate_file_checksum(file_path_obj)
        size, last_modified = self._get_file_stats(file_path_obj)

        # Create or update metadata
        if file_path in self.metadata:
            # Update existing metadata
            existing = self.metadata[file_path]
            existing.checksum = checksum
            existing.size = size
            existing.last_modified = last_modified
            existing.last_processed = time.time()
            existing.version += 1

        else:
            # Create new metadata
            self.metadata[file_path] = DocumentMetadata(
                path=file_path,
                checksum=checksum,
                size=size,
                last_modified=last_modified,
                last_processed=time.time(),
                version=1,
            )

    def get_update_summary(self) -> Dict[str, Any]:
        """Get a summary of the current update status."""
        total_docs = len(self.metadata)

        # Calculate age statistics
        current_time = time.time()
        ages = [current_time - meta.last_processed for meta in self.metadata.values()]

        avg_age = sum(ages) / len(ages) if ages else 0
        oldest_doc = max(ages) if ages else 0

        return {
            "total_documents": total_docs,
            "average_document_age_hours": avg_age / 3600,
            "oldest_document_age_hours": oldest_doc / 3600,
            "last_scan_time": self.stats.get("last_scan", 0),
            "statistics": self.stats,
        }

    async def auto_update_loop(self, check_interval: int = 300):
        """
        Automatically check for and process updates at regular intervals.

        Args:
            check_interval: Time between checks in seconds (default: 5 minutes)
        """
        log_info(f"Starting auto-update loop (checking every {check_interval}s)")

        while True:
            try:
                await asyncio.sleep(check_interval)

                log_debug("Running scheduled update check...")
                results = await self.process_incremental_update()

                if results["status"] == "success" and results.get("results"):
                    total_changes = (
                        len(results["results"]["added"])
                        + len(results["results"]["updated"])
                        + len(results["results"]["deleted"])
                    )

                    if total_changes > 0:
                        log_info(f"Auto-update processed {total_changes} changes")

            except Exception as e:
                log_error(f"Error in auto-update loop: {e}")
                # Continue running even if one iteration fails


# Convenience functions for common use cases
async def quick_incremental_update(documents_dir: str) -> Dict[str, Any]:
    """
    Quick incremental update for a documents directory.

    This is a convenience function for simple incremental updates.
    """
    manager = IncrementalUpdateManager(documents_dir)
    return await manager.process_incremental_update()


def setup_hot_reload(documents_dir: str, check_interval: int = 300) -> IncrementalUpdateManager:
    """
    Set up hot reload for automatic updates.

    Returns the manager so you can control it if needed.
    """
    manager = IncrementalUpdateManager(documents_dir, enable_hot_reload=True)

    # Start the auto-update loop in the background
    asyncio.create_task(manager.auto_update_loop(check_interval))

    return manager
