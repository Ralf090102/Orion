"""
File system watcher for Orion.

Monitors knowledge base directories for file changes and triggers
incremental ingestion/deletion operations.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Timer
from typing import TYPE_CHECKING, Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.retrieval.vector_store import ChromaVectorStore
from src.utilities.utils import ensure_config, log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
    from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)


class DebouncedEventHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing to reduce duplicate events.
    
    Debouncing ensures that rapid successive events (e.g., multiple writes
    during file save) are consolidated into a single callback.
    """

    def __init__(
        self,
        on_created: Optional[Callable[[str], None]] = None,
        on_modified: Optional[Callable[[str], None]] = None,
        on_deleted: Optional[Callable[[str], None]] = None,
        debounce_seconds: float = 1.0,
        ignore_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the debounced event handler.

        Args:
            on_created: Callback for file creation events
            on_modified: Callback for file modification events
            on_deleted: Callback for file deletion events
            debounce_seconds: Delay before triggering callbacks
            ignore_patterns: List of glob patterns to ignore
        """
        super().__init__()
        self.on_created_callback = on_created
        self.on_modified_callback = on_modified
        self.on_deleted_callback = on_deleted
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns or []
        
        # Debounce tracking
        self._timers: dict[str, Timer] = {}
        self._lock = Lock()

    def _should_ignore(self, path: str) -> bool:
        """Check if path matches any ignore pattern."""
        from fnmatch import fnmatch
        
        path_obj = Path(path)
        
        # Always ignore directories
        if path_obj.is_dir():
            return True
        
        # Check against ignore patterns
        for pattern in self.ignore_patterns:
            if fnmatch(path_obj.name, pattern) or fnmatch(str(path_obj), pattern):
                return True
        
        return False

    def _debounced_callback(self, event_type: str, file_path: str) -> None:
        """
        Execute debounced callback after delay.

        Args:
            event_type: Type of event ('created', 'modified', 'deleted')
            file_path: Path to the affected file
        """
        with self._lock:
            # Cancel existing timer for this file
            if file_path in self._timers:
                self._timers[file_path].cancel()
                del self._timers[file_path]

        # Schedule new callback
        def execute_callback():
            with self._lock:
                if file_path in self._timers:
                    del self._timers[file_path]
            
            # Execute the appropriate callback
            if event_type == "created" and self.on_created_callback:
                self.on_created_callback(file_path)
            elif event_type == "modified" and self.on_modified_callback:
                self.on_modified_callback(file_path)
            elif event_type == "deleted" and self.on_deleted_callback:
                self.on_deleted_callback(file_path)

        timer = Timer(self.debounce_seconds, execute_callback)
        
        with self._lock:
            self._timers[file_path] = timer
        
        timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation event."""
        if not event.is_directory and not self._should_ignore(event.src_path):
            self._debounced_callback("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event."""
        if not event.is_directory and not self._should_ignore(event.src_path):
            self._debounced_callback("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion event."""
        if not event.is_directory and not self._should_ignore(event.src_path):
            self._debounced_callback("deleted", event.src_path)

    def cancel_all_timers(self) -> None:
        """Cancel all pending debounced callbacks."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class FileWatcher:
    """
    Watches file system for changes and triggers ingestion/deletion operations.
    
    Integrates with vector store to maintain knowledge base in sync with
    file system state.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        on_file_added: Optional[Callable[[str], None]] = None,
        on_file_modified: Optional[Callable[[str], None]] = None,
        on_file_deleted: Optional[Callable[[str], None]] = None,
        config: Optional["OrionConfig"] = None,
    ):
        """
        Initialize file watcher.

        Args:
            vector_store: Vector store for deletion operations
            on_file_added: Callback for file additions
            on_file_modified: Callback for file modifications
            on_file_deleted: Callback for file deletions (if None, uses vector_store)
            config: Orion configuration
        """
        self.config = ensure_config(config)
        self.watchdog_config = self.config.watchdog
        self.vector_store = vector_store
        
        # Callbacks
        self._on_file_added = on_file_added
        self._on_file_modified = on_file_modified
        self._on_file_deleted = on_file_deleted
        
        # Watchdog components
        self.observer: Optional["Observer"] = None
        self.event_handler: Optional["DebouncedEventHandler"] = None
        self.executor: Optional["ThreadPoolExecutor"] = None
        
        # State tracking
        self._is_watching = False
        self._watched_paths: list[str] = []

    def _handle_file_created(self, file_path: str) -> None:
        """
        Handle file creation event.

        Args:
            file_path: Path to created file
        """
        log_info(f"File created: {file_path}", config=self.config)
        
        if self._on_file_added:
            try:
                # Submit to executor for async processing
                if self.executor:
                    self.executor.submit(self._on_file_added, file_path)
                else:
                    self._on_file_added(file_path)
            except Exception as e:
                log_error(f"Error processing created file {file_path}: {e}", config=self.config)

    def _handle_file_modified(self, file_path: str) -> None:
        """
        Handle file modification event.

        Args:
            file_path: Path to modified file
        """
        log_info(f"File modified: {file_path}", config=self.config)
        
        if self._on_file_modified:
            try:
                # Submit to executor for async processing
                if self.executor:
                    self.executor.submit(self._on_file_modified, file_path)
                else:
                    self._on_file_modified(file_path)
            except Exception as e:
                log_error(f"Error processing modified file {file_path}: {e}", config=self.config)

    def _handle_file_deleted(self, file_path: str) -> None:
        """
        Handle file deletion event.

        Args:
            file_path: Path to deleted file
        """
        log_info(f"File deleted: {file_path}", config=self.config)
        
        # Use custom callback if provided, otherwise use vector store
        if self._on_file_deleted:
            try:
                if self.executor:
                    self.executor.submit(self._on_file_deleted, file_path)
                else:
                    self._on_file_deleted(file_path)
            except Exception as e:
                log_error(f"Error processing deleted file {file_path}: {e}", config=self.config)
        
        elif self.vector_store:
            try:
                # Delete from vector store
                self._delete_from_vector_store(file_path)
            except Exception as e:
                log_error(f"Error deleting from vector store {file_path}: {e}", config=self.config)

    def _delete_from_vector_store(self, file_path: str) -> None:
        """
        Delete document chunks associated with file from vector store.

        Args:
            file_path: Path to the deleted file
        """
        if not self.vector_store:
            log_warning("No vector store available for deletion", config=self.config)
            return
        
        try:
            # Get all chunks from this source file
            chunks = self.vector_store.get_chunks_by_source(file_path)
            
            if not chunks:
                log_debug(f"No chunks found for deleted file: {file_path}", self.config)
                return
            
            # Extract document IDs
            doc_ids = [chunk["id"] for chunk in chunks]
            
            # Delete from vector store
            success = self.vector_store.delete_documents(doc_ids)
            
            if success:
                log_info(f"Deleted {len(doc_ids)} chunks for file: {file_path}", config=self.config)
            else:
                log_warning(f"Failed to delete chunks for file: {file_path}", config=self.config)
                
        except Exception as e:
            log_error(f"Error deleting chunks for {file_path}: {e}", config=self.config)

    def start(self, paths: Optional[list[str]] = None) -> bool:
        """
        Start watching directories for file changes.

        Args:
            paths: List of directory paths to watch. If None, uses config.watchdog.paths

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_watching:
            log_warning("File watcher is already running", config=self.config)
            return False
        
        # Use provided paths or fall back to config
        watch_paths = paths or self.watchdog_config.paths
        
        if not watch_paths:
            log_error("No paths specified for file watching", config=self.config)
            return False
        
        # Validate paths
        valid_paths = []
        for path_str in watch_paths:
            path = Path(path_str)
            if not path.exists():
                log_warning(f"Watch path does not exist: {path_str}", config=self.config)
                continue
            if not path.is_dir():
                log_warning(f"Watch path is not a directory: {path_str}", config=self.config)
                continue
            valid_paths.append(str(path.resolve()))
        
        if not valid_paths:
            log_error("No valid paths to watch", config=self.config)
            return False
        
        try:
            # Create executor for async processing
            self.executor = ThreadPoolExecutor(
                max_workers=self.watchdog_config.max_workers,
                thread_name_prefix="watchdog_worker"
            )
            
            # Create event handler with debouncing
            self.event_handler = DebouncedEventHandler(
                on_created=self._handle_file_created,
                on_modified=self._handle_file_modified,
                on_deleted=self._handle_file_deleted,
                debounce_seconds=self.watchdog_config.debounce_seconds,
                ignore_patterns=self.watchdog_config.ignore_patterns,
            )
            
            # Create and start observer
            self.observer = Observer()
            
            for path in valid_paths:
                self.observer.schedule(
                    self.event_handler,
                    path,
                    recursive=self.watchdog_config.recursive
                )
                log_info(f"Watching directory: {path} (recursive={self.watchdog_config.recursive})", config=self.config)
            
            self.observer.start()
            self._is_watching = True
            self._watched_paths = valid_paths
            
            log_info(f"File watcher started for {len(valid_paths)} directories", config=self.config)
            return True
            
        except Exception as e:
            log_error(f"Failed to start file watcher: {e}", config=self.config)
            self._cleanup()
            return False

    def stop(self) -> bool:
        """
        Stop watching directories.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._is_watching:
            log_warning("File watcher is not running", config=self.config)
            return False
        
        try:
            log_info("Stopping file watcher...", config=self.config)
            
            # Cancel pending debounced events
            if self.event_handler:
                self.event_handler.cancel_all_timers()
            
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5.0)
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            
            self._cleanup()
            
            log_info("File watcher stopped successfully", config=self.config)
            return True
            
        except Exception as e:
            log_error(f"Error stopping file watcher: {e}", config=self.config)
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_watching = False
        self._watched_paths = []
        self.observer = None
        self.event_handler = None
        self.executor = None

    def is_watching(self) -> bool:
        """Check if watcher is currently active."""
        return self._is_watching

    def get_watched_paths(self) -> list[str]:
        """Get list of currently watched paths."""
        return self._watched_paths.copy()

    def get_status(self) -> dict:
        """
        Get watcher status information.

        Returns:
            Dictionary with status details
        """
        return {
            "is_watching": self._is_watching,
            "watched_paths": self._watched_paths,
            "path_count": len(self._watched_paths),
            "debounce_seconds": self.watchdog_config.debounce_seconds,
            "max_workers": self.watchdog_config.max_workers,
            "ignore_patterns": self.watchdog_config.ignore_patterns,
            "recursive": self.watchdog_config.recursive,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def create_file_watcher(
    vector_store: Optional[ChromaVectorStore] = None,
    on_file_added: Optional[Callable[[str], None]] = None,
    on_file_modified: Optional[Callable[[str], None]] = None,
    on_file_deleted: Optional[Callable[[str], None]] = None,
    config: Optional["OrionConfig"] = None,
) -> FileWatcher:
    """
    Factory function to create a FileWatcher instance.

    Args:
        vector_store: Vector store for deletion operations
        on_file_added: Callback for file additions
        on_file_modified: Callback for file modifications
        on_file_deleted: Callback for file deletions
        config: Orion configuration

    Returns:
        Configured FileWatcher instance

    Example:
        # Basic usage with vector store
        watcher = create_file_watcher(vector_store=store)
        watcher.start()
        
        # Custom callbacks
        def handle_new_file(path):
            print(f"New file: {path}")
        
        watcher = create_file_watcher(
            on_file_added=handle_new_file,
            on_file_modified=handle_new_file,
        )
        watcher.start()
        
        # Context manager
        with create_file_watcher(vector_store=store) as watcher:
            # Watcher is active
            time.sleep(60)
        # Watcher automatically stopped
    """
    return FileWatcher(
        vector_store=vector_store,
        on_file_added=on_file_added,
        on_file_modified=on_file_modified,
        on_file_deleted=on_file_deleted,
        config=config,
    )
