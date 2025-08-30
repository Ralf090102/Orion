"""
Utility functions for Orion project.
"""

import time
import os
import logging
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from .config import OrionConfig

console = Console()

# Logger instance - will be configured by config
logger = logging.getLogger("orion")


def setup_logging(config: "OrionConfig"):
    """Setup logging based on configuration"""
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set log level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger.setLevel(level_map[config.logging.level.value])
    
    # Console handler
    if config.logging.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_map[config.logging.level.value])
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.logging.log_to_file:
        os.makedirs(os.path.dirname(config.logging.log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(config.logging.log_file_path)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def log_info(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log info message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or (config and config.logging.verbose):
        console.print(f"[bold cyan]ℹ️  {message}[/bold cyan]")
        logger.info(message)


def log_success(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log success message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or (config and config.logging.verbose):
        console.print(f"[bold green]✅ {message}[/bold green]")
        logger.info(f"SUCCESS: {message}")


def log_warning(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log warning message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or (config and config.logging.verbose):
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
        logger.warning(message)


def log_error(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log error message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or (config and config.logging.verbose):
        console.print(f"[bold red]❌ {message}[/bold red]")
        logger.error(message)


def log_debug(message: str, config: Optional["OrionConfig"] = None):
    """Log debug message - only shows in verbose mode."""
    if config and config.logging.verbose:
        console.print(f"[dim]🔍 {message}[/dim]")
        logger.debug(message)


def log_progress(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log progress message with special formatting."""
    if not verbose_only or (config and config.logging.verbose):
        console.print(f"[bold blue]🔄 {message}[/bold blue]")
        logger.info(f"PROGRESS: {message}")


def timer(func):
    """Decorator to measure execution time of functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{func.__name__} completed in {end - start:.2f}s")
            return result
        except Exception as e:
            end = time.time()
            logger.error(f"{func.__name__} failed after {end - start:.2f}s: {str(e)}")
            raise

    return wrapper


def validate_path(
    path: str,
    must_exist: bool = True,
    path_type: Literal["any", "dir", "file"] = "dir",
) -> Path:
    """
    Validates and converts path to Path object.

    Args:
        path: String path to validate
        must_exist: Whether the path must exist
        path_type: "dir", "file", or "any"

    Returns:
        Path object

    Raises:
        FileNotFoundError: If path doesn't exist and must_exist=True
        NotADirectoryError: If path exists but is not a directory
        IsADirectoryError: If path exists but is a directory when file is expected
    """
    p = Path(path)

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if p.exists():
        if path_type == "dir" and not p.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        if path_type == "file" and not p.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")

    return p


def ensure_directory(path: str) -> Path:
    """
    Ensures directory exists, creates if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_count(folder_path: str, extensions: Optional[list] = None) -> int:
    """
    Count files in a directory with optional extension filtering.

    Args:
        folder_path: Path to directory
        extensions: List of extensions to filter (e.g., ['.pdf', '.docx'])

    Returns:
        Number of matching files
    """
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        return 0

    if extensions:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

        return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in exts)

    return sum(1 for f in path.iterdir() if f.is_file())


def create_progress_bar(description: str = "Processing"):
    """Create a Rich progress bar with spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


@contextmanager
def progress_task(description: str = "Processing"):
    """Context manager for a progress task."""
    with create_progress_bar(description) as progress:
        task_id = progress.add_task(description=description, total=None)
        try:
            yield progress, task_id
        finally:
            progress.update(task_id, completed=True)


def validate_path_exists(path: str) -> bool:
    """
    Check if path exists.

    Args:
        path: The path to check.

    Returns:
        True if path exists, False otherwise.
    """
    if not os.path.exists(path):
        log_error(f"Path not found: {path}")
        return False
    return True


def validate_nonempty_string(value: str, error_message: str) -> bool:
    """
    Check if a string is non-empty after stripping.

    Args:
        value: The string value to check.
        error_message: The error message to log if validation fails.

    Returns:
        True if the string is non-empty, False otherwise.
    """
    if not value.strip():
        log_error(error_message)
        return False
    return True
