"""
Utility functions for Orion project.
"""

import time
import os
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Global verbose flag - can be set by config
_verbose_mode = False


def set_verbose_mode(verbose: bool):
    """Set global verbose mode for logging"""
    global _verbose_mode
    _verbose_mode = verbose


def log_info(message: str, verbose_only: bool = False):
    """Log info message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or _verbose_mode:
        console.print(f"[bold cyan]ℹ️  {message}[/bold cyan]")


def log_success(message: str, verbose_only: bool = False):
    """Log success message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or _verbose_mode:
        console.print(f"[bold green]✅ {message}[/bold green]")


def log_warning(message: str, verbose_only: bool = False):
    """Log warning message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or _verbose_mode:
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")


def log_error(message: str, verbose_only: bool = False):
    """Log error message. If verbose_only=True, only shows in verbose mode."""
    if not verbose_only or _verbose_mode:
        console.print(f"[bold red]❌ {message}[/bold red]")


def log_debug(message: str):
    """Log debug message - only shows in verbose mode."""
    if _verbose_mode:
        console.print(f"[dim]🔍 {message}[/dim]")


def log_progress(message: str, verbose_only: bool = False):
    """Log progress message with special formatting."""
    if not verbose_only or _verbose_mode:
        console.print(f"[bold blue]🔄 {message}[/bold blue]")


def timer(func):
    """Decorator to measure execution time of functions (logs on success/failure)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            log_info(f"{func.__name__} completed in {end - start:.2f}s", verbose_only=True)

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
