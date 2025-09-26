import asyncio
import logging
import sys
import os
import re
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

if TYPE_CHECKING:
    from .config import OrionConfig

console = Console()

# Logger instance - will be configured by config
logger = logging.getLogger("orion")

# Minimal fallback logger
_fallback_logger = logging.getLogger("orion.fallback")
if not _fallback_logger.handlers:
    fh = logging.StreamHandler(sys.stderr)
    fh.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    _fallback_logger.addHandler(fh)
    _fallback_logger.setLevel(logging.WARNING)


def _fallback_log(level: str, message: str) -> None:
    try:
        if level == "error":
            _fallback_logger.error(message)
        elif level == "warning":
            _fallback_logger.warning(message)
        elif level == "info":
            _fallback_logger.info(message)
        else:
            _fallback_logger.debug(message)
    except Exception:
        # Last resort: write directly to stderr
        sys.stderr.write(message + "\n")


def get_colored_text(text: str, color: str = "white") -> str:
    """
    Return colored text for console output.

    Args:
        text: Text to colorize
        color: Color name (white, red, green, blue, yellow, cyan, magenta, etc.)

    Returns:
        Colored text string that can be printed directly
    """
    color_map = {
        "white": "white",
        "red": "red",
        "green": "green",
        "blue": "blue",
        "yellow": "yellow",
        "cyan": "cyan",
        "magenta": "magenta",
        "black": "black",
        "bright_red": "bright_red",
        "bright_green": "bright_green",
        "bright_blue": "bright_blue",
        "bright_yellow": "bright_yellow",
        "bright_cyan": "bright_cyan",
        "bright_magenta": "bright_magenta",
    }

    rich_color = color_map.get(color.lower(), "white")

    rich_text = Text(text, style=rich_color)

    with console.capture() as capture:
        console.print(rich_text, end="")

    return capture.get()


def ensure_config(config: Optional["OrionConfig"] = None, from_env: bool = True) -> "OrionConfig":
    """
    Ensure we have a valid config object, using default if none provided.

    Args:
        config: Optional configuration object
        from_env: If True and config is None, load from environment variables

    Returns:
        Configuration object (either provided, from env, or default)
    """
    if config is None:
        from .config import get_config, OrionConfig

        try:
            return get_config(from_env=from_env)
        except Exception as e:
            _fallback_log("warning", f"Failed to load config from environment: {e}. Using defaults.")
            try:
                return get_config(from_env=False)
            except Exception as e2:
                _fallback_log("warning", f"Failed to load default config: {e2}. Returning minimal OrionConfig().")
                return OrionConfig()
    return config


def setup_logging(config: Optional["OrionConfig"] = None):
    """Setup logging based on configuration with enhanced error handling"""

    config = ensure_config(config)

    logger.handlers.clear()

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger.setLevel(level_map[config.logging.level.value])

    # Console handler
    if config.logging.log_to_console:
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level_map[config.logging.level.value])
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            _fallback_log("warning", f"Failed to setup console logging: {e}")

    # File handler
    if config.logging.log_to_file:
        try:
            log_dir = os.path.dirname(config.logging.log_file_path)
            if log_dir:  # Only create directory if there is one
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(config.logging.log_file_path)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            _fallback_log("warning", f"Failed to setup file logging: {e}")
            # Fall back to console logging if file logging fails
            if not config.logging.log_to_console and not logger.handlers:
                try:
                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(level_map[config.logging.level.value])
                    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                    console_handler.setFormatter(formatter)
                    logger.addHandler(console_handler)
                    _fallback_log("info", "Fallback: Using console logging instead")
                except Exception:
                    pass


def log_info(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log info message. If verbose_only=True, only shows in verbose mode."""

    config = ensure_config(config)

    if not verbose_only or config.logging.verbose:
        console.print(f"[bold cyan]ℹ️  {message}[/bold cyan]")
        logger.info(message)


def log_success(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log success message. If verbose_only=True, only shows in verbose mode."""

    config = ensure_config(config)

    if not verbose_only or config.logging.verbose:
        console.print(f"[bold green]✅ {message}[/bold green]")
        logger.info(f"SUCCESS: {message}")


def log_warning(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log warning message. If verbose_only=True, only shows in verbose mode."""

    config = ensure_config(config)

    if not verbose_only or config.logging.verbose:
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
        logger.warning(message)


def log_error(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log error message. If verbose_only=True, only shows in verbose mode."""

    config = ensure_config(config)

    if not verbose_only or config.logging.verbose:
        console.print(f"[bold red]❌ {message}[/bold red]")
        logger.error(message)


def log_debug(message: str, config: Optional["OrionConfig"] = None):
    """Log debug message - only shows in verbose mode."""

    config = ensure_config(config)

    if config.logging.verbose:
        console.print(f"[dim]🔍 {message}[/dim]")
        logger.debug(message)


def log_progress(message: str, verbose_only: bool = False, config: Optional["OrionConfig"] = None):
    """Log progress message with special formatting."""

    config = ensure_config(config)

    if not verbose_only or config.logging.verbose:
        console.print(f"[bold blue]🔄 {message}[/bold blue]")
        logger.info(f"PROGRESS: {message}")


def timer(func):
    """Decorator to measure execution time of functions with enhanced monitoring."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        start_process_time = time.process_time()

        try:
            result = func(*args, **kwargs)
            end = time.time()
            end_process_time = time.process_time()

            wall_time = end - start
            cpu_time = end_process_time - start_process_time

            logger.info(f"{func.__name__} completed - Wall time: {wall_time:.2f}s, " f"CPU time: {cpu_time:.2f}s")
            return result

        except Exception as e:
            end = time.time()
            end_process_time = time.process_time()

            wall_time = end - start
            cpu_time = end_process_time - start_process_time

            logger.error(f"{func.__name__} failed after {wall_time:.2f}s " f"(CPU: {cpu_time:.2f}s): {str(e)}")
            raise

    return wrapper


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


def get_file_count(folder_path: str, extensions: list | None = None) -> int:
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


# ========== RAG-SPECIFIC UTILITIES ==========
def clean_text(text: str) -> str:
    """
    Clean text for RAG processing by removing extra whitespace and normalizing.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)

    return text


def validate_chunk_size(text: str, min_size: int = 10, max_size: int = 2000) -> bool:
    """
    Validate if text chunk is appropriate size for processing.

    Args:
        text: Text chunk to validate
        min_size: Minimum acceptable character count
        max_size: Maximum acceptable character count

    Returns:
        True if chunk size is valid
    """
    if not text or not text.strip():
        return False

    length = len(text.strip())
    return min_size <= length <= max_size


def extract_file_metadata(file_path: str | Path) -> dict:
    """
    Extract metadata from a file for RAG indexing.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary containing file metadata
    """
    path = Path(file_path)

    if not path.exists():
        return {}

    try:
        stat = path.stat()
        created_time = getattr(stat, "st_birthtime", None) or getattr(stat, "st_ctime", None)
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": created_time,
            "relative_path": str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path),
        }
    except Exception as e:
        log_warning(f"Could not extract metadata from {file_path}: {e}")
        return {"filename": path.name, "extension": path.suffix.lower()}


def get_supported_document_extensions() -> list[str]:
    """
    Get list of supported document extensions for RAG processing.

    Returns:
        List of supported file extensions
    """
    return [
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".doc",
    ]


def is_supported_document(file_path: str | Path) -> bool:
    """
    Check if file extension is supported for document processing.

    Args:
        file_path: Path to the file

    Returns:
        True if file type is supported
    """
    extension = Path(file_path).suffix.lower()
    return extension in get_supported_document_extensions()


def calculate_text_stats(text: str) -> dict:
    """
    Calculate basic statistics about text content.

    Args:
        text: Text to analyze

    Returns:
        Dictionary containing text statistics
    """
    if not text:
        return {"char_count": 0, "word_count": 0, "line_count": 0, "paragraph_count": 0}

    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count("\n") + 1,
        "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage and indexing.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    sanitized = re.sub(r"_{2,}", "_", sanitized)
    sanitized = sanitized.strip("_")

    if not sanitized:
        sanitized = "unnamed_file"

    return sanitized


def estimate_processing_time(file_count: int, avg_file_size_kb: float = 100) -> str:
    """
    Estimate processing time for RAG indexing.

    Args:
        file_count: Number of files to process
        avg_file_size_kb: Average file size in KB

    Returns:
        Formatted time estimate string
    """
    # Rough estimates: ~10 files/second for small files
    base_time_per_file = 0.1  # seconds
    size_factor = max(1.0, avg_file_size_kb / 100)  # Scale by size

    total_seconds = file_count * base_time_per_file * size_factor

    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds / 60)} minutes"
    else:
        return f"~{int(total_seconds / 3600)} hours"


# ========== ASYNC UTILITIES ==========


def async_timer(func):
    """Decorator to measure execution time of async functions."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        start_process_time = time.process_time()

        try:
            result = await func(*args, **kwargs)
            end = time.time()
            end_process_time = time.process_time()

            wall_time = end - start
            cpu_time = end_process_time - start_process_time

            logger.info(f"{func.__name__} completed - Wall time: {wall_time:.2f}s, " f"CPU time: {cpu_time:.2f}s")
            return result

        except Exception as e:
            end = time.time()
            end_process_time = time.process_time()

            wall_time = end - start
            cpu_time = end_process_time - start_process_time

            logger.error(f"{func.__name__} failed after {wall_time:.2f}s " f"(CPU: {cpu_time:.2f}s): {str(e)}")
            raise

    return wrapper


@asynccontextmanager
async def async_progress_task(description: str = "Processing"):
    """Async context manager for progress tracking."""
    start_time = time.time()
    log_progress(f"Starting: {description}")

    try:
        yield
        end_time = time.time()
        log_success(f"Completed: {description} in {end_time - start_time:.2f}s")
    except Exception as e:
        end_time = time.time()
        log_error(f"Failed: {description} after {end_time - start_time:.2f}s - {str(e)}")
        raise


async def run_with_timeout(coro: Awaitable, timeout: float, description: str = "Operation") -> Any | None:
    """
    Run an async function with timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        description: Description for logging

    Returns:
        Result or None if timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log_warning(f"{description} timed out after {timeout}s")
        return None
    except Exception as e:
        log_error(f"{description} failed: {str(e)}")
        raise