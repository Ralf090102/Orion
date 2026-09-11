"""
Root-logger configuration, called once at import time in backend/app.py,
before any other backend/src module logs anything.

One root configuration covers both backend/*.py's `logging.getLogger(__name__)`
calls and src/*.py's "orion" logger (src/utilities/utils.py), which has no
handler of its own and relies entirely on propagation to root. Deliberately
does not touch the pre-existing, never-called setup_logging()/LoggingConfig
in src/utilities/utils.py / src/utilities/config.py -- that's dead code
(confirmed: setup_logging() has no callers anywhere in the repo), left alone
and out of scope here rather than revived or deleted.

Independent of the Rust shell's own capture of this process's stdout/stderr
(frontend/src-tauri/src/backend.rs, which relays piped output into
orion-shell.log) -- this also persists logs when the backend runs standalone
(dev/debug outside Tauri), and Python's own structured log records survive
even if the Rust-side capture has issues.

Every log record is tagged with the current request's correlation ID (see
src/utilities/request_context.py) via a logging.Filter, so log lines for one
chat query -- across backend/*.py and src/*.py alike -- can be grepped
together by that ID.
"""

import logging
import logging.handlers
import os
from pathlib import Path

_FMT = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"


class _RequestIdFilter(logging.Filter):
    """Attaches the current request's correlation ID (or "-" outside any
    request context) to every log record, so the formatter above can
    include it. See src/utilities/request_context.py for how the ID is
    minted and bound."""

    def filter(self, record: logging.LogRecord) -> bool:
        from src.utilities.request_context import request_id_var

        record.request_id = request_id_var.get()
        return True


def _log_dir() -> Path:
    # Same ORION_DATA_DIR convention as backend/dependencies.py's
    # _session_storage_dir() -- Tauri sets this to a stable per-user
    # app-data dir; falls back to ./data relative to CWD when unset
    # (e.g. running the backend standalone outside Tauri).
    data_dir = os.environ.get("ORION_DATA_DIR")
    return (Path(data_dir) if data_dir else Path("./data")) / "logs"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler and a rotating
    file handler under ${ORION_DATA_DIR}/logs/orion-backend.log. Idempotent
    -- safe to call more than once (e.g. from tests), since it clears any
    handlers it previously installed before adding new ones."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    rid_filter = _RequestIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    console.addFilter(rid_filter)
    root.addHandler(console)

    log_dir = _log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "orion-backend.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # bounded: ~60MB total, not unbounded growth
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_FMT))
        file_handler.addFilter(rid_filter)
        root.addHandler(file_handler)
    except OSError as e:
        # Don't crash startup over a logging directory problem -- console
        # logging above still works, just without persistence this run.
        root.warning(f"Could not set up file logging at {log_dir}: {e}")
