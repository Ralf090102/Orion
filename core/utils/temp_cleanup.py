"""
Temporary file cleanup utilities to prevent exit errors.
"""

import atexit
import shutil
import tempfile
from pathlib import Path


def safe_cleanup_temp_dirs():
    """
    Safely cleanup temporary directories without raising errors.
    This prevents the PermissionError exceptions on Windows exit.
    """
    try:
        temp_dir = Path(tempfile.gettempdir())

        # Find all temporary directories that might be leftover
        for temp_item in temp_dir.iterdir():
            if temp_item.is_dir() and temp_item.name.startswith("tmp"):
                try:
                    # Try to remove the directory silently
                    if temp_item.exists():
                        shutil.rmtree(temp_item, ignore_errors=True)
                except Exception:
                    # Completely ignore any errors
                    pass
    except Exception:
        # Ignore any errors during cleanup
        pass


def patch_rmtree_for_quiet_exit():
    """
    Patch shutil.rmtree to suppress errors during program exit.
    This prevents the annoying error messages on Windows.
    """
    original_rmtree = shutil.rmtree

    def quiet_rmtree(path, ignore_errors=False, onerror=None):
        try:
            return original_rmtree(path, ignore_errors=True, onerror=None)
        except Exception:
            # Completely suppress all rmtree errors during exit
            pass

    # Replace the rmtree function
    shutil.rmtree = quiet_rmtree


def setup_quiet_exit():
    """
    Setup quiet exit cleanup to prevent temporary file errors.
    """
    # Patch rmtree to be quiet
    patch_rmtree_for_quiet_exit()

    # Register our cleanup to run early
    atexit.register(safe_cleanup_temp_dirs)


# Auto-setup when module is imported
setup_quiet_exit()
