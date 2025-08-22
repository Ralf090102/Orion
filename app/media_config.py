"""
Configuration for Enhanced Media Processing
"""

import importlib.util

# Installation Commands for Optional Dependencies
INSTALLATION_COMMANDS = {
    "easyocr": "pip install easyocr",
    "pytesseract": "pip install pytesseract",
    "camelot": "pip install camelot-py[cv]",
    "tabula": "pip install tabula-py",
    "tesseract_binary": {
        "windows": "Download from: https://github.com/UB-Mannheim/tesseract/wiki",
        "linux": "sudo apt-get install tesseract-ocr",
        "mac": "brew install tesseract",
    },
}

# Processing Configuration
PROCESSING_CONFIG = {
    "ocr": {
        "preferred_backend": "easyocr",  # or "pytesseract"
        "min_confidence": 0.5,
        "preprocess_images": True,
        "min_text_length": 10,  # Minimum characters to consider OCR successful
        "resize_threshold": 300,  # Minimum dimension for image resize
        "contrast_enhancement": 1.5,
    },
    "table_detection": {
        "preferred_method": "camelot",  # or "tabula"
        "extract_all_pages": True,
        "min_table_rows": 2,
        "min_accuracy": 0.7,  # For camelot
    },
    "performance": {
        "max_image_processing_time": 30,  # seconds
        "max_pdf_table_extraction_time": 60,  # seconds
        "batch_size": 10,  # Process images in batches
        "enable_caching": True,
        "cache_ocr_results": True,
    },
}

# File Type Categories
FILE_CATEGORIES = {
    "images": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"},
    "documents": {".pdf", ".docx", ".txt", ".md", ".rtf"},
    "spreadsheets": {".xlsx", ".xls", ".csv"},
    "archives": {".zip", ".tar", ".gz", ".7z"},
    "media": {".mp3", ".mp4", ".wav", ".avi", ".mov"},  # Future support
}

# Quality Thresholds
QUALITY_THRESHOLDS = {
    "ocr_confidence": 0.6,
    "maintainability_index": 60.0,
    "complexity_warning": 10,
    "min_useful_text": 20,
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_ocr": True,
    "enable_table_detection": True,
    "enable_image_metadata": True,
    "enable_performance_tracking": True,
    "fallback_on_failure": True,
    "detailed_logging": True,
}


def get_processing_config():
    """Get the current processing configuration."""
    return PROCESSING_CONFIG


def get_installation_help():
    """Get installation instructions for optional dependencies."""
    help_text = []
    help_text.append("=== Phase 1 & 2 Enhanced Processing Dependencies ===\n")

    help_text.append("Core Image Processing:")
    help_text.append(f"  {INSTALLATION_COMMANDS['easyocr']}")
    help_text.append(f"  {INSTALLATION_COMMANDS['pytesseract']}")
    help_text.append("")

    help_text.append("Table Extraction (Phase 2):")
    help_text.append(f"  {INSTALLATION_COMMANDS['camelot']}")
    help_text.append(f"  {INSTALLATION_COMMANDS['tabula']}")
    help_text.append("")

    help_text.append("Tesseract Binary (if using pytesseract):")
    for platform, cmd in INSTALLATION_COMMANDS["tesseract_binary"].items():
        help_text.append(f"  {platform}: {cmd}")
    help_text.append("")

    help_text.append("Test Installation:")
    test_command = (
        'python -c "from app.media_processing import media_processor; '
        'print(media_processor.ocr_processor.available_backends)"'
    )
    help_text.append(f"  {test_command}")

    return "\n".join(help_text)


def check_dependencies():
    """Check which optional dependencies are available."""
    available = {
        "easyocr": False,
        "pytesseract": False,
        "camelot": False,
        "tabula": False,
        "pillow": False,
    }

    # Use importlib.util.find_spec to check for module availability
    available["easyocr"] = importlib.util.find_spec("easyocr") is not None
    available["pytesseract"] = importlib.util.find_spec("pytesseract") is not None
    available["camelot"] = importlib.util.find_spec("camelot") is not None
    available["tabula"] = importlib.util.find_spec("tabula") is not None
    available["pillow"] = importlib.util.find_spec("PIL") is not None

    return available
