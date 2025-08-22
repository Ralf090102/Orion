"""
Media Processing Module for Orion RAG Pipeline
Handles OCR, image analysis, and basic media metadata extraction.

Phase 1: OCR and basic image processing
Phase 2: Table detection and structured extraction
"""

import importlib.util
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageEnhance
from datetime import datetime

from app.utils import log_info, log_warning, log_error, timer


class OCRProcessor:
    """
    Handles Optical Character Recognition for images and scanned documents.
    Supports multiple OCR backends with fallback options.
    """

    def __init__(self, preferred_backend: str = "easyocr"):
        self.preferred_backend = preferred_backend
        self.available_backends = self._check_available_backends()

        if not self.available_backends:
            log_warning("No OCR backends available. Install easyocr or pytesseract.")

        log_info(f"OCR initialized with backends: {self.available_backends}")

    def _check_available_backends(self) -> List[str]:
        """Check which OCR backends are available."""
        backends = []

        # Check EasyOCR
        if importlib.util.find_spec("easyocr") is not None:
            backends.append("easyocr")

        # Check Tesseract
        if importlib.util.find_spec("pytesseract") is not None:
            backends.append("pytesseract")

        return backends

    def _preprocess_image(self, image_path: Path) -> Image.Image:
        """
        Preprocess image for better OCR results.
        - Convert to grayscale
        - Enhance contrast
        - Resize if too small
        """
        img = Image.open(image_path)

        # Convert to RGB if needed (handles RGBA, P mode, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Convert to grayscale for better OCR
        img = img.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # Resize if too small (OCR works better on larger images)
        width, height = img.size
        if width < 300 or height < 300:
            scale_factor = max(300 / width, 300 / height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    @timer
    def extract_text_easyocr(self, image_path: Path) -> Dict[str, Any]:
        """Extract text using EasyOCR (more accurate, slower)."""
        try:
            import easyocr

            # Initialize reader (this is cached after first use)
            if not hasattr(self, "_easyocr_reader"):
                self._easyocr_reader = easyocr.Reader(
                    ["en"], gpu=False
                )  # Use CPU for compatibility

            # Preprocess image
            img = self._preprocess_image(image_path)

            # Extract text
            results = self._easyocr_reader.readtext(img, detail=1)

            # Process results
            text_blocks = []
            full_text = ""
            confidence_scores = []

            for bbox, text, confidence in results:
                if confidence > 0.5:  # Filter low-confidence text
                    text_blocks.append(
                        {"text": text, "confidence": confidence, "bbox": bbox}
                    )
                    full_text += text + " "
                    confidence_scores.append(confidence)

            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0
            )

            return {
                "text": full_text.strip(),
                "method": "easyocr",
                "confidence": avg_confidence,
                "text_blocks": text_blocks,
                "block_count": len(text_blocks),
            }

        except Exception as e:
            log_error(f"EasyOCR failed for {image_path}: {e}")
            return {"text": "", "method": "easyocr", "error": str(e)}

    @timer
    def extract_text_tesseract(self, image_path: Path) -> Dict[str, Any]:
        """Extract text using Tesseract (faster, less accurate)."""
        try:
            import pytesseract

            # Preprocess image
            img = self._preprocess_image(image_path)

            # Extract text
            text = pytesseract.image_to_string(img, lang="eng")

            # Get confidence data
            try:
                data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )
            except Exception:
                avg_confidence = 0.8  # Fallback confidence

            return {
                "text": text.strip(),
                "method": "pytesseract",
                "confidence": avg_confidence / 100.0,  # Convert to 0-1 scale
                "word_count": len(text.split()) if text.strip() else 0,
            }

        except Exception as e:
            log_error(f"Tesseract failed for {image_path}: {e}")
            return {"text": "", "method": "pytesseract", "error": str(e)}

    def extract_text(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract text from image using the best available method.
        """
        if not self.available_backends:
            return {"text": "", "error": "No OCR backends available"}

        # Try preferred backend first
        if self.preferred_backend == "easyocr" and "easyocr" in self.available_backends:
            result = self.extract_text_easyocr(image_path)
            if result.get("text") or not result.get("error"):
                return result

        # Try tesseract as backup
        if "pytesseract" in self.available_backends:
            result = self.extract_text_tesseract(image_path)
            if result.get("text") or not result.get("error"):
                return result

        # Try easyocr as final backup
        if "easyocr" in self.available_backends and self.preferred_backend != "easyocr":
            return self.extract_text_easyocr(image_path)

        return {"text": "", "error": "All OCR methods failed"}


class TableDetector:
    """
    Detects and extracts tables from images and PDFs.
    Phase 2 feature for structured data extraction.
    """

    def __init__(self):
        self.available_methods = self._check_table_detection_methods()
        log_info(f"Table detection initialized with methods: {self.available_methods}")

    def _check_table_detection_methods(self) -> List[str]:
        """Check which table detection methods are available."""
        methods = []

        # Check if camelot is available (for PDF table extraction)
        if importlib.util.find_spec("camelot") is not None:
            methods.append("camelot")

        # Check if tabula is available (alternative PDF table extraction)
        if importlib.util.find_spec("tabula") is not None:
            methods.append("tabula")

        # Basic table detection always available (using image processing)
        methods.append("basic")

        return methods

    def detect_tables_in_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF using camelot or tabula."""
        tables = []

        if "camelot" in self.available_methods:
            try:
                import camelot

                # Extract tables
                table_list = camelot.read_pdf(str(pdf_path), pages="all")

                for i, table in enumerate(table_list):
                    df = table.df
                    if not df.empty:
                        tables.append(
                            {
                                "table_id": i,
                                "data": df.to_dict("records"),
                                "csv": df.to_csv(index=False),
                                "shape": df.shape,
                                "method": "camelot",
                                "accuracy": getattr(table, "accuracy", 0.8),
                            }
                        )

                log_info(
                    f"Extracted {len(tables)} tables from {pdf_path} using camelot"
                )

            except Exception as e:
                log_warning(f"Camelot table extraction failed: {e}")

        elif "tabula" in self.available_methods:
            try:
                import tabula

                # Extract tables
                dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)

                for i, df in enumerate(dfs):
                    if not df.empty:
                        tables.append(
                            {
                                "table_id": i,
                                "data": df.to_dict("records"),
                                "csv": df.to_csv(index=False),
                                "shape": df.shape,
                                "method": "tabula",
                            }
                        )

                log_info(f"Extracted {len(tables)} tables from {pdf_path} using tabula")

            except Exception as e:
                log_warning(f"Tabula table extraction failed: {e}")

        return tables

    def detect_tables_in_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Basic table detection in images using image processing."""
        # This would use computer vision to detect table-like structures
        # For now, return empty - this is a complex feature
        return []


class MediaProcessor:
    """
    Main media processing coordinator.
    Handles all media types and delegates to specialized processors.
    """

    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.table_detector = TableDetector()

        # Track processing statistics
        self.stats = {
            "images_processed": 0,
            "text_extracted": 0,
            "tables_found": 0,
            "processing_time": 0,
        }

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process an image file and extract all possible information.
        """
        start_time = datetime.now()

        try:
            # Get basic image metadata
            with Image.open(image_path) as img:
                metadata = {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "size_bytes": image_path.stat().st_size,
                }

            # Extract text using OCR
            ocr_result = self.ocr_processor.extract_text(image_path)

            # Detect tables (if text was found)
            tables = []
            if ocr_result.get("text"):
                tables = self.table_detector.detect_tables_in_image(image_path)

            # Update statistics
            self.stats["images_processed"] += 1
            if ocr_result.get("text"):
                self.stats["text_extracted"] += 1
            self.stats["tables_found"] += len(tables)

            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["processing_time"] += processing_time

            return {
                "type": "image",
                "metadata": metadata,
                "ocr": ocr_result,
                "tables": tables,
                "processing_time": processing_time,
                "success": True,
            }

        except Exception as e:
            log_error(f"Image processing failed for {image_path}: {e}")
            return {"type": "image", "error": str(e), "success": False}

    def process_pdf_enhanced(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Enhanced PDF processing with table detection.
        """
        start_time = datetime.now()

        try:
            # Extract tables from PDF
            tables = self.table_detector.detect_tables_in_pdf(pdf_path)

            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["tables_found"] += len(tables)
            self.stats["processing_time"] += processing_time

            return {
                "type": "pdf_enhanced",
                "tables": tables,
                "processing_time": processing_time,
                "success": True,
            }

        except Exception as e:
            log_error(f"Enhanced PDF processing failed for {pdf_path}: {e}")
            return {"type": "pdf_enhanced", "error": str(e), "success": False}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "avg_processing_time": (
                self.stats["processing_time"] / max(self.stats["images_processed"], 1)
            ),
            "text_extraction_rate": (
                self.stats["text_extracted"] / max(self.stats["images_processed"], 1)
            ),
        }


# Global instance for easy access
media_processor = MediaProcessor()
