"""
Additional tests for enhanced ingest functionality with media processing.
Tests the integration of OCR and table extraction in document ingestion.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from core.rag.ingest import extract_metadata, load_documents


class TestEnhancedMetadataExtraction:
    """Test enhanced metadata extraction with media processing."""

    @patch("app.ingest.MetadataEnricher")
    @patch("app.media_processing.media_processor")
    def test_extract_metadata_with_image_ocr(self, mock_media_processor, mock_enricher):
        """Test metadata extraction for images with OCR."""
        # Mock document intelligence
        mock_doc_metadata = Mock()
        mock_doc_metadata.document_type = "image"
        mock_doc_metadata.title = None
        mock_doc_metadata.author = None
        mock_doc_metadata.word_count = 0
        mock_doc_metadata.line_count = 0
        mock_doc_metadata.paragraph_count = 0
        mock_doc_metadata.topics = []
        mock_doc_metadata.keywords = []
        mock_doc_metadata.programming_language = None
        mock_doc_metadata.functions_detected = []
        mock_doc_metadata.classes_detected = []
        mock_doc_metadata.imports_detected = []
        mock_doc_metadata.content_hash = "abc123"
        mock_doc_metadata.confidence_score = 0.8

        mock_enricher.return_value.extract_metadata.return_value = mock_doc_metadata

        # Mock media processor OCR result
        mock_image_analysis = {
            "success": True,
            "metadata": {"width": 800, "height": 600, "format": "PNG", "mode": "RGB"},
            "ocr": {
                "text": "This is extracted text from image",
                "confidence": 0.95,
                "method": "easyocr",
            },
            "tables": [],
            "processing_time": 2.5,
        }
        mock_media_processor.process_image.return_value = mock_image_analysis

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = Path(tmp.name)

        try:
            # Test the function
            metadata = extract_metadata(tmp_path, ".png")

            # Verify basic metadata
            assert metadata["type"] == ".png"
            assert metadata["filename"] == tmp_path.name
            assert metadata["ext"] == ".png"

            # Verify document intelligence metadata
            assert metadata["document_type"] == "image"
            assert metadata["intelligence_confidence"] == 0.8

            # Verify OCR text was extracted
            assert "ocr_text" in metadata
            assert metadata["ocr_text"] == "This is extracted text from image"

            # Verify image analysis metadata
            assert "image_analysis" in metadata
            assert metadata["image_analysis"]["dimensions"] == "800x600"
            assert metadata["image_analysis"]["ocr_confidence"] == 0.95
            assert metadata["image_analysis"]["ocr_method"] == "easyocr"

            # Verify media processor was called
            mock_media_processor.process_image.assert_called_once_with(tmp_path)

        finally:
            tmp_path.unlink()

    @patch("app.ingest.MetadataEnricher")
    @patch("app.media_processing.media_processor")
    def test_extract_metadata_image_no_text(self, mock_media_processor, mock_enricher):
        """Test metadata extraction for images with no OCR text."""
        # Mock document intelligence
        mock_doc_metadata = Mock()
        mock_doc_metadata.document_type = "image"
        mock_doc_metadata.title = None
        mock_doc_metadata.author = None
        mock_doc_metadata.word_count = 0
        mock_doc_metadata.line_count = 0
        mock_doc_metadata.paragraph_count = 0
        mock_doc_metadata.topics = []
        mock_doc_metadata.keywords = []
        mock_doc_metadata.programming_language = None
        mock_doc_metadata.functions_detected = []
        mock_doc_metadata.classes_detected = []
        mock_doc_metadata.imports_detected = []
        mock_doc_metadata.content_hash = "abc123"
        mock_doc_metadata.confidence_score = 0.8

        mock_enricher.return_value.extract_metadata.return_value = mock_doc_metadata

        # Mock media processor with no OCR text
        mock_image_analysis = {
            "success": True,
            "metadata": {"width": 800, "height": 600, "format": "PNG", "mode": "RGB"},
            "ocr": {
                "text": "",  # No text extracted
                "confidence": 0.1,
                "method": "easyocr",
            },
            "tables": [],
            "processing_time": 1.2,
        }
        mock_media_processor.process_image.return_value = mock_image_analysis

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = Path(tmp.name)

        try:
            # Test the function
            metadata = extract_metadata(tmp_path, ".png")

            # Verify OCR text was not added (too short)
            assert "ocr_text" not in metadata

            # But image analysis metadata should still be there
            assert "image_analysis" in metadata
            assert metadata["image_analysis"]["ocr_text_length"] == 0

        finally:
            tmp_path.unlink()

    @patch("app.ingest.MetadataEnricher")
    @patch("app.media_processing.media_processor")
    @patch("pypdf.PdfReader")
    def test_extract_metadata_pdf_with_tables(
        self, mock_pdf_reader, mock_media_processor, mock_enricher
    ):
        """Test metadata extraction for PDFs with table detection."""
        # Mock document intelligence
        mock_doc_metadata = Mock()
        mock_doc_metadata.document_type = "pdf"
        mock_doc_metadata.title = "Test Document"
        mock_doc_metadata.author = "Test Author"
        mock_doc_metadata.word_count = 1000
        mock_doc_metadata.line_count = 50
        mock_doc_metadata.paragraph_count = 10
        mock_doc_metadata.topics = ["technology", "data"]
        mock_doc_metadata.keywords = ["analysis", "results"]
        mock_doc_metadata.programming_language = None
        mock_doc_metadata.functions_detected = []
        mock_doc_metadata.classes_detected = []
        mock_doc_metadata.imports_detected = []
        mock_doc_metadata.content_hash = "def456"
        mock_doc_metadata.confidence_score = 0.9

        mock_enricher.return_value.extract_metadata.return_value = mock_doc_metadata

        # Mock PDF reader
        mock_reader = Mock()
        mock_reader.metadata.title = "PDF Title"
        mock_reader.metadata.author = "PDF Author"
        mock_pdf_reader.return_value = mock_reader

        # Mock media processor table extraction
        mock_pdf_analysis = {
            "success": True,
            "tables": [
                {"table_id": 0, "shape": (5, 3), "method": "camelot"},
                {"table_id": 1, "shape": (8, 4), "method": "camelot"},
            ],
            "processing_time": 15.3,
        }
        mock_media_processor.process_pdf_enhanced.return_value = mock_pdf_analysis

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf data")
            tmp_path = Path(tmp.name)

        try:
            # Test the function
            metadata = extract_metadata(tmp_path, ".pdf")

            # Verify basic metadata
            assert metadata["type"] == ".pdf"
            assert metadata["document_type"] == "pdf"

            # Verify PDF table extraction metadata
            assert "pdf_tables" in metadata
            assert metadata["pdf_tables"]["table_count"] == 2
            assert metadata["pdf_tables"]["total_rows"] == 13  # 5 + 8
            assert metadata["pdf_tables"]["total_columns"] == 7  # 3 + 4
            assert "camelot" in metadata["pdf_tables"]["extraction_methods"]

            # Verify media processor was called
            mock_media_processor.process_pdf_enhanced.assert_called_once_with(tmp_path)

        finally:
            tmp_path.unlink()

    @patch("app.ingest.MetadataEnricher")
    @patch("app.media_processing.media_processor")
    def test_extract_metadata_processing_failure(
        self, mock_media_processor, mock_enricher
    ):
        """Test metadata extraction when media processing fails."""
        # Mock document intelligence
        mock_doc_metadata = Mock()
        mock_doc_metadata.document_type = "image"
        mock_doc_metadata.title = None
        mock_doc_metadata.author = None
        mock_doc_metadata.word_count = 0
        mock_doc_metadata.line_count = 0
        mock_doc_metadata.paragraph_count = 0
        mock_doc_metadata.topics = []
        mock_doc_metadata.keywords = []
        mock_doc_metadata.programming_language = None
        mock_doc_metadata.functions_detected = []
        mock_doc_metadata.classes_detected = []
        mock_doc_metadata.imports_detected = []
        mock_doc_metadata.content_hash = "abc123"
        mock_doc_metadata.confidence_score = 0.8

        mock_enricher.return_value.extract_metadata.return_value = mock_doc_metadata

        # Make media processor raise exception
        mock_media_processor.process_image.side_effect = Exception("Processing failed")

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = Path(tmp.name)

        try:
            # Test the function - should not crash
            metadata = extract_metadata(tmp_path, ".png")

            # Should still have basic metadata
            assert metadata["type"] == ".png"
            assert metadata["document_type"] == "image"

            # Should not have OCR or image analysis data
            assert "ocr_text" not in metadata
            assert "image_analysis" not in metadata

        finally:
            tmp_path.unlink()


class TestEnhancedDocumentLoading:
    """Test enhanced document loading with OCR integration."""

    @patch("app.ingest.extract_metadata")
    @patch("app.ingest.get_loader_for_file")
    @patch("app.ingest.validate_path")
    @patch("app.ingest.create_progress_bar")
    def test_load_documents_with_ocr_text(
        self,
        mock_progress_bar,
        mock_validate_path,
        mock_get_loader,
        mock_extract_metadata,
    ):
        """Test loading documents where images have OCR text."""
        # Setup mocks
        test_folder = Path("test_folder")
        mock_validate_path.return_value = test_folder

        # Mock progress bar
        mock_task = Mock()
        mock_progress = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_progress_bar.return_value.__enter__ = Mock(return_value=mock_progress)
        mock_progress_bar.return_value.__exit__ = Mock(return_value=None)

        # Mock folder structure with one image file
        image_file = test_folder / "test_image.png"

        with patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.return_value = [image_file]

            with patch.object(Path, "is_file", return_value=True):
                # Mock metadata extraction with OCR text
                mock_extract_metadata.return_value = {
                    "source": str(image_file),
                    "type": ".png",
                    "filename": "test_image.png",
                    "ext": ".png",
                    "ocr_text": "This is text extracted from the image via OCR",
                    "image_analysis": {"dimensions": "800x600", "ocr_confidence": 0.92},
                }

                # Mock loader to return metadata_only (for images)
                mock_get_loader.return_value = "metadata_only"

                # Test the function
                docs = load_documents("test_folder")

                # Should have one document
                assert len(docs) == 1

                # Document should have OCR text as content
                doc = docs[0]
                assert (
                    doc.page_content == "This is text extracted from the image via OCR"
                )
                assert doc.metadata["filename"] == "test_image.png"
                assert "ocr_text" in doc.metadata

    @patch("app.ingest.extract_metadata")
    @patch("app.ingest.get_loader_for_file")
    @patch("app.ingest.validate_path")
    @patch("app.ingest.create_progress_bar")
    def test_load_documents_image_no_ocr_text(
        self,
        mock_progress_bar,
        mock_validate_path,
        mock_get_loader,
        mock_extract_metadata,
    ):
        """Test loading documents where images have no meaningful OCR text."""
        # Setup mocks
        test_folder = Path("test_folder")
        mock_validate_path.return_value = test_folder

        # Mock progress bar
        mock_task = Mock()
        mock_progress = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_progress_bar.return_value.__enter__ = Mock(return_value=mock_progress)
        mock_progress_bar.return_value.__exit__ = Mock(return_value=None)

        # Mock folder structure with one image file
        image_file = test_folder / "chart.png"

        with patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.return_value = [image_file]

            with patch.object(Path, "is_file", return_value=True):
                # Mock metadata extraction with no useful OCR text
                mock_extract_metadata.return_value = {
                    "source": str(image_file),
                    "type": ".png",
                    "filename": "chart.png",
                    "ext": ".png",
                    # No 'ocr_text' key means no meaningful text was found
                    "image_analysis": {
                        "dimensions": "1200x800",
                        "ocr_confidence": 0.15,  # Low confidence
                    },
                }

                # Mock loader to return metadata_only (for images)
                mock_get_loader.return_value = "metadata_only"

                # Test the function
                docs = load_documents("test_folder")

                # Should have one document
                assert len(docs) == 1

                # Document should have empty content (no useful OCR text)
                doc = docs[0]
                assert doc.page_content == ""
                assert doc.metadata["filename"] == "chart.png"

    @patch("app.ingest.extract_metadata")
    @patch("app.ingest.get_loader_for_file")
    @patch("app.ingest.validate_path")
    @patch("app.ingest.create_progress_bar")
    def test_load_documents_mixed_content(
        self,
        mock_progress_bar,
        mock_validate_path,
        mock_get_loader,
        mock_extract_metadata,
    ):
        """Test loading mixed document types including enhanced processing."""
        # Setup mocks
        test_folder = Path("test_folder")
        mock_validate_path.return_value = test_folder

        # Mock progress bar
        mock_task = Mock()
        mock_progress = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_progress_bar.return_value.__enter__ = Mock(return_value=mock_progress)
        mock_progress_bar.return_value.__exit__ = Mock(return_value=None)

        # Mock folder structure with mixed files
        text_file = test_folder / "document.txt"
        image_file = test_folder / "scan.png"

        with patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.return_value = [text_file, image_file]

            with patch.object(Path, "is_file", return_value=True):

                def mock_metadata_side_effect(path, ext):
                    if ext == ".txt":
                        return {
                            "source": str(path),
                            "type": ".txt",
                            "filename": path.name,
                            "ext": ".txt",
                            "document_type": "text_plain",
                        }
                    elif ext == ".png":
                        return {
                            "source": str(path),
                            "type": ".png",
                            "filename": path.name,
                            "ext": ".png",
                            "ocr_text": "Scanned document text content",
                            "image_analysis": {
                                "dimensions": "2100x2970",  # A4 scan dimensions
                                "ocr_confidence": 0.88,
                            },
                        }

                mock_extract_metadata.side_effect = mock_metadata_side_effect

                def mock_loader_side_effect(path):
                    if path.suffix == ".txt":
                        # Mock text loader
                        mock_loader = Mock()
                        mock_doc = Mock()
                        mock_doc.page_content = "Original text file content"
                        mock_doc.metadata = {}
                        mock_loader.load.return_value = [mock_doc]
                        return mock_loader
                    elif path.suffix == ".png":
                        return "metadata_only"

                mock_get_loader.side_effect = mock_loader_side_effect

                # Test the function
                docs = load_documents("test_folder")

                # Should have two documents
                assert len(docs) == 2

                # Find the text and image documents
                text_doc = next(
                    d for d in docs if d.metadata["filename"] == "document.txt"
                )
                image_doc = next(
                    d for d in docs if d.metadata["filename"] == "scan.png"
                )

                # Text document should have original content
                assert text_doc.page_content == "Original text file content"

                # Image document should have OCR content
                assert image_doc.page_content == "Scanned document text content"
                assert "ocr_text" in image_doc.metadata


class TestEnhancedProcessingIntegration:
    """Integration tests for enhanced processing features."""

    def test_supported_extensions_includes_images(self):
        """Test that supported extensions include image formats."""
        # This test ensures our constants include the image formats we're processing
        from core.rag.ingest import SUPPORTED_EXTENSIONS

        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

        # Check that all image extensions are supported
        for ext in image_extensions:
            assert ext in SUPPORTED_EXTENSIONS

    def test_processing_error_resilience(self):
        """Test that processing continues even when individual files fail."""
        # This is more of a design principle test - ensuring robust error handling
        from core.rag.ingest import extract_metadata

        # Test with non-existent file - should not crash
        fake_path = Path("non_existent_file.png")

        try:
            metadata = extract_metadata(fake_path, ".png")
            # Should return basic metadata even if file doesn't exist
            assert "type" in metadata
            assert metadata["type"] == ".png"
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            pytest.fail(
                f"extract_metadata should handle missing files gracefully, but raised: {e}"
            )


# Additional fixtures specific to enhanced processing
@pytest.fixture
def mock_ocr_result():
    """Provide a mock OCR result."""
    return {
        "text": "Sample OCR extracted text content for testing purposes",
        "method": "easyocr",
        "confidence": 0.87,
        "text_blocks": [
            {"text": "Sample OCR", "confidence": 0.92},
            {"text": "extracted text", "confidence": 0.85},
            {"text": "content for testing", "confidence": 0.89},
        ],
    }


@pytest.fixture
def mock_table_result():
    """Provide a mock table extraction result."""
    return [
        {
            "table_id": 0,
            "data": [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"},
            ],
            "shape": (2, 2),
            "method": "camelot",
            "accuracy": 0.95,
        }
    ]
