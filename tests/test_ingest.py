"""
Tests for Orion's document ingestion functions.
"""

from pathlib import Path
from unittest.mock import patch
from app.ingest import get_loader_for_file, load_documents


class TestGetLoaderForFile:
    """Tests for get_loader_for_file()"""

    def test_pdf_loader_created(self):
        """Should return a PyPDFLoader for .pdf files."""
        pdf_path = Path("test.pdf")
        with patch("app.ingest.PyPDFLoader") as mock_loader:
            loader = get_loader_for_file(pdf_path)
            mock_loader.assert_called_once_with("test.pdf")
            assert loader == mock_loader.return_value

    def test_docx_loader_created(self):
        """Should return a Docx2txtLoader for .docx files."""
        docx_path = Path("test.docx")
        with patch("app.ingest.Docx2txtLoader") as mock_loader:
            loader = get_loader_for_file(docx_path)
            mock_loader.assert_called_once_with("test.docx")
            assert loader == mock_loader.return_value

    def test_unsupported_file_returns_none(self):
        """Should return None for unsupported file types."""
        path = Path("test.xyz")
        assert get_loader_for_file(path) is None


class TestLoadDocuments:
    """Tests for load_documents()"""

    def test_empty_folder_returns_empty_list(self, tmp_path):
        """Should return empty list when no documents exist."""
        docs = load_documents(tmp_path)
        assert docs == []

    def test_load_mixed_files(self, tmp_path, fake_docs):
        """Should load only supported files from folder."""
        # Create fake files
        (tmp_path / "file1.pdf").write_text("Fake PDF")
        (tmp_path / "file2.xyz").write_text("Unsupported")
        with patch("app.ingest.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            docs = load_documents(tmp_path)
            assert docs == fake_docs
            mock_loader.assert_called_once()
