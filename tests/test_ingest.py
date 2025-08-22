"""
Tests for Orion's enhanced document ingestion functions.
"""

from pathlib import Path
from unittest.mock import patch
from app.ingest import get_loader_for_file, load_documents, chunk_documents


class TestGetLoaderForFile:
    """Tests for enhanced get_loader_for_file()"""

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

    def test_text_files_get_text_loader(self):
        """Should return TextLoader for various text file formats."""
        text_extensions = [".txt", ".md", ".py", ".js", ".json", ".yaml", ".html"]

        for ext in text_extensions:
            path = Path(f"test{ext}")
            with patch("app.ingest.TextLoader") as mock_loader:
                loader = get_loader_for_file(path)
                mock_loader.assert_called_once_with(str(path), autodetect_encoding=True)
                assert loader == mock_loader.return_value

    def test_image_files_return_metadata_only(self):
        """Should return 'metadata_only' for image files."""
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

        for ext in image_extensions:
            path = Path(f"test{ext}")
            loader = get_loader_for_file(path)
            assert loader == "metadata_only"

    def test_email_files_return_email_loader(self):
        """Should return 'email_loader' for email formats."""
        email_extensions = [".eml", ".msg", ".mbox"]

        for ext in email_extensions:
            path = Path(f"test{ext}")
            loader = get_loader_for_file(path)
            assert loader == "email_loader"

    def test_excel_files_return_none(self):
        """Should return None for Excel files (handled separately)."""
        excel_extensions = [".xlsx", ".xls"]

        for ext in excel_extensions:
            path = Path(f"test{ext}")
            loader = get_loader_for_file(path)
            assert loader is None

    def test_unsupported_file_returns_none(self):
        """Should return None for unsupported file types."""
        path = Path("test.xyz")
        assert get_loader_for_file(path) is None


class TestEnhancedChunking:
    """Tests for enhanced chunk_documents with semantic chunking"""

    def test_chunk_documents_uses_semantic_chunker(self):
        """Should use SemanticChunker for document chunking."""
        docs = [
            {
                "text": "# Header 1\nContent here\n## Header 2\nMore content",
                "metadata": {"source": "test.md"},
            }
        ]

        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        # Should return chunks with metadata
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert any("chunk_type" in chunk["metadata"] for chunk in chunks)

    def test_chunk_documents_fallback_on_error(self):
        """Should fallback to basic chunking if semantic chunking fails."""
        docs = [
            {
                "text": "Simple text without structure",
                "metadata": {"source": "test.txt"},
            }
        ]

        # Mock both SmartChunker and SemanticChunker to raise exceptions
        with patch("app.ingest.SmartChunker") as mock_smart_chunker, \
             patch("app.ingest.SemanticChunker") as mock_semantic_chunker:
            mock_smart_chunker.return_value.chunk_document.side_effect = Exception(
                "Smart chunking failed"
            )
            mock_semantic_chunker.return_value.chunk_document.side_effect = Exception(
                "Chunking failed"
            )

            chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)

            # Should still return chunks with fallback
            assert len(chunks) > 0
            # Should have fallback chunk type
            assert any(
                chunk["metadata"].get("chunk_type") == "fallback" for chunk in chunks
            )


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
