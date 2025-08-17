"""
Tests for enhanced semantic chunking.
"""

from app.chunking import SemanticChunker


class TestSemanticChunker:
    """Tests for SemanticChunker functionality"""

    def test_chunk_by_headers_markdown(self):
        """Should chunk markdown text by headers"""
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)

        text = """# Main Title
This is the introduction.

## Section 1
Content for section 1.
Some more content here.

## Section 2  
Content for section 2.
More details in this section.

### Subsection 2.1
Nested content here."""

        metadata = {"source": "test.md"}
        chunks = chunker.chunk_by_headers(text, metadata)

        # Should create multiple chunks based on headers
        assert len(chunks) > 1

        # Each chunk should have the right structure
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["chunk_type"] == "semantic_header"
            assert "header" in chunk["metadata"]

    def test_chunk_document_detects_markdown(self):
        """Should detect and use header-based chunking for markdown"""
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)

        markdown_text = """# Introduction
This is a markdown document.

## Methods
Our approach is...

## Results
We found that..."""

        metadata = {"source": "test.md"}
        chunks = chunker.chunk_document(markdown_text, metadata)

        # Should use semantic chunking
        assert len(chunks) > 0
        assert any(
            chunk["metadata"].get("chunk_type") == "semantic_header" for chunk in chunks
        )

    def test_chunk_document_fallback_to_standard(self):
        """Should fall back to standard chunking for non-structured text"""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        plain_text = "This is just plain text without any structure or headers. " * 10
        metadata = {"source": "test.txt"}

        chunks = chunker.chunk_document(plain_text, metadata)

        # Should use standard chunking
        assert len(chunks) > 0
        assert any(
            chunk["metadata"].get("chunk_type") == "standard" for chunk in chunks
        )

    def test_preserves_metadata(self):
        """Should preserve and enhance metadata in chunks"""
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)

        text = "# Header\nContent here"
        metadata = {"source": "test.md", "author": "test_user", "custom_field": "value"}

        chunks = chunker.chunk_document(text, metadata)

        # Should preserve original metadata
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.md"
            assert chunk["metadata"]["author"] == "test_user"
            assert chunk["metadata"]["custom_field"] == "value"
            assert "chunk_type" in chunk["metadata"]  # Added by chunker
