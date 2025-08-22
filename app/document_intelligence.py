"""
Enhanced Document Intelligence for Orion RAG System

This module provides advanced document processing capabilities:
- Document type detection (PDF, code, markdown, text, etc.)
- Metadata enrichment (titles, authors, creation dates, topics)
- Smart chunking based on document structure
"""

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re

from app.utils import log_debug, log_warning


class DocumentType:
    """Document type constants"""

    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE_PYTHON = "python"
    CODE_JAVASCRIPT = "javascript"
    CODE_TYPESCRIPT = "typescript"
    CODE_HTML = "html"
    CODE_CSS = "css"
    CODE_JSON = "json"
    CODE_YAML = "yaml"
    CODE_XML = "xml"
    TEXT_PLAIN = "text"
    TEXT_LOG = "log"
    OFFICE_WORD = "word"
    OFFICE_EXCEL = "excel"
    OFFICE_POWERPOINT = "powerpoint"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Enhanced document metadata structure"""

    # Basic metadata
    filename: str
    filepath: str
    file_size: int
    created_date: Optional[datetime]
    modified_date: Optional[datetime]
    document_type: str

    # Content metadata
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    encoding: str = "utf-8"

    # Structure metadata
    line_count: Optional[int] = None
    word_count: Optional[int] = None
    paragraph_count: Optional[int] = None

    # Topic/content analysis
    topics: List[str] = None
    keywords: List[str] = None

    # Technical metadata (for code files)
    programming_language: Optional[str] = None
    functions_detected: List[str] = None
    classes_detected: List[str] = None
    imports_detected: List[str] = None

    # Quality indicators
    content_hash: Optional[str] = None
    confidence_score: float = 1.0

    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.keywords is None:
            self.keywords = []
        if self.functions_detected is None:
            self.functions_detected = []
        if self.classes_detected is None:
            self.classes_detected = []
        if self.imports_detected is None:
            self.imports_detected = []


class DocumentTypeDetector:
    """Detects document types based on file extensions and content analysis"""

    def __init__(self):
        self.extension_mappings = {
            # Code files
            ".py": DocumentType.CODE_PYTHON,
            ".js": DocumentType.CODE_JAVASCRIPT,
            ".ts": DocumentType.CODE_TYPESCRIPT,
            ".tsx": DocumentType.CODE_TYPESCRIPT,
            ".jsx": DocumentType.CODE_JAVASCRIPT,
            ".html": DocumentType.CODE_HTML,
            ".htm": DocumentType.CODE_HTML,
            ".css": DocumentType.CODE_CSS,
            ".scss": DocumentType.CODE_CSS,
            ".sass": DocumentType.CODE_CSS,
            ".json": DocumentType.CODE_JSON,
            ".yaml": DocumentType.CODE_YAML,
            ".yml": DocumentType.CODE_YAML,
            ".xml": DocumentType.CODE_XML,
            # Documents
            ".pdf": DocumentType.PDF,
            ".md": DocumentType.MARKDOWN,
            ".markdown": DocumentType.MARKDOWN,
            ".txt": DocumentType.TEXT_PLAIN,
            ".log": DocumentType.TEXT_LOG,
            # Office documents
            ".docx": DocumentType.OFFICE_WORD,
            ".doc": DocumentType.OFFICE_WORD,
            ".xlsx": DocumentType.OFFICE_EXCEL,
            ".xls": DocumentType.OFFICE_EXCEL,
            ".pptx": DocumentType.OFFICE_POWERPOINT,
            ".ppt": DocumentType.OFFICE_POWERPOINT,
        }

    def detect_type(self, filepath: str, content: str = None) -> str:
        """
        Detect document type based on file extension and optionally content analysis

        Args:
            filepath: Path to the document
            content: Optional content for content-based detection

        Returns:
            Document type string
        """
        # First try extension-based detection
        file_extension = Path(filepath).suffix.lower()
        if file_extension in self.extension_mappings:
            detected_type = self.extension_mappings[file_extension]
            log_debug(
                f"Detected type '{detected_type}' for {filepath} based on extension"
            )
            return detected_type

        # Fallback to content-based detection if content is provided
        if content:
            return self._detect_from_content(content, filepath)

        # Use system MIME type as fallback
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type:
            if mime_type.startswith("text/"):
                return DocumentType.TEXT_PLAIN
            elif "json" in mime_type:
                return DocumentType.CODE_JSON
            elif "xml" in mime_type:
                return DocumentType.CODE_XML

        log_warning(f"Could not detect document type for {filepath}, using 'unknown'")
        return DocumentType.UNKNOWN

    def _detect_from_content(self, content: str, filepath: str) -> str:
        """Detect document type based on content analysis"""
        content_lower = content.lower().strip()

        # Check for code patterns
        if any(
            pattern in content for pattern in ["def ", "import ", "class ", "from "]
        ):
            return DocumentType.CODE_PYTHON
        elif any(
            pattern in content for pattern in ["function ", "const ", "let ", "var "]
        ):
            return DocumentType.CODE_JAVASCRIPT
        elif content.startswith("<?xml") or "<html" in content_lower:
            return DocumentType.CODE_HTML
        elif content.startswith("{") and content.endswith("}"):
            try:
                import json

                json.loads(content)
                return DocumentType.CODE_JSON
            except Exception as e:
                log_warning(f"Failed to parse JSON from {filepath}: {e}")
                pass

        # Check for markdown patterns
        if any(
            pattern in content for pattern in ["# ", "## ", "### ", "```", "* ", "- "]
        ):
            return DocumentType.MARKDOWN

        return DocumentType.TEXT_PLAIN


class MetadataEnricher:
    """Enriches documents with comprehensive metadata"""

    def extract_metadata(self, filepath: str, content: str) -> DocumentMetadata:
        """
        Extract comprehensive metadata from a document

        Args:
            filepath: Path to the document
            content: Document content

        Returns:
            DocumentMetadata object with extracted information
        """
        # Basic file metadata
        file_path = Path(filepath)
        file_stats = file_path.stat() if file_path.exists() else None

        detector = DocumentTypeDetector()
        doc_type = detector.detect_type(filepath, content)

        metadata = DocumentMetadata(
            filename=file_path.name,
            filepath=str(file_path.absolute()),
            file_size=(
                file_stats.st_size if file_stats else len(content.encode("utf-8"))
            ),
            created_date=(
                datetime.fromtimestamp(file_stats.st_ctime) if file_stats else None
            ),
            modified_date=(
                datetime.fromtimestamp(file_stats.st_mtime) if file_stats else None
            ),
            document_type=doc_type,
            content_hash=hashlib.md5(content.encode("utf-8")).hexdigest(),
        )

        # Content analysis
        self._analyze_content_structure(content, metadata)

        # Type-specific analysis
        if doc_type == DocumentType.CODE_PYTHON:
            self._analyze_code_content(content, metadata, doc_type)
        elif doc_type == DocumentType.CODE_JAVASCRIPT:
            self._analyze_code_content(content, metadata, doc_type)
        elif doc_type == DocumentType.CODE_TYPESCRIPT:
            self._analyze_code_content(content, metadata, doc_type)
        elif doc_type == DocumentType.MARKDOWN:
            self._analyze_markdown_content(content, metadata)
        elif doc_type == DocumentType.PDF:
            self._analyze_pdf_metadata(content, metadata)

        # Extract topics and keywords
        self._extract_topics_and_keywords(content, metadata)

        log_debug(
            f"Extracted metadata for {filepath}: type={doc_type}, words={metadata.word_count}"
        )
        return metadata

    def _analyze_content_structure(self, content: str, metadata: DocumentMetadata):
        """Analyze basic content structure"""
        lines = content.split("\n")
        words = content.split()
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        metadata.line_count = len(lines)
        metadata.word_count = len(words)
        metadata.paragraph_count = len(paragraphs)

        # Try to detect title
        for line in lines:
            line = line.strip()
            if line:
                # Remove markdown heading markers
                title = re.sub(r"^#+\s*", "", line)
                # Remove common prefixes
                title = re.sub(
                    r"^(class|def|function|import)\s+", "", title, flags=re.IGNORECASE
                )
                if len(title) < 100 and title:
                    metadata.title = title
                    break

    def _analyze_code_content(
        self, content: str, metadata: DocumentMetadata, doc_type: str
    ):
        """Analyze code-specific content"""
        
        # Language-specific analysis
        if doc_type == DocumentType.CODE_PYTHON:
            metadata.programming_language = "Python"

            # Extract functions (including methods)
            function_matches = re.findall(r"^\s*def\s+(\w+)", content, re.MULTILINE)
            metadata.functions_detected = list(set(function_matches))

            # Extract classes
            class_matches = re.findall(r"^\s*class\s+(\w+)", content, re.MULTILINE)
            metadata.classes_detected = list(set(class_matches))

            # Extract imports
            import_matches = re.findall(
                r"(?:from\s+(\S+)\s+import|import\s+(\S+))", content
            )
            imports = [match[0] if match[0] else match[1] for match in import_matches]
            metadata.imports_detected = list(set(imports))

        elif doc_type in [DocumentType.CODE_JAVASCRIPT, DocumentType.CODE_TYPESCRIPT]:
            metadata.programming_language = (
                "JavaScript"
                if doc_type == DocumentType.CODE_JAVASCRIPT
                else "TypeScript"
            )

            # Extract functions
            function_matches = re.findall(
                r"function\s+(\w+)|const\s+(\w+)\s*=.*?=>", content
            )
            functions = [
                match[0] if match[0] else match[1]
                for match in function_matches
                if any(match)
            ]
            metadata.functions_detected = list(set(functions))

            # Extract imports
            import_matches = re.findall(r'import.*?from\s+[\'"]([^\'"]+)[\'"]', content)
            metadata.imports_detected = list(set(import_matches))

    def _analyze_markdown_content(self, content: str, metadata: DocumentMetadata):
        """Analyze markdown-specific content"""
        # Extract headings as potential topics
        heading_matches = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        if heading_matches:
            metadata.topics = list(
                set(heading_matches[:10])
            )  # Top 10 headings as topics

        # Extract links for reference tracking
        link_matches = re.findall(r"\[([^\]]+)\]\([^\)]+\)", content)
        if len(link_matches) > 5:  # Lots of links suggests reference document
            metadata.topics.append("reference_document")

    def _analyze_pdf_metadata(self, content: str, metadata: DocumentMetadata):
        """Analyze PDF-specific metadata (if content is extracted text)"""

        # Look for common PDF patterns
        if re.search(r"page\s+\d+", content.lower()):
            metadata.topics.append("multi_page_document")

        # Look for academic paper patterns
        if any(
            term in content.lower()
            for term in ["abstract", "references", "bibliography", "doi:"]
        ):
            metadata.topics.append("academic_paper")

    def _extract_topics_and_keywords(self, content: str, metadata: DocumentMetadata):
        """Extract topics and keywords using simple heuristics"""
        # This is a simplified version - in production you might use NLP libraries

        # Common technical keywords
        technical_keywords = [
            "api",
            "database",
            "server",
            "client",
            "authentication",
            "authorization",
            "machine learning",
            "neural network",
            "algorithm",
            "data structure",
            "microservice",
            "container",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "react",
            "vue",
            "angular",
            "node.js",
            "python",
            "javascript",
            "typescript",
        ]

        content_lower = content.lower()
        found_keywords = [kw for kw in technical_keywords if kw in content_lower]
        metadata.keywords = found_keywords[:10]  # Limit to top 10

        # Extract topics based on repeated terms (simple frequency analysis)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content_lower)  # Words with 4+ chars
        word_freq = {}
        for word in words:
            if word not in [
                "this",
                "that",
                "with",
                "have",
                "will",
                "from",
                "they",
                "been",
                "their",
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get most frequent words as topics
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_words = [word for word, freq in frequent_words if freq > 2]
        metadata.topics.extend(topic_words)

        # Remove duplicates
        metadata.topics = list(set(metadata.topics))
        metadata.keywords = list(set(metadata.keywords))


class SmartChunker:
    """Context-aware document chunking based on document structure"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self, content: str, metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """
        Create smart chunks based on document type and structure

        Args:
            content: Document content
            metadata: Document metadata with type information

        Returns:
            List of chunk dictionaries with content and metadata
        """
        doc_type = metadata.document_type

        # Type-specific chunking strategies
        if doc_type == DocumentType.MARKDOWN:
            return self._chunk_markdown(content, metadata)
        elif doc_type in [
            DocumentType.CODE_PYTHON,
            DocumentType.CODE_JAVASCRIPT,
            DocumentType.CODE_TYPESCRIPT,
        ]:
            return self._chunk_code(content, metadata)
        elif doc_type == DocumentType.PDF:
            return self._chunk_pdf_text(content, metadata)
        else:
            return self._chunk_generic_text(content, metadata)

    def _chunk_markdown(
        self, content: str, metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Chunk markdown documents by sections and headings"""
        chunks = []

        # Split by headings
        sections = re.split(r"\n(?=#)", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Extract heading for context
            heading_match = re.match(r"^(#+)\s*(.+)", section)
            heading = heading_match.group(2) if heading_match else f"Section {i+1}"

            # If section is too long, split it further
            if len(section) > self.chunk_size:
                sub_chunks = self._split_long_text(
                    section, self.chunk_size, self.overlap
                )
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append(
                        {
                            "text": sub_chunk,
                            "metadata": {
                                **metadata.__dict__,
                                "chunk_index": len(chunks),
                                "chunk_type": "markdown_section",
                                "section_heading": heading,
                                "subsection": (
                                    f"{heading} - Part {j+1}"
                                    if len(sub_chunks) > 1
                                    else heading
                                ),
                            },
                        }
                    )
            else:
                chunks.append(
                    {
                        "text": section,
                        "metadata": {
                            **metadata.__dict__,
                            "chunk_index": len(chunks),
                            "chunk_type": "markdown_section",
                            "section_heading": heading,
                        },
                    }
                )

        log_debug(f"Created {len(chunks)} markdown chunks for {metadata.filename}")
        return chunks

    def _chunk_code(
        self, content: str, metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Chunk code documents by functions, classes, and logical blocks"""
        chunks = []

        if metadata.programming_language == "Python":
            # Split by functions and classes
            parts = re.split(r"\n(?=(?:def |class |import |from ))", content)
        else:
            # Generic code splitting
            parts = re.split(
                r"\n(?=(?:function |class |const |let |var |import ))", content
            )

        current_chunk = ""
        for part in parts:
            if not part.strip():
                continue

            # If adding this part would exceed chunk size, start a new chunk
            if current_chunk and len(current_chunk + part) > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        self._create_code_chunk(current_chunk, metadata, len(chunks))
                    )
                current_chunk = part
            else:
                current_chunk += part

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_code_chunk(current_chunk, metadata, len(chunks)))

        log_debug(f"Created {len(chunks)} code chunks for {metadata.filename}")
        return chunks

    def _create_code_chunk(
        self, content: str, metadata: DocumentMetadata, index: int
    ) -> Dict[str, Any]:
        """Create a code chunk with appropriate metadata"""
        # Detect what's in this chunk
        chunk_type = "code_block"
        if content.strip().startswith(("def ", "function ")):
            chunk_type = "function"
        elif content.strip().startswith(("class ", "class:")):
            chunk_type = "class"
        elif any(
            content.strip().startswith(imp)
            for imp in ["import ", "from ", "const ", "let ", "var "]
        ):
            chunk_type = "declarations"

        return {
            "text": content,
            "metadata": {
                **metadata.__dict__,
                "chunk_index": index,
                "chunk_type": chunk_type,
                "programming_language": metadata.programming_language,
            },
        }

    def _chunk_pdf_text(
        self, content: str, metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Chunk PDF extracted text, preserving paragraphs when possible"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, start new chunk
            if (
                current_chunk
                and len(current_chunk + "\n\n" + paragraph) > self.chunk_size
            ):
                chunks.append(
                    {
                        "text": current_chunk,
                        "metadata": {
                            **metadata.__dict__,
                            "chunk_index": len(chunks),
                            "chunk_type": "pdf_section",
                        },
                    }
                )
                current_chunk = paragraph
            else:
                current_chunk = (
                    (current_chunk + "\n\n" + paragraph) if current_chunk else paragraph
                )

        # Add final chunk
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk,
                    "metadata": {
                        **metadata.__dict__,
                        "chunk_index": len(chunks),
                        "chunk_type": "pdf_section",
                    },
                }
            )

        return chunks

    def _chunk_generic_text(
        self, content: str, metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Generic text chunking with overlap"""
        chunks = []
        text_parts = self._split_long_text(content, self.chunk_size, self.overlap)

        for i, part in enumerate(text_parts):
            chunks.append(
                {
                    "text": part,
                    "metadata": {
                        **metadata.__dict__,
                        "chunk_index": i,
                        "chunk_type": "text_chunk",
                    },
                }
            )

        return chunks

    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split long text into chunks with overlap"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in ".!?\n":
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks
