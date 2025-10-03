"""
Orion Ingestion Module

Main interface for all document ingestion operations:
- Document loading and preprocessing
- Text chunking strategies
- Embedding generation
- Vector store management
- Batch processing with progress tracking
- Support for multiple file formats (PDF, TXT, DOCX, etc.)

The ingestion pipeline orchestrates the entire process from raw documents
to searchable vector embeddings stored in ChromaDB.
"""

import hashlib
import re
import shutil
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Vector store
import chromadb
from chromadb.config import Settings
from langchain.schema import Document

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.retrieval.embeddings import EmbeddingManager

# Local imports
from src.utilities.config import ChunkerType, OrionConfig
from src.utilities.utils import ensure_config, log_debug, log_error, log_info, log_warning


# ========== CUSTOM DOCUMENT LOADERS ==========
class PyMuPDFLoader:
    """Custom PyMuPDF loader for better PDF text extraction"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        """Load PDF using PyMuPDF with better text extraction"""

        documents = []

        try:
            pdf_document = fitz.open(self.file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)

                # Extract text with better spacing preservation
                text = page.get_text("text")

                # If text is empty or very short, try alternative extraction
                if not text or len(text.strip()) < 50:
                    # Try extracting text blocks for better formatting
                    blocks = page.get_text("blocks")
                    text_blocks = []
                    for block in blocks:
                        if len(block) >= 5 and block[4]:
                            text_blocks.append(block[4])
                    text = "\n".join(text_blocks)

                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"page": page_num + 1, "source": self.file_path, "extraction_method": "pymupdf"},
                    )
                    documents.append(doc)

            pdf_document.close()

        except Exception as e:
            raise Exception(f"Failed to extract text with PyMuPDF: {e}")

        return documents

# ========== SUPPORTED FILE TYPES ==========
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".txt": "Text File",
    ".json": "JSON File",
}

DOCUMENT_LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".json": TextLoader,  # Placeholder, implement JSON loader if needed
}

# ========== DATA STRUCTURES ==========
@dataclass
class IngestionStats:
    """Statistics from ingestion process"""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    processing_time: float = 0.0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        self.failed_files += 1

    def summary(self) -> dict[str, Any]:
        """Get summary dictionary"""
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "processing_time": f"{self.processing_time:.2f}s",
            "errors": self.errors,
        }


@dataclass
class DocumentMetadata:
    """Metadata for ingested documents"""

    file_path: str
    file_name: str
    file_type: str
    file_size: int
    file_hash: str
    chunk_count: int
    ingestion_timestamp: str
    source_directory: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "chunk_count": self.chunk_count,
            "ingestion_timestamp": self.ingestion_timestamp,
            "source_directory": self.source_directory,
        }


# ========== DOCUMENT PROCESSING ==========
class DocumentProcessor:
    """Handles document loading and preprocessing"""

    def __init__(self, config: OrionConfig | None = None):
        self.config = ensure_config(config)

    def load_document(self, file_path: str | Path) -> list[Document]:
        """
        Load a document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            List of loaded Document objects

        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        # Treat files containing '.txt' in their name as .txt files
        if file_ext not in SUPPORTED_EXTENSIONS and ".txt" in file_path.name.lower():
            file_ext = ".txt"

        if file_ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")

        loader_class = DOCUMENT_LOADERS[file_ext]

        try:
            loader = loader_class(str(file_path))
            documents = loader.load()

            # Add file metadata to each document
            for doc in documents:
                doc.metadata.update(
                    {
                        "source_file": str(file_path),
                        "file_name": file_path.name,
                        "file_type": SUPPORTED_EXTENSIONS[file_ext],
                        "file_extension": file_ext,
                    }
                )

            log_debug(f"Loaded {len(documents)} documents from {file_path}", self.config)
            return documents

        except Exception as e:
            log_error(f"Failed to load document {file_path}: {e}", self.config)
            raise

    def get_file_hash(self, file_path: str | Path) -> str:
        """Generate SHA-256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_file_info(self, file_path: str | Path) -> DocumentMetadata:
        """Get comprehensive file information"""
        file_path = Path(file_path)

        return DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_type=SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown"),
            file_size=file_path.stat().st_size,
            file_hash=self.get_file_hash(file_path),
            chunk_count=0,
            ingestion_timestamp="",
            source_directory=str(file_path.parent),
        )


# ========== TEXT PREPROCESSING ==========
class TextPreprocessor:
    """Handles text normalization and deduplication"""

    def __init__(self, config: OrionConfig | None = None):
        self.config = ensure_config(config)
        self.similarity_threshold = self.config.rag.preprocessing.similarity_threshold
        self.min_text_length = self.config.rag.preprocessing.min_text_length

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing format.
        Includes intelligent word boundary detection for PDF text extraction issues.

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return ""

        # Fix missing spaces
        text = self._fix_missing_spaces(text)

        # Remove excessive whitespace and normalize line endings
        text = re.sub(r"\s+", " ", text.strip())

        # Remove control characters but keep most printable characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Remove leading numbers (verse/section numbers) from the beginning of text
        text = re.sub(r"^\d+\s+", "", text)

        # Normalize common text patterns
        text = re.sub(r"\.{2,}", "...", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"\!{2,}", "!", text)

        # Remove redundant spaces around punctuation
        text = re.sub(r"\s+([\.,:;!?])", r"\1", text)
        text = re.sub(r"([\.,:;!?])\s+", r"\1 ", text)

        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        return text.strip()

    def _fix_missing_spaces(self, text: str) -> str:
        """
        Fix missing spaces in PDF-extracted text using pattern recognition.

        Args:
            text: Text with potentially missing spaces

        Returns:
            Text with spaces added where appropriate
        """
        # Basic camelCase detection (lowercase followed by uppercase)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Numbers and letters
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

        # Punctuation and letters
        text = re.sub(r"([a-zA-Z])([\.,:;!?])", r"\1 \2", text)
        text = re.sub(r"([\.,:;!?])([a-zA-Z])", r"\1 \2", text)

        # Prepositions
        text = re.sub(r"\bof([a-z]{3,})", r"of \1", text)
        text = re.sub(r"\bin([a-z]{3,})", r"in \1", text)
        text = re.sub(r"\bon([a-z]{3,})", r"on \1", text)
        text = re.sub(r"\bto([a-z]{3,})", r"to \1", text)
        text = re.sub(r"\bby([a-z]{3,})", r"by \1", text)
        text = re.sub(r"\bat([a-z]{3,})", r"at \1", text)
        text = re.sub(r"\bfor([a-z]{3,})", r"for \1", text)
        text = re.sub(r"\bwith([a-z]{3,})", r"with \1", text)
        text = re.sub(r"\bfrom([a-z]{3,})", r"from \1", text)
        text = re.sub(r"\binto([a-z]{3,})", r"into \1", text)

        # Articles/Conjunctions
        text = re.sub(r"\bthe([a-z]{3,})", r"the \1", text)
        text = re.sub(r"\band([a-z]{3,})", r"and \1", text)
        text = re.sub(r"\bbut([a-z]{3,})", r"but \1", text)
        text = re.sub(r"\bor([a-z]{3,})", r"or \1", text)

        # Word endings
        text = re.sub(r"\b(tion)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ness)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ment)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(able)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ible)([a-z]{2,})", r"\1 \2", text)

        # Clean up any double spaces created
        text = re.sub(r"\s+", " ", text)

        return text

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using sequence matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Normalize texts for comparison
        norm_text1 = self.normalize_text(text1).lower()
        norm_text2 = self.normalize_text(text2).lower()

        if not norm_text1 or not norm_text2:
            return 0.0

        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, norm_text1, norm_text2)
        return matcher.ratio()

    def is_similar_content(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar enough to be considered duplicates.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts are similar, False otherwise
        """
        similarity = self.calculate_similarity(text1, text2)
        return similarity >= self.similarity_threshold

    def is_valid_text(self, text: str) -> bool:
        """
        Check if text meets minimum quality requirements.

        Args:
            text: Text to validate

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False

        # Check minimum length
        if len(text.strip()) < self.min_text_length:
            return False

        meaningful_chars = re.sub(r"[^\w\s]", "", text)
        if len(meaningful_chars.strip()) < self.min_text_length * 0.7:
            return False

        return True

    def deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        """
        Remove duplicate and similar documents from a list.

        Args:
            documents: List of documents to deduplicate

        Returns:
            List of deduplicated documents
        """
        if not documents:
            return documents

        unique_documents = []
        seen_hashes = set()
        processed_count = 0
        duplicate_count = 0
        similar_count = 0

        log_info(f"Starting deduplication of {len(documents)} documents", self.config)

        for doc in documents:
            processed_count += 1

            # Normalize the text content
            normalized_content = self.normalize_text(doc.page_content)

            if not self.is_valid_text(normalized_content):
                log_debug("Skipping invalid text (too short or low quality)", self.config)
                continue

            # Check for exact duplicates using hash
            content_hash = hashlib.md5(normalized_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicate_count += 1
                log_debug("Skipping exact duplicate document", self.config)
                continue

            is_similar = False
            # Check against last 10 documents for performance
            for existing_doc in unique_documents[-10:]:
                if self.is_similar_content(normalized_content, existing_doc.page_content):
                    is_similar = True
                    similar_count += 1
                    log_debug(f"Skipping similar document (similarity > {self.similarity_threshold})", self.config)
                    break

            if not is_similar:
                # Update document with normalized content
                doc.page_content = normalized_content
                unique_documents.append(doc)
                seen_hashes.add(content_hash)

            if processed_count % 100 == 0:
                log_debug(f"Processed {processed_count}/{len(documents)} documents", self.config)

        removed_count = len(documents) - len(unique_documents)
        log_info(
            f"Deduplication complete: {len(unique_documents)}/{len(documents)} documents kept "
            f"(removed {duplicate_count} exact duplicates, {similar_count} similar, "
            f"{removed_count - duplicate_count - similar_count} invalid)",
            self.config,
        )

        return unique_documents

    def preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """
        Complete preprocessing pipeline: normalize and deduplicate.

        Args:
            documents: List of documents to preprocess

        Returns:
            List of preprocessed documents
        """
        if not documents:
            return documents

        log_info(f"Starting text preprocessing for {len(documents)} documents", self.config)

        # Normalize
        for doc in documents:
            doc.page_content = self.normalize_text(doc.page_content)

        # Deduplicate
        unique_documents = self.deduplicate_documents(documents)

        log_info(f"Text preprocessing complete: {len(unique_documents)} documents ready for chunking", self.config)
        return unique_documents


# ========== TEXT CHUNKING ==========
class TextChunker:
    """Handles different text chunking strategies"""

    def __init__(self, config: OrionConfig | None = None):
        self.config = ensure_config(config)
        self.chunking_config = self.config.rag.chunking

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk documents based on configured strategy.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        if self.chunking_config.strategy == ChunkerType.RECURSIVE:
            return self._chunk_recursive(documents)
        elif self.chunking_config.strategy == ChunkerType.SEMANTIC:
            return self._chunk_semantic(documents)
        elif self.chunking_config.strategy == ChunkerType.SMART:
            return self._chunk_smart(documents)
        else:
            log_warning(f"Unknown chunking strategy: {self.chunking_config.strategy}, using recursive", self.config)
            return self._chunk_recursive(documents)

    def _chunk_recursive(self, documents: list[Document]) -> list[Document]:
        """Recursive character text splitting with min_chunk_size enforcement"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "! ", "? ", ". ", " ", ""],
        )

        min_chunk_size = getattr(self.chunking_config, "min_chunk_size", 0)
        chunked_docs = []
        for doc in documents:
            chunks = splitter.split_documents([doc])

            # Merge small chunks if min_chunk_size is set
            if min_chunk_size > 0 and len(chunks) > 1:
                merged_chunks = []
                buffer = ""
                buffer_meta = None

                for i, chunk in enumerate(chunks):
                    text = chunk.page_content.strip()

                    if not buffer:
                        buffer = text
                        buffer_meta = dict(chunk.metadata)
                    else:
                        # Add space between chunks only if needed
                        if buffer and text:
                            buffer += " " + text

                    # Check if flushing buffer
                    is_last_chunk = i == len(chunks) - 1
                    buffer_size = len(buffer)

                    if buffer_size >= min_chunk_size or is_last_chunk:
                        if buffer.strip():
                            buffer_clean = self._clean_chunk_boundaries(buffer)
                            if buffer_clean:
                                new_chunk = Document(page_content=buffer_clean, metadata=buffer_meta)
                                merged_chunks.append(new_chunk)
                        buffer = ""
                        buffer_meta = None

                chunks = merged_chunks if merged_chunks else chunks
            else:
                for chunk in chunks:
                    chunk.page_content = self._clean_chunk_boundaries(chunk.page_content)

            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "chunking_strategy": "recursive",
                    }
                )

            chunked_docs.extend(chunks)

        log_debug(f"Recursive chunking: {len(documents)} docs -> {len(chunked_docs)} chunks", self.config)
        return chunked_docs

    def _clean_chunk_boundaries(self, text: str) -> str:
        """Clean chunk boundaries to start and end at word boundaries"""
        if not text or not text.strip():
            return ""

        text = text.strip()

        # Remove leading punctuation and ellipsis
        text = re.sub(r"^[^\w\s]*", "", text)
        text = re.sub(r"^\.\.\.+", "", text)

        # Remove trailing incomplete words and ellipsis
        text = re.sub(r"\s*\.\.\.+$", "", text)

        # Remove leading fragments (single or two letter words that might be incomplete)
        text = re.sub(r"^[a-z]{1,2}\b\s*", "", text)

        # If text starts with lowercase, look for the next complete word
        if text and text[0].islower():
            word_match = re.search(r"\s+(\w+)", text)
            if word_match:
                text = text[word_match.start() + 1 :].strip()

        return text.strip()

    def _chunk_semantic(self, documents: list[Document]) -> list[Document]:
        """Semantic-aware chunking (placeholder for future implementation)"""
        log_warning("Semantic chunking not yet implemented, falling back to recursive", self.config)
        return self._chunk_recursive(documents)

    def _chunk_smart(self, documents: list[Document]) -> list[Document]:
        """Smart chunking with content-aware splitting (placeholder)"""
        log_warning("Smart chunking not yet implemented, falling back to recursive", self.config)
        return self._chunk_recursive(documents)


# ========== VECTOR STORE MANAGER ==========
class VectorStoreManager:
    """Manages ChromaDB vector store operations"""

    def __init__(self, config: OrionConfig | None = None):
        self.config = ensure_config(config)
        self.vectorstore_config = self.config.rag.vectorstore
        self.embedding_config = self.config.rag.embedding
        self.embedding_manager = EmbeddingManager(self.config)

        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create data directory if it doesn't exist
            persist_dir = Path(self.vectorstore_config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False,
            )

            self.client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)

            # Get or create collection
            collection_name = "orion_knowledge_base"
            try:
                self.collection = self.client.get_collection(name=collection_name)
                log_info(f"Connected to existing ChromaDB collection: {collection_name}", self.config)
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name, metadata={"description": "Orion knowledge base collection"}
                )
                log_info(f"Created new ChromaDB collection: {collection_name}", self.config)

        except Exception as e:
            log_error(f"Failed to initialize ChromaDB: {e}", self.config)
            raise

    def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to vector store with embeddings.

        Args:
            documents: List of chunked documents to add

        Returns:
            Number of documents successfully added
        """
        if not documents:
            return 0

        try:
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, doc in enumerate(documents):
                # Generate unique ID for each chunk
                doc_id = f"{doc.metadata.get('file_name', 'unknown')}_{i}_{hash(doc.page_content) % 10000}"
                ids.append(doc_id)
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

            # Process in smaller batches to avoid BGE-M3 token limits
            batch_size = self.embedding_config.batch_size
            log_info(
                f"""
                Generating embeddings for {len(texts)} documents using
                {self.embedding_config.model} (batch size: {batch_size})
                """,
                self.config,
            )

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                log_debug(
                    f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}", self.config
                )
                batch_embeddings = self.embedding_manager.encode_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Add to ChromaDB in batches to avoid max batch size limits
            chroma_batch_size = 4096
            total_added = 0

            log_info(f"Adding {len(texts)} documents to vector store in batches of {chroma_batch_size}", self.config)

            for i in range(0, len(texts), chroma_batch_size):
                batch_end = min(i + chroma_batch_size, len(texts))
                batch_ids = ids[i:batch_end]
                batch_texts = texts[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_embeddings = all_embeddings[i:batch_end]

                log_debug(
                    f"""
                    Adding ChromaDB batch {i//chroma_batch_size + 1}/{(len(texts) + chroma_batch_size - 1)//chroma_batch_size}
                    ({len(batch_ids)} documents)
                    """,
                    self.config,
                )

                self.collection.add(
                    ids=batch_ids, documents=batch_texts, metadatas=batch_metadatas, embeddings=batch_embeddings
                )

                total_added += len(batch_ids)

            log_info(f"Successfully added {total_added} documents to vector store with custom embeddings", self.config)
            return total_added

        except Exception as e:
            log_error(f"Failed to add documents to vector store: {e}", self.config)
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "persist_directory": self.vectorstore_config.persist_directory,
            }
        except Exception as e:
            log_error(f"Failed to get collection stats: {e}", self.config)
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            collection_name = self.collection.name
            self.client.delete_collection(name=collection_name)
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"description": "Orion knowledge base collection"}
            )
            log_info("Cleared vector store collection", self.config)
            return True
        except Exception as e:
            log_error(f"Failed to clear collection: {e}", self.config)
            return False


# ========== MAIN INGESTION ORCHESTRATOR ==========
class DocumentIngestor:
    """Main orchestrator for document ingestion pipeline"""

    def __init__(self, config: OrionConfig | None = None):
        self.config = ensure_config(config)
        self.processor = DocumentProcessor(config)
        self.preprocessor = TextPreprocessor(config)
        self.chunker = TextChunker(config)
        self.vector_store = VectorStoreManager(config)

        # Setup knowledge base directory
        self.knowledge_base_path = Path(self.config.system.storage.knowledge_base_directory)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, file_path: str | Path) -> tuple[bool, DocumentMetadata, list[str]]:
        """
        Ingest a single file through the complete pipeline.

        Args:
            file_path: Path to the file to ingest

        Returns:
            Tuple of (success, metadata, errors)
        """
        file_path = Path(file_path)
        errors = []

        try:
            metadata = self.processor.get_file_info(file_path)

            # Load document
            documents = self.processor.load_document(file_path)
            metadata.total_documents = len(documents)

            # Preprocess documents (normalize and deduplicate)
            preprocessed_documents = self.preprocessor.preprocess_documents(documents)
            log_info(f"Preprocessed documents: {len(documents)} -> {len(preprocessed_documents)}", self.config)

            # Chunk preprocessed documents
            chunks = self.chunker.chunk_documents(preprocessed_documents)
            metadata.chunk_count = len(chunks)

            # Add to vector store
            added_count = self.vector_store.add_documents(chunks)
            log_debug(f"Added {added_count} chunks to vector store for {file_path.name}", self.config)

            # Update metadata with timestamp
            from datetime import datetime

            metadata.ingestion_timestamp = datetime.now().isoformat()

            log_info(f"Successfully ingested {file_path.name}: {len(chunks)} chunks", self.config)
            return True, metadata, errors

        except Exception as e:
            error_msg = f"Failed to ingest {file_path.name}: {e}"
            errors.append(error_msg)
            log_error(error_msg, self.config)
            return False, None, errors

    def ingest_directory(self, directory_path: str | Path, recursive: bool = True, max_workers: int = 4) -> IngestionStats:
        """
        Ingest all supported files from a directory.

        Args:
            directory_path: Path to directory containing files
            recursive: Whether to search subdirectories
            max_workers: Number of parallel workers for processing

        Returns:
            IngestionStats object with processing statistics
        """
        directory_path = Path(directory_path)
        stats = IngestionStats()

        if not directory_path.exists():
            stats.add_error(f"Directory not found: {directory_path}")
            return stats

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = []

        for ext in SUPPORTED_EXTENSIONS.keys():
            files = list(directory_path.glob(f"{pattern}{ext}"))
            all_files.extend(files)

        txt_like_files = []
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and ".txt" in file_path.name.lower():
                # Only add if not already found by extension matching
                if file_path not in all_files:
                    txt_like_files.append(file_path)

        all_files.extend(txt_like_files)

        stats.total_files = len(all_files)

        if stats.total_files == 0:
            log_warning(f"No supported files found in {directory_path}", self.config)
            return stats

        log_info(f"Found {stats.total_files} files to ingest from {directory_path}", self.config)

        # Process files with progress tracking
        import time

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(self.ingest_file, file_path): file_path for file_path in all_files}

            # Process results with progress bar
            with tqdm(total=len(all_files), desc="Ingesting files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]

                    try:
                        success, metadata, errors = future.result()

                        if success:
                            stats.successful_files += 1
                            stats.total_chunks += metadata.chunk_count
                            stats.total_embeddings += metadata.chunk_count
                        else:
                            stats.failed_files += 1
                            stats.errors.extend(errors)

                    except Exception as e:
                        stats.add_error(f"Unexpected error processing {file_path}: {e}")

                    pbar.update(1)

        stats.processing_time = time.time() - start_time

        log_info(f"Ingestion complete: {stats.successful_files}/{stats.total_files} files processed", self.config)
        return stats

    def ingest_knowledge_base(self, source_path: str | Path, clear_existing: bool = False) -> IngestionStats:
        """
        Ingest documents into the default knowledge base location.

        Args:
            source_path: Path to source files or directory
            clear_existing: Whether to clear existing data first

        Returns:
            IngestionStats object with processing statistics
        """
        source_path = Path(source_path)

        if clear_existing:
            log_info("Clearing existing vector store", self.config)
            self.vector_store.clear_collection()

        # Copy files to knowledge base directory if needed
        if source_path != self.knowledge_base_path:
            log_info(f"Copying files from {source_path} to {self.knowledge_base_path}", self.config)

            if source_path.is_file():
                # Single file
                dest_file = self.knowledge_base_path / source_path.name
                shutil.copy2(source_path, dest_file)
                source_path = dest_file
            else:
                # Copy entire directory recursively
                for file in source_path.rglob("*"):
                    if file.is_file():
                        dest = self.knowledge_base_path / file.relative_to(source_path)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, dest)

        # Ingest from knowledge base directory
        if source_path.is_file():
            # Single file ingestion
            stats = IngestionStats()
            stats.total_files = 1

            success, metadata, errors = self.ingest_file(source_path)
            if success:
                stats.successful_files = 1
                stats.total_chunks = metadata.chunk_count
                stats.total_embeddings = metadata.chunk_count
            else:
                stats.failed_files = 1
                stats.errors.extend(errors)
        else:
            stats = self.ingest_directory(source_path, recursive=True)

        return stats

    def get_ingestion_summary(self) -> dict[str, Any]:
        """Get summary of current ingestion state"""
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "knowledge_base_path": str(self.knowledge_base_path),
            "vector_store": vector_stats,
            "supported_formats": list(SUPPORTED_EXTENSIONS.keys()),
            "preprocessing": {
                "text_normalization": self.config.rag.preprocessing.enable_normalization,
                "deduplication": self.config.rag.preprocessing.enable_deduplication,
                "similarity_threshold": self.config.rag.preprocessing.similarity_threshold,
                "min_text_length": self.config.rag.preprocessing.min_text_length,
            },
            "chunking_strategy": self.config.rag.chunking.strategy.value,
            "chunk_size": self.config.rag.chunking.chunk_size,
            "embedding_model": self.config.rag.embedding.model,
        }


# ========== CONVENIENCE FUNCTIONS ==========
def ingest_documents(
    source_path: str | Path, config: OrionConfig | None = None, clear_existing: bool = False
) -> IngestionStats:
    """
    Convenience function to ingest documents.

    Args:
        source_path: Path to file or directory to ingest
        config: Optional Orion configuration
        clear_existing: Whether to clear existing data first

    Returns:
        IngestionStats with processing results
    """
    ingestor = DocumentIngestor(config)
    return ingestor.ingest_knowledge_base(source_path, clear_existing)


def get_supported_formats() -> dict[str, str]:
    """Get dictionary of supported file formats"""
    return SUPPORTED_EXTENSIONS.copy()


def clear_knowledge_base(config: OrionConfig | None = None) -> bool:
    """
    Clear all data from the knowledge base.

    Args:
        config: Optional Orion configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        vector_store = VectorStoreManager(config)
        return vector_store.clear_collection()
    except Exception as e:
        log_error(f"Failed to clear knowledge base: {e}", config)
        return False
