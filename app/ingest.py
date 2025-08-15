"""
Loads and embeds documents into a FAISS vectorstore for retrieval.
"""

import json
import time
import pandas as pd
from hashlib import md5
from pathlib import Path
from typing import List, Optional
from app.utils import (
    log_info,
    log_success,
    log_warning,
    log_error,
    timer,
    validate_path,
    ensure_directory,
    create_progress_bar,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

EMBEDDING_MODEL = "nomic-embed-text"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt"}


def load_excel_with_pandas(file_path: Path) -> list[Document]:
    """
    Pure-Python Excel loader using pandas; returns one Document per sheet.

    Args:
        file_path: Path to the Excel file to load

    Returns:
        List of Document objects, one per sheet in the Excel file
    """
    docs = []
    try:
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)  # all sheets
        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            # compact textual table (headers + first N rows)
            text = df.to_csv(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "type": "excel",
                    },
                )
            )
    except Exception as e:
        log_error(f"Excel load failed for {file_path.name}: {e}")
    return docs


def get_loader_for_file(file_path: Path):
    """
    Returns appropriate document loader for file type.

    Args:
        file_path: Path to the file

    Returns:
        Document loader instance or None if unsupported
    """
    suffix = file_path.suffix.lower()

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        try:
            return loader
        except Exception as e:
            log_warning(f"Standard PDF load failed for {file_path.name}: {e}")
            log_info(
                "TODO: OCR fallback (e.g., pytesseract + pdf2image) not yet implemented."
            )
            # Example placeholder:
            # from pdf2image import convert_from_path
            # import pytesseract
            # pages = convert_from_path(str(file_path))
            # text = "\n".join(pytesseract.image_to_string(p) for p in pages)
            # return [Document(page_content=text, metadata={"source": str(file_path), "type": "pdf-ocr"})]
            return None
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path))
    if suffix in {".xlsx", ".xls"}:
        return None  # special-case Excel
    if suffix == ".txt":
        return TextLoader(str(file_path), autodetect_encoding=True)
    return None


def load_documents(folder_path: str) -> List[Document]:
    """
    Loads documents from the given folder with improved error handling and progress tracking.
    Supported formats: PDF, DOCX, Excel, TXT.

    Args:
        folder_path: Path to folder containing documents

    Returns:
        List of loaded documents
    """
    try:
        folder = validate_path(folder_path, must_exist=True)
    except (FileNotFoundError, NotADirectoryError) as e:
        log_error(str(e))
        return []

    # Get list of supported files
    supported_files = [
        f
        for f in folder.glob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        log_warning(f"No supported files found in '{folder_path}'")
        return []

    log_info(f"Found {len(supported_files)} supported files")

    docs = []
    failed_files = []

    with create_progress_bar("Loading documents") as progress:
        task = progress.add_task("Loading...", total=len(supported_files))

        for file_path in supported_files:
            try:
                if file_path.suffix.lower() in {".xlsx", ".xls"}:
                    file_docs = load_excel_with_pandas(file_path)
                    docs.extend(file_docs)
                    log_info(f"✅ Loaded {file_path.name} ({len(file_docs)} sheets)")
                else:
                    loader = get_loader_for_file(file_path)
                    if loader:
                        file_docs = loader.load()
                        docs.extend(file_docs)
                        log_info(
                            f"✅ Loaded {file_path.name} ({len(file_docs)} chunks)"
                        )
                    else:
                        log_warning(f"No loader available for {file_path.name}")

            except Exception as e:
                log_error(f"Failed to load {file_path.name}: {e}")
                failed_files.append(file_path.name)

            progress.advance(task)

    if failed_files:
        log_warning(
            f"Failed to load {len(failed_files)} files: {', '.join(failed_files)}"
        )

    log_success(
        f"Successfully loaded {len(docs)} document chunks from {len(supported_files) - len(failed_files)} files"
    )
    return docs


@timer
def build_vectorstore(
    folder_path: str,
    persist_path: str = "vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = EMBEDDING_MODEL,
    mode: str = "rebuild",  # or "append"
) -> Optional[FAISS]:
    """
    Splits documents, creates embeddings, and saves FAISS index locally.

    Args:
        folder_path: Path to folder containing documents
        persist_path: Path to save vectorstore
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        embedding_model: Ollama embedding model to use
        mode: Mode for building the vectorstore ("rebuild" or "append")

    Returns:
        FAISS vectorstore or None if failed
    """
    try:
        # Validate inputs
        ensure_directory(persist_path)

        # Load documents
        docs = load_documents(folder_path)
        if not docs:
            log_warning("No documents loaded. Aborting ingestion.")
            return None

        # Normalize document content and metadata
        docs = normalize_documents(docs)

        log_info(
            f"Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})..."
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        log_info(f"Created {len(split_docs)} text chunks")

        if not split_docs:
            log_error("No text chunks created after splitting")
            return None

        # Deduplicate documents
        split_docs = dedupe_documents_by_content(split_docs)
        log_info(f"After dedupe: {len(split_docs)} unique chunks")

        log_info(f"Generating embeddings with '{embedding_model}'...")
        try:
            embeddings = OllamaEmbeddings(model=embedding_model)

            if mode == "append" and Path(persist_path).exists():
                from langchain_community.vectorstores import FAISS

                existing_meta = read_index_meta(persist_path)
                if (
                    existing_meta
                    and existing_meta.get("embedding_model") != embedding_model
                ):
                    log_warning(
                        f"Embedding model mismatch (existing={existing_meta.get('embedding_model')} "
                        f"vs new={embedding_model}); falling back to rebuild."
                    )
                    mode = "rebuild"

            if mode == "append" and Path(persist_path).exists():
                try:
                    vectorstore = FAISS.load_local(
                        persist_path,
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    vectorstore.add_documents(split_docs)
                    log_info("Appended new chunks to existing index")
                except Exception as e:
                    log_warning(f"Append failed ({e}); rebuilding index")
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
            else:
                vectorstore = FAISS.from_documents(split_docs, embeddings)

            vectorstore.save_local(persist_path)
            write_index_meta(
                persist_path,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            log_success(
                f"Vectorstore with {len(split_docs)} chunks saved to '{persist_path}'"
            )
            return vectorstore

        except Exception as e:
            log_error(f"Failed to initialize embeddings: {e}")
            log_info("Make sure Ollama is running and the model is available")
            return None

    except Exception as e:
        log_error(f"Vectorstore creation failed: {e}")
        return None


def normalize_documents(docs: list[Document]) -> list[Document]:
    """
    Normalize document content and metadata.

    Args:
        docs: List of documents to normalize.

    Returns:
        List of normalized documents.
    """
    norm = []
    for d in docs:
        content = (d.page_content or "").strip()
        if not content:
            continue
        meta = dict(d.metadata or {})
        meta.setdefault("source", meta.get("file_path", "unknown"))
        # Carry through page/sheet if present
        if "page" in meta:
            meta["page"] = meta["page"]
        if "sheet" in meta:
            meta["sheet"] = meta["sheet"]
        norm.append(Document(page_content=content, metadata=meta))
    return norm


def dedupe_documents_by_content(docs: list[Document]) -> list[Document]:
    """
    Remove duplicate documents based on their content.

    Args:
        docs: List of documents to deduplicate.

    Returns:
        List of deduplicated documents.
    """
    seen = set()
    out = []
    for d in docs:
        h = md5(d.page_content.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out


def write_index_meta(
    persist_path: str, *, embedding_model: str, chunk_size: int, chunk_overlap: int
):
    """
    Write metadata for the index.

    Args:
        persist_path: Path to the directory where metadata will be saved.
        embedding_model: Name of the embedding model used.
        chunk_size: Size of the document chunks.
        chunk_overlap: Overlap between document chunks.
    """
    meta = {
        "created_at": time.time(),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    with open(Path(persist_path) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_index_meta(persist_path: str) -> Optional[dict]:
    """
    Read metadata for the index.

    Args:
        persist_path: Path to the directory where metadata is stored.

    Returns:
        Dictionary containing the metadata, or None if not found.
    """
    p = Path(persist_path) / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
