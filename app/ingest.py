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
from langchain_ollama import OllamaEmbeddings
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
    Returns the appropriate document loader for a given file type.

    Args:
        file_path (Path): Path to the file.

    Returns:
        Document loader instance or None if unsupported.
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

    # Recursively get list of supported files in all subfolders
    supported_files = [
        f
        for f in folder.rglob("*")
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
    mode: str = "rebuild",  # or "increment"
):
    """
    Builds or updates a FAISS vectorstore from documents in a folder.

    Args:
        folder_path (str): Path to folder containing documents.
        persist_path (str): Path to save vectorstore.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        embedding_model (str): Ollama embedding model to use.
        mode (str): 'rebuild' to wipe and rebuild, 'increment' to update existing index.

    Returns:
        FAISS vectorstore or None if failed.
    """
    if mode == "rebuild":
        return rebuild_vectorstore(
            folder_path, persist_path, chunk_size, chunk_overlap, embedding_model
        )
    elif mode == "increment":
        return incremental_vectorstore(
            folder_path, persist_path, chunk_size, chunk_overlap, embedding_model
        )
    else:
        log_error(f"Invalid mode '{mode}'. Use 'rebuild' or 'increment'.")
        return None


@timer
def rebuild_vectorstore(
    folder_path: str,
    persist_path: str = "vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = EMBEDDING_MODEL,
) -> Optional[FAISS]:
    """
    Wipes and rebuilds the FAISS vectorstore from all documents in the folder.

    Args:
        folder_path (str): Path to folder containing documents.
        persist_path (str): Path to save vectorstore.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        embedding_model (str): Ollama embedding model to use.

    Returns:
        Optional[FAISS]: The rebuilt FAISS vectorstore, or None if failed.
    """
    try:
        ensure_directory(persist_path)

        docs = load_documents(folder_path)
        if not docs:
            log_warning("No documents loaded. Aborting ingestion.")
            return None

        docs = normalize_documents(docs)

        log_info(f"Splitting {len(docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        split_docs = dedupe_documents_by_content(split_docs)
        log_info(f"Final chunk count: {len(split_docs)}")

        embeddings = OllamaEmbeddings(model=embedding_model)

        with create_progress_bar("Generating embeddings") as progress:
            task = progress.add_task("Embedding...", total=len(split_docs))
            orig_embed_documents = embeddings.embed_documents

            def embed_documents_with_progress(texts):
                results = []
                for t in texts:
                    results.append(orig_embed_documents([t])[0])
                    progress.advance(task)
                return results

            embeddings.embed_documents = embed_documents_with_progress
            vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(persist_path)

        write_index_meta(
            persist_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        log_success(f"Rebuilt vectorstore with {len(split_docs)} chunks")
        return vectorstore

    except Exception as e:
        log_error(f"Rebuild failed: {e}")
        return None


@timer
def incremental_vectorstore(
    folder_path: str,
    persist_path: str = "vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = EMBEDDING_MODEL,
) -> Optional[FAISS]:
    """
    Incrementally updates an existing FAISS index:
    - Adds new/changed documents
    - Removes deleted documents

    Args:
        folder_path (str): Path to folder containing documents.
        persist_path (str): Path to save vectorstore.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        embedding_model (str): Ollama embedding model to use.

    Returns:
        Optional[FAISS]: The updated FAISS vectorstore, or None if failed.
    """
    try:
        ensure_directory(persist_path)

        docs = load_documents(folder_path)
        if not docs:
            log_warning("No documents loaded. Aborting ingestion.")
            return None
        docs = normalize_documents(docs)

        log_info(f"Splitting {len(docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        split_docs = dedupe_documents_by_content(split_docs)

        embeddings = OllamaEmbeddings(model=embedding_model)

        if Path(persist_path).exists():
            try:
                vectorstore = FAISS.load_local(
                    persist_path,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                )
                log_info("Loaded existing FAISS index")

                # Detect existing doc IDs
                # existing_meta = read_index_meta(persist_path)
                existing_docs = {
                    d.metadata.get("source", "")
                    for d in vectorstore.docstore._dict.values()
                }

                current_docs = {d.metadata.get("source", "") for d in split_docs}

                # Handle deletions
                deleted_docs = existing_docs - current_docs
                if deleted_docs:
                    log_info(f"Pruning {len(deleted_docs)} deleted files from index")
                    keys_to_delete = [
                        k
                        for k, v in vectorstore.docstore._dict.items()
                        if v.metadata.get("source", "") in deleted_docs
                    ]
                    for k in keys_to_delete:
                        del vectorstore.docstore._dict[k]
                        if k in vectorstore.index_to_docstore_id:
                            del vectorstore.index_to_docstore_id[k]

                # Add/update docs
                new_docs = [
                    d
                    for d in split_docs
                    if d.metadata.get("source", "") not in existing_docs
                ]
                if new_docs:
                    with create_progress_bar("Adding new chunks") as progress:
                        task = progress.add_task("Embedding...", total=len(new_docs))
                        orig_embed_documents = embeddings.embed_documents

                        def embed_documents_with_progress(texts):
                            results = []
                            for t in texts:
                                results.append(orig_embed_documents([t])[0])
                                progress.advance(task)
                            return results

                        embeddings.embed_documents = embed_documents_with_progress
                        vectorstore.add_documents(new_docs)
                    log_info(f"Added {len(new_docs)} new chunks")

            except Exception as e:
                log_warning(f"Incremental load failed ({e}); rebuilding index")
                return rebuild_vectorstore(
                    folder_path,
                    persist_path,
                    chunk_size,
                    chunk_overlap,
                    embedding_model,
                )
        else:
            log_info("No existing index found, creating new one")

            with create_progress_bar("Generating embeddings") as progress:
                task = progress.add_task("Embedding...", total=len(split_docs))
                orig_embed_documents = embeddings.embed_documents

                def embed_documents_with_progress(texts):
                    results = []
                    for t in texts:
                        results.append(orig_embed_documents([t])[0])
                        progress.advance(task)
                    return results

                embeddings.embed_documents = embed_documents_with_progress
                vectorstore = FAISS.from_documents(split_docs, embeddings)

        vectorstore.save_local(persist_path)
        write_index_meta(
            persist_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        log_success("Incremental update complete")
        return vectorstore

    except Exception as e:
        log_error(f"Incremental update failed: {e}")
        return None


def normalize_documents(docs: list[Document]) -> list[Document]:
    """
    Normalizes document content and metadata for consistency.

    Args:
        docs (list[Document]): List of documents to normalize.

    Returns:
        list[Document]: List of normalized documents.
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
    Removes duplicate documents based on their content hash.

    Args:
        docs (list[Document]): List of documents to deduplicate.

    Returns:
        list[Document]: List of deduplicated documents.
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
    Writes metadata for the FAISS index to disk.

    Args:
        persist_path (str): Path to the directory where metadata will be saved.
        embedding_model (str): Name of the embedding model used.
        chunk_size (int): Size of the document chunks.
        chunk_overlap (int): Overlap between document chunks.

    Returns:
        None
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
    Reads metadata for the FAISS index from disk.

    Args:
        persist_path (str): Path to the directory where metadata is stored.

    Returns:
        dict or None: Dictionary containing the metadata, or None if not found.
    """
    p = Path(persist_path) / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
