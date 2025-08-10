"""
Loads and embeds documents into a FAISS vectorstore for retrieval.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def load_documents(folder_path: str):
    docs = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
        elif file_path.suffix.lower() == ".docx":
            docs.extend(Docx2txtLoader(str(file_path)).load())
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            docs.extend(UnstructuredExcelLoader(str(file_path)).load())
    return docs


def build_vectorstore(folder_path: str, persist_path: str = "vectorstore"):
    docs = load_documents(folder_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore
