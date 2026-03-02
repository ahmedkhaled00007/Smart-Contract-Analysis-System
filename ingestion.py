"""
Ingestion Pipeline — Extract, Chunk, Embed.
Loads documents from various file formats, splits them into chunks,
generates embeddings, and stores them in a FAISS vector store.
"""

from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    FAISS_INDEX_PATH,
    logger,
)


# ── File Loader Registry ──────────────────────────────────────────────────
LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
}


def _get_loader(file_path: str):
    """Returns the appropriate document loader for the given file type."""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext, UnstructuredFileLoader)
    return loader_cls(file_path)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the configured HuggingFace embeddings model (free, runs locally)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns the configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )


# ── Public Functions ───────────────────────────────────────────────────────


def ingest_documents(file_paths: List[str]) -> Optional[FAISS]:
    """
    Loads, splits, and embeds documents into a FAISS vector store.

    Args:
        file_paths: List of file paths to ingest.

    Returns:
        FAISS vector store, or None if no files were provided.
    """
    if not file_paths:
        logger.warning("No file paths provided for ingestion.")
        return None

    all_docs: List[Document] = []
    for path in file_paths:
        try:
            loader = _get_loader(path)
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from: {Path(path).name}")
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")

    if not all_docs:
        logger.error("No documents were successfully loaded.")
        return None

    splitter = _get_text_splitter()
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Split into {len(chunks)} chunks.")

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info(f"Created FAISS index with {len(chunks)} vectors.")

    return vectorstore


def save_index(vectorstore: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    """Saves the FAISS index to disk."""
    vectorstore.save_local(path)
    logger.info(f"FAISS index saved to: {path}")


def load_index(path: str = FAISS_INDEX_PATH) -> Optional[FAISS]:
    """
    Loads an existing FAISS index from disk.

    Returns:
        FAISS vector store, or None if not found.
    """
    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS index from: {path}")
        return vectorstore
    except Exception as e:
        logger.warning(f"Could not load FAISS index from {path}: {e}")
        return None


def load_or_create_empty_index(path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Loads an existing FAISS index, or creates an empty one as fallback.
    Used by the server to ensure it can always start.
    """
    vectorstore = load_index(path)
    if vectorstore is not None:
        return vectorstore

    logger.info("Creating empty FAISS index as fallback.")
    embeddings = _get_embeddings()
    dummy_doc = Document(page_content="Empty index — no documents ingested yet.")
    return FAISS.from_documents([dummy_doc], embeddings)
