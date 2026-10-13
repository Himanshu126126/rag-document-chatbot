"""Document ingestion: PDF -> pages -> chunks -> FAISS index."""
import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it across calls."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_pdf(path: Path) -> list:
    """Load a PDF as one Document per page, tagged with the file name."""
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata["source"] = Path(path).name
    return pages


def split_documents(docs: list) -> list:
    """Split pages into overlapping chunks, keeping metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_index(pdf_paths: List[Path]) -> FAISS:
    """Create an in-memory FAISS index from one or more PDF files."""
    docs = []
    for path in pdf_paths:
        docs.extend(load_pdf(path))
    if not any(d.page_content.strip() for d in docs):
        raise ValueError("No extractable text found. Scanned PDFs are not supported.")
    chunks = split_documents(docs)
    logger.info("Indexed %d pages into %d chunks", len(docs), len(chunks))
    return FAISS.from_documents(chunks, get_embeddings())
