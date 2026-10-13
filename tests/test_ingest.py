from langchain_core.documents import Document

from src.chain import format_context
from src.ingest import split_documents


def test_split_creates_multiple_chunks_and_keeps_metadata():
    text = "RAG systems retrieve context before generating. " * 200
    doc = Document(page_content=text, metadata={"source": "t.pdf", "page": 0})
    chunks = split_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "t.pdf" for c in chunks)


def test_format_context_uses_one_based_page_numbers():
    doc = Document(page_content="hello", metadata={"source": "a.pdf", "page": 0})
    assert "[a.pdf | p.1]" in format_context([doc])
