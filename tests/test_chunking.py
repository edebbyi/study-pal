"""test_chunking.py: Tests for test_chunking.py."""

from src.data.chunking import chunk_document, split_text
from src.core.models import Document, Page


def test_split_text_respects_overlap() -> None:
    """Test split text respects overlap.
    """

    chunks = split_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_document_adds_citations() -> None:
    """Test chunk document adds citations.
    """

    document = Document(
        filename="notes.txt",
        session_id="abc123",
        source_type="txt",
        pages=[Page(page_number=1, text="Plants use light to produce glucose.")],
    )

    chunks = chunk_document(
        document,
        document_topic="Plant Biology",
        chapters_by_page={1: "Chapter 1"},
    )

    assert len(chunks) == 1
    assert chunks[0].citation == "notes.txt, Chapter 1, page 1"
    assert chunks[0].chapter == "Chapter 1"
    assert chunks[0].topic == "Plant Biology"
