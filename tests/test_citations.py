from src.citations import collect_citations, format_citations
from src.models import RetrievedChunk


def test_collect_citations_dedupes_in_order() -> None:
    chunks = [
        RetrievedChunk(text="a", filename="one.pdf", page=1, citation="one.pdf, page 1", score=0.9, chunk_id=1),
        RetrievedChunk(text="b", filename="one.pdf", page=1, citation="one.pdf, page 1", score=0.8, chunk_id=2),
        RetrievedChunk(text="c", filename="two.pdf", page=2, citation="two.pdf, page 2", score=0.7, chunk_id=3),
    ]

    assert collect_citations(chunks) == ["one.pdf, page 1", "two.pdf, page 2"]


def test_format_citations_returns_bulleted_block() -> None:
    formatted = format_citations(["one.pdf, page 1"])
    assert "Sources:" in formatted
    assert "- one.pdf, page 1" in formatted
