"""test_citations.py: Tests for test_citations.py."""

from src.data.citations import collect_citations, format_citations
from src.core.models import RetrievedChunk


def test_collect_citations_dedupes_in_order() -> None:
    """Test collect citations dedupes in order.
    """

    chunks = [
        RetrievedChunk(text="a", filename="one.pdf", page=1, citation="one.pdf, page 1", score=0.9, chunk_id=1),
        RetrievedChunk(text="b", filename="one.pdf", page=1, citation="one.pdf, page 1", score=0.8, chunk_id=2),
        RetrievedChunk(text="c", filename="two.pdf", page=2, citation="two.pdf, page 2", score=0.7, chunk_id=3),
    ]

    assert collect_citations(chunks) == ["one.pdf, page 1", "two.pdf, page 2"]


def test_format_citations_returns_bulleted_block() -> None:
    """Test format citations returns bulleted block.
    """

    formatted = format_citations(["one.pdf, page 1"])
    assert "Sources:" in formatted
    assert "- one.pdf, page 1" in formatted


def test_collect_citations_collapses_toc_heavy_chapter_labels() -> None:
    """Ensure noisy table-of-contents chapter labels are shortened for display."""
    noisy = (
        "anatomy_example.pdf, Chapter 1 Brain Basics 10 Chapter 2 Senses & Perception 18 "
        "Chapter 3 Movement 26 Contents, page 38"
    )
    chunks = [
        RetrievedChunk(
            text="a",
            filename="anatomy_example.pdf",
            page=38,
            citation=noisy,
            score=0.9,
            chunk_id=1,
        )
    ]

    assert collect_citations(chunks) == ["anatomy_example.pdf, page 38"]
