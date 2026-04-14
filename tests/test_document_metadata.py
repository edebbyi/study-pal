"""test_document_metadata.py: Tests for test_document_metadata.py."""

from src.document_metadata import detect_chapters, extract_document_metadata
from src.models import Document, Page


def test_detect_chapters_carries_forward_latest_chapter() -> None:
    """Test detect chapters carries forward latest chapter.
    """

    document = Document(
        filename="anatomy_example.pdf",
        session_id="abc123",
        source_type="pdf",
        pages=[
            Page(page_number=1, text="Chapter 1 Introduction to Anatomy"),
            Page(page_number=2, text="The brain is part of the nervous system."),
            Page(page_number=3, text="Chapter 2 Skeletal System"),
        ],
    )

    chapters = detect_chapters(document)

    assert chapters[1] == "Chapter 1 Introduction To Anatomy"
    assert chapters[2] == "Chapter 1 Introduction To Anatomy"
    assert chapters[3] == "Chapter 2 Skeletal System"


def test_extract_document_metadata_falls_back_without_model() -> None:
    """Test extract document metadata falls back without model.
    """

    document = Document(
        filename="anatomy_example.pdf",
        session_id="abc123",
        source_type="pdf",
        pages=[Page(page_number=1, text="This textbook covers anatomy and physiology.")],
    )

    metadata = extract_document_metadata(document)

    assert metadata.document_title == "Anatomy Example"
    assert metadata.document_topic == "Anatomy Example"
    assert "anatomy and physiology" in metadata.document_summary.lower()
