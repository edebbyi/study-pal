"""document_metadata.py: Extract titles, topics, and summaries from documents."""

from __future__ import annotations

import re
from pathlib import Path

from src.llm.llm_client import generate_document_metadata
from src.core.models import Document, DocumentMetadata
from src.core.utils import clean_ocr_text


_CHAPTER_PATTERNS = (
    re.compile(
        r"\bchapter\s*(\d+|[ivxlcdm]+)\b(?:[:\-\s]+[A-Za-z0-9 ,&()'\".-]{0,80})?",
        re.IGNORECASE,
    ),
    # OCR variant with no space: CHAPTERVI
    re.compile(
        r"\bchapter(\d+|[ivxlcdm]+)\b(?:[:\-\s]+[A-Za-z0-9 ,&()'\".-]{0,80})?",
        re.IGNORECASE,
    ),
)



def _sample_document_text(document: Document, max_pages: int = 5, max_chars: int = 5000) -> str:
    """Create a short excerpt to summarize the document.
    
    Args:
        document (Document): Document object to analyze.
        max_pages (int): Input parameter.
        max_chars (int): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    sampled_pages = document.pages[:max_pages]  # sample early pages to keep metadata generation cheap
    excerpt = "\n\n".join(page.text for page in sampled_pages if page.text.strip())
    return excerpt[:max_chars]



def _fallback_title(document: Document) -> str:
    """Use a filename-based title when generation is unavailable.
    
    Args:
        document (Document): Document object to analyze.
    
    Returns:
        str: Formatted text result.
    """

    return Path(document.filename).stem.replace("_", " ").replace("-", " ").title()



def _fallback_topic(document: Document) -> str:
    """Use a filename-based topic when generation is unavailable.
    
    Args:
        document (Document): Document object to analyze.
    
    Returns:
        str: Formatted text result.
    """

    return _fallback_title(document)



def _fallback_summary(document: Document) -> str:
    """Build a short summary from the opening pages.
    
    Args:
        document (Document): Document object to analyze.
    
    Returns:
        str: Formatted text result.
    """

    excerpt = _sample_document_text(document, max_pages=2, max_chars=240)
    if excerpt:
        return excerpt
    return f"Study material from {document.filename}."



def extract_document_metadata(document: Document) -> DocumentMetadata:
    """Extract title, topic, and summary for a document.
    
    Args:
        document (Document): Document object to analyze.
    
    Returns:
        DocumentMetadata: Result value.
    """

    metadata = generate_document_metadata(document.filename, _sample_document_text(document))
    if metadata is not None:
        return metadata

    return DocumentMetadata(
        document_title=_fallback_title(document),
        document_topic=_fallback_topic(document),
        document_summary=_fallback_summary(document),
        key_hooks=[],
    )



def detect_chapters(document: Document) -> dict[int, str]:
    """Try to detect chapter headings by page.
    
    Args:
        document (Document): Document object to analyze.
    
    Returns:
        dict[int, str]: Mapping of computed results.
    """

    chapters_by_page: dict[int, str] = {}
    current_chapter: str | None = None

    for page in document.pages:
        # Run chapter detection on a cleaned preview to offset OCR spacing artifacts.
        preview = clean_ocr_text(page.text[:1600])
        for pattern in _CHAPTER_PATTERNS:
            match = pattern.search(preview)
            if match is not None:
                raw_heading = match.group(0).strip()
                # Keep compact heading label to stabilize chapter tags.
                current_chapter = re.sub(r"\s{2,}", " ", raw_heading).title()
                break
        chapters_by_page[page.page_number] = current_chapter or "Overview"

    return chapters_by_page
