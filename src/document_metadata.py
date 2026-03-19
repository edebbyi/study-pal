from __future__ import annotations

import re
from pathlib import Path

from src.llm_client import generate_document_metadata
from src.models import Document, DocumentMetadata


chapter_pattern = re.compile(r"\bchapter\s+\d+\b(?:[:\-\s]+[A-Za-z0-9 ,&()]+)?", re.IGNORECASE)


def _sample_document_text(document: Document, max_pages: int = 5, max_chars: int = 5000) -> str:
    sampled_pages = document.pages[:max_pages]
    excerpt = "\n\n".join(page.text for page in sampled_pages if page.text.strip())
    return excerpt[:max_chars]


def _fallback_title(document: Document) -> str:
    return Path(document.filename).stem.replace("_", " ").replace("-", " ").title()


def _fallback_topic(document: Document) -> str:
    return _fallback_title(document)


def _fallback_summary(document: Document) -> str:
    excerpt = _sample_document_text(document, max_pages=2, max_chars=240)
    if excerpt:
        return excerpt
    return f"Study material from {document.filename}."


def extract_document_metadata(document: Document) -> DocumentMetadata:
    metadata = generate_document_metadata(document.filename, _sample_document_text(document))
    if metadata is not None:
        return metadata

    return DocumentMetadata(
        document_title=_fallback_title(document),
        document_topic=_fallback_topic(document),
        document_summary=_fallback_summary(document),
    )


def detect_chapters(document: Document) -> dict[int, str]:
    chapters_by_page: dict[int, str] = {}
    current_chapter: str | None = None

    for page in document.pages:
        match = chapter_pattern.search(page.text[:1000])
        if match is not None:
            current_chapter = match.group(0).strip().title()
        chapters_by_page[page.page_number] = current_chapter or "Overview"

    return chapters_by_page
