from __future__ import annotations

from src.config import settings
from src.models import Chunk, Document


def split_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def chunk_document(
    document: Document,
    *,
    document_id: str | None = None,
    document_title: str | None = None,
    document_summary: str | None = None,
    document_topic: str | None = None,
    chapters_by_page: dict[int, str] | None = None,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    global_chunk_counter = 0

    for page in document.pages:
        page_chunks = split_text(page.text)
        for chunk_text in page_chunks:
            chapter = chapters_by_page.get(page.page_number) if chapters_by_page else None
            citation_parts = [document.filename]
            if chapter:
                citation_parts.append(chapter)
            citation_parts.append(f"page {page.page_number}")
            all_chunks.append(
                Chunk(
                    id=f"{document.session_id}-{global_chunk_counter}",
                    text=chunk_text,
                    filename=document.filename,
                    page=page.page_number,
                    chunk_id=global_chunk_counter,
                    session_id=document.session_id,
                    citation=", ".join(citation_parts),
                    source_type=document.source_type,
                    document_id=document_id,
                    document_title=document_title,
                    document_summary=document_summary,
                    topic=document_topic,
                    chapter=chapter,
                )
            )
            global_chunk_counter += 1

    return all_chunks
