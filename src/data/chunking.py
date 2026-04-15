"""chunking.py: Split documents into clean, sized chunks."""

from __future__ import annotations

from src.core.config import settings
from src.core.models import Chunk, Document


def split_text(
    text: str,
    chunk_size: int | None = None,

    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks for retrieval.
    
    Args:
        text (str): Input text to process.
        chunk_size (int | None): Input parameter.
        overlap (int | None): Input parameter.
    
    Returns:
        list[str]: List of results.
    """

    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    min_chunk = max(80, int(chunk_size * 0.6))  # avoid tiny fragments when adjusting to whitespace

    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length and not text[end - 1].isspace():
            window_start = min(text_length, start + min_chunk)  # don't backtrack too early in the chunk
            if window_start < end:
                boundary = text.rfind(" ", window_start, end)
                if boundary > start:
                    end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, 0)
    return chunks


def chunk_document(
    document: Document,
    *,
    document_id: str | None = None,
    document_title: str | None = None,
    document_summary: str | None = None,
    document_topic: str | None = None,
    user_id: str | None = None,

    chapters_by_page: dict[int, str] | None = None,
) -> list[Chunk]:
    """Break a document into chunk objects with citations.
    
    Args:
        document (Document): Document object to analyze.
        document_id (str | None): Document identifier for the current workspace.
        document_title (str | None): Input parameter.
        document_summary (str | None): Input parameter.
        document_topic (str | None): Input parameter.
        user_id (str | None): Input parameter.
        chapters_by_page (dict[int, str] | None): Input parameter.
    
    Returns:
        list[Chunk]: List of results.
    """

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
                    user_id=user_id,
                )
            )
            global_chunk_counter += 1

    return all_chunks
