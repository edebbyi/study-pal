"""citations.py: Build and format citation lists for answers."""

from __future__ import annotations

import re

from src.core.models import RetrievedChunk
from src.core.utils import dedupe_preserve_order


def _normalize_citation(citation: str) -> str:
    """Normalize noisy citation strings into short, readable labels.

    Args:
        citation (str): Raw citation string captured during indexing or retrieval.

    Returns:
        str: Cleaned citation string for display.
    """
    text = " ".join(str(citation or "").split())
    if not text:
        return text

    page_match = re.search(r"(page\s+\d+)", text, flags=re.IGNORECASE)
    page_label = page_match.group(1).lower() if page_match else ""

    parts = [part.strip() for part in text.split(",") if part.strip()]
    filename = parts[0] if parts else text

    # Some PDFs leak full table-of-contents text into the chapter field.
    # If we detect repeated "Chapter" labels, keep only filename + page.
    if len(re.findall(r"\bchapter\b", text, flags=re.IGNORECASE)) > 1:
        return f"{filename}, {page_label}" if page_label else filename

    if page_label and not text.lower().endswith(page_label):
        return f"{filename}, {page_label}"

    if len(text) > 140 and page_label:
        return f"{filename}, {page_label}"
    return text



def collect_citations(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Collect a de-duplicated list of citations from retrieved chunks.
    
    Args:
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        list[str]: List of results.
    """

    return dedupe_preserve_order(_normalize_citation(chunk.citation) for chunk in retrieved_chunks)



def format_citations(citations: list[str]) -> str:
    """Format citations as a display-friendly string.
    
    Args:
        citations (list[str]): Citation strings to format or display.
    
    Returns:
        str: Formatted text result.
    """

    if not citations:
        return "Sources: none"
    return "Sources:\n" + "\n".join(f"- {citation}" for citation in citations)
