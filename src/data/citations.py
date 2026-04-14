"""citations.py: Build and format citation lists for answers."""

from __future__ import annotations

from src.core.models import RetrievedChunk
from src.core.utils import dedupe_preserve_order



def collect_citations(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Collect a de-duplicated list of citations from retrieved chunks.
    
    Args:
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        list[str]: List of results.
    """

    return dedupe_preserve_order(chunk.citation for chunk in retrieved_chunks)  # keep top hits first



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
