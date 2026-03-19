from __future__ import annotations

from src.models import RetrievedChunk
from src.utils import dedupe_preserve_order


def collect_citations(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    return dedupe_preserve_order(chunk.citation for chunk in retrieved_chunks)


def format_citations(citations: list[str]) -> str:
    if not citations:
        return "Sources: none"
    return "Sources:\n" + "\n".join(f"- {citation}" for citation in citations)
