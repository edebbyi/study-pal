from __future__ import annotations

from src.citations import collect_citations
from src.llm_client import answer_from_context
from src.models import RetrievedChunk, TeachingResponse


def answer_question(question: str, retrieved_chunks: list[RetrievedChunk]) -> TeachingResponse:
    response = answer_from_context(question, retrieved_chunks)
    response.citations = collect_citations(retrieved_chunks)
    return response
