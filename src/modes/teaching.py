"""teaching.py: Provide grounded answers from retrieved note chunks."""

from __future__ import annotations

from src.core.models import RetrievedChunk, TeachingResponse
from src.data.citations import collect_citations
from src.llm.llm_client import answer_from_context, generate_follow_up


def answer_question(question: str, retrieved_chunks: list[RetrievedChunk]) -> TeachingResponse:
    """Answer a question using only the retrieved note chunks.

    Args:
        question (str): The user's question.
        retrieved_chunks (list[RetrievedChunk]): Chunks used for grounding.

    Returns:
        TeachingResponse: Answer, citations, and follow-up text.
    """
    response = answer_from_context(question, retrieved_chunks)
    response.citations = collect_citations(retrieved_chunks)
    response.follow_up = generate_follow_up(
        question=question,
        retrieved_chunks=retrieved_chunks,
        answer=response.answer,
        used_fallback=response.used_fallback,
    )
    return response
