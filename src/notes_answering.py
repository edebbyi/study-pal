from __future__ import annotations

import streamlit as st

from src.citations import collect_citations
from src.models import TeachingResponse
from src.models import RetrievedChunk
from src.retrieval import retrieve_chunks
from src.teaching import answer_question


def format_answer(answer_text: str, citations: list[str]) -> str:
    if not citations:
        return answer_text
    return answer_text + "\n\nSources:\n" + "\n".join(f"- {citation}" for citation in citations)


def build_answer_response(question: str) -> TeachingResponse:
    retrieved_chunks = retrieve_chunks(
        question=question,
        chunks=st.session_state.chunks,
        session_id=st.session_state.session_id,
    )
    return answer_question(question, retrieved_chunks)


def build_answer_message(question: str) -> str:
    response = build_answer_response(question)
    return format_answer(response.answer, response.citations)


def retrieve_note_chunks(query: str) -> list[RetrievedChunk]:
    try:
        session_chunks = st.session_state.chunks
        session_id = st.session_state.session_id
    except (AttributeError, KeyError):
        session_chunks = []
        session_id = ""

    return retrieve_chunks(
        question=query,
        chunks=session_chunks,
        session_id=session_id,
    )


def get_supporting_citations(query: str) -> list[str]:
    return collect_citations(retrieve_note_chunks(query))
