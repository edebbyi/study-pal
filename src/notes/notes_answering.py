"""notes_answering.py: Build grounded answers and structured responses from notes."""

from __future__ import annotations

import re

import streamlit as st

from src.modes.mode_router import extract_conversation_topic

from src.core.models import RetrievedChunk, StructuredAnswer, TeachingResponse
from src.data.citations import collect_citations
from src.data.retrieval import retrieve_chunks
from src.llm.llm_client import generate_structured_answer
from src.modes.teaching import answer_question


def _is_notes_miss(answer_text: str) -> bool:
    """Check whether the response matches known "not in notes" phrases.

    Args:
        answer_text (str): The model response to evaluate.

    Returns:
        bool: True if the response is a notes-miss fallback.
    """
    cleaned = " ".join(answer_text.strip().lower().split())
    fallback_phrases = {
        "i couldn't find that in the notes yet.",
        "i couldn't find that in the notes yet",
        "i couldn't find relevant support in the uploaded notes for that question yet.",
        "i couldn't find relevant support in the uploaded notes for that question yet",
        "i could not find that in the notes yet.",
        "i could not find that in the notes yet",
    }
    return cleaned in fallback_phrases


def _summarize_from_chunk(text: str) -> str:
    """Summarize a chunk into one or two sentences for display.

    Args:
        text (str): Raw chunk text from the notes.

    Returns:
        str: A cleaned, short summary sentence or two.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:2]).strip()
    summary = _clean_leading_fragment(summary or cleaned[:240].rstrip())
    return summary


def _clean_leading_fragment(text: str) -> str:
    """Remove stray punctuation or fragments at the start of a summary.

    Args:
        text (str): Summary text to clean.

    Returns:
        str: A cleaned summary that starts at a readable boundary.
    """
    if not text:
        return text
    stripped = text.lstrip(" \u2014-")
    if not stripped:
        return text
    if stripped[0].isupper() or stripped[0].isdigit() or stripped[0] in ('"', "“", "("):
        return stripped
    match = re.search(r"\b[A-Z]", stripped)
    if match and match.start() > 0:
        return stripped[match.start():].lstrip()
    return stripped


def _extract_query_terms(question: str) -> list[str]:
    """Return non-trivial tokens from a question for scoring relevance.

    Args:
        question (str): The user query to tokenize.

    Returns:
        list[str]: Filtered tokens used for overlap scoring.
    """
    cleaned = re.sub(r"[^\w\s]", " ", question.lower())
    tokens = [token for token in cleaned.split() if token]
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "about",
        "more",
        "tell",
        "me",
        "why",
        "how",
        "what",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "explain",
        "describe",
        "detail",
        "details",
    }
    terms = [token for token in tokens if token not in stopwords and len(token) > 2]
    return terms


def _select_summary_chunk(question: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Pick the chunk with the highest term overlap to summarize.

    Args:
        question (str): The original user question.
        chunks (list[RetrievedChunk]): Retrieved chunks to score.

    Returns:
        RetrievedChunk | None: The best chunk or None if none exist.
    """
    if not chunks:
        return None
    terms = _extract_query_terms(question)
    if not terms:
        return chunks[0]

    best_chunk = chunks[0]
    best_score = -1
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for term in terms if term in text)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def _is_followup_question(question: str) -> bool:
    """Return True for short follow-ups like "I don't get it" or "why?".

    Args:
        question (str): The user's latest message.

    Returns:
        bool: True when the message should be expanded with context.
    """
    normalized = " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())
    followup_phrases = {
        "i dont get it",
        "i don't get it",
        "dont get it",
        "don't get it",
        "im confused",
        "i'm confused",
        "confused",
        "what do you mean",
        "can you explain",
        "can you clarify",
        "clarify",
        "explain that",
        "explain this",
        "say more",
        "tell me more",
        "go deeper",
        "why is that",
        "how so",
    }
    if not normalized:
        return False
    if normalized in followup_phrases:
        return True
    if len(normalized.split()) <= 3 and normalized in {"why", "how", "what", "huh"}:
        return True
    return False


def _resolve_followup_query(question: str) -> str:
    """Return a contextualized query when the user asks a follow-up.

    Args:
        question (str): The user's latest message.

    Returns:
        str: The original or expanded query for retrieval.
    """
    if not _is_followup_question(question):
        return question

    explicit_topic = extract_conversation_topic(question, fallback_topic=None)
    messages = list(st.session_state.messages)
    topic_subject = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            topic_subject = message.get("topic_subject") or message.get("topic")
            if isinstance(topic_subject, str) and topic_subject.strip():
                break

    if explicit_topic and isinstance(explicit_topic, str):
        cleaned_topic = explicit_topic.strip()
        vague_topics = {
            "i dont get it",
            "i don't get it",
            "dont get it",
            "don't get it",
            "im confused",
            "i'm confused",
            "confused",
            "explain this",
            "explain that",
            "clarify",
            "help",
        }
        if cleaned_topic and cleaned_topic.lower() not in vague_topics:  # Preserve explicit topic shifts.
            if not topic_subject or cleaned_topic.lower() != topic_subject.strip().lower():
                return question

    last_user_question = None
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip() and content.strip() != question.strip():
                last_user_question = content.strip()
                break

    candidates = [item for item in [topic_subject, last_user_question] if item]
    if not candidates:
        return question
    context_hint = " ".join(candidates).strip()
    if len(context_hint) > 200:
        context_hint = context_hint[:200].rstrip()
    return f"{context_hint}. {question}"


def _truncate_message(text: str, limit: int = 360) -> str:
    """Trim a message for inclusion in prompt history.

    Args:
        text (str): Message content to shorten.
        limit (int): Maximum character length to keep.

    Returns:
        str: Shortened text with ellipsis if needed.
    """
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _build_last_turn_history() -> str:
    """Return a two-line history snippet with the last user and assistant turn.

    Returns:
        str: Formatted history or an empty string if unavailable.
    """
    messages = list(st.session_state.messages)
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return ""

    last_user_idx = None
    for idx in range(last_assistant_idx - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break

    parts: list[str] = []
    if last_user_idx is not None:
        content = messages[last_user_idx].get("content")
        if isinstance(content, str) and content.strip():
            parts.append(f"User: {_truncate_message(content)}")
    assistant_content = messages[last_assistant_idx].get("content")
    if isinstance(assistant_content, str) and assistant_content.strip():
        parts.append(f"Assistant: {_truncate_message(assistant_content)}")
    return "\n".join(parts)


def format_answer(answer_text: str, citations: list[str]) -> str:
    """Append a Sources section to the answer when citations exist.

    Args:
        answer_text (str): The answer to display.
        citations (list[str]): Citations to append.

    Returns:
        str: Answer text with formatted citations if provided.
    """
    if not citations:
        return answer_text
    return answer_text + "\n\nSources:\n" + "\n".join(f"- {citation}" for citation in citations)


def build_answer_response(question: str) -> TeachingResponse:
    """Return a grounded answer using retrieved note chunks.

    Args:
        question (str): The user's question.

    Returns:
        TeachingResponse: Answer and citation payload.
    """
    retrieval_query = _resolve_followup_query(question)
    retrieved_chunks = retrieve_chunks(
        question=retrieval_query,
        chunks=st.session_state.chunks,
        session_id=st.session_state.session_id,
        document_id=st.session_state.active_document_id,
        user_id=st.session_state.user_id,
    )
    return answer_question(question, retrieved_chunks)


def build_answer_message(question: str) -> str:
    """Return the answer plus formatted citations for UI display.

    Args:
        question (str): The user's question.

    Returns:
        str: The answer text formatted for display.
    """
    response = build_answer_response(question)
    return format_answer(response.answer, response.citations)


def build_action_prompt(topic: str | None) -> str:
    """Return a friendly follow-up prompt based on the topic.

    Args:
        topic (str | None): The topic to reference, if available.

    Returns:
        str: A suggested follow-up prompt.
    """
    if topic:
        cleaned_topic = topic.strip()
        if cleaned_topic:
            return f"Would you like to learn more about {cleaned_topic}?"
    return "Would you like a quick example or a short quiz on this topic?"


def build_structured_answer_response(question: str) -> StructuredAnswer:
    """Build a structured answer with info and quiz lanes plus fallback logic.

    Args:
        question (str): The user's question.

    Returns:
        StructuredAnswer: JSON-ready answer with lanes and citations.
    """
    retrieval_query = _resolve_followup_query(question)
    retrieved_chunks = retrieve_chunks(
        question=retrieval_query,
        chunks=st.session_state.chunks,
        session_id=st.session_state.session_id,
        document_id=st.session_state.active_document_id,
        user_id=st.session_state.user_id,
    )
    if not retrieved_chunks and st.session_state.chunks:
        # Safety net: if scoped retrieval returns nothing, retry against the currently
        # loaded workspace chunks to avoid false "not found" responses.
        retrieved_chunks = retrieve_chunks(
            question=retrieval_query,
            chunks=st.session_state.chunks,
            session_id="",
            document_id=None,
            user_id=None,
        )
    example = (
        "Example:\n"
        "Context:\n"
        "[Neuro Notes, p.32]\n"
        "The hippocampus encodes and consolidates short-term memories into long-term storage.\n"
        "Question:\n"
        "What does the hippocampus do?\n\n"
        "Expected JSON:\n"
        "{\n"
        "  \"answer\": \"The hippocampus encodes short-term memories and helps consolidate them into long-term storage.\",\n"
        "  \"citations\": [\"Neuro Notes, p.32\"],\n"
        "  \"topic_subject\": \"Hippocampus\",\n"
        "  \"info_lane\": {\n"
        "    \"button_label\": \"🧠 How the hippocampus builds your memories\",\n"
        "    \"query\": \"Tell me more about how the hippocampus encodes and consolidates memories.\"\n"
        "  },\n"
        "  \"quiz_lane\": {\n"
        "    \"button_label\": \"Test your memory knowledge\",\n"
        "    \"intent\": \"START_QUIZ_LOOP\"\n"
        "  }\n"
        "}"
    )
    last_turn_history = _build_last_turn_history()
    structured = generate_structured_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        persona_name="Study Pal",
        example=example,
        chat_history=last_turn_history,
    )
    structured.citations = collect_citations(retrieved_chunks)
    if retrieved_chunks and _is_notes_miss(structured.answer):  # Recover when the model misses despite context.
        best_chunk = _select_summary_chunk(question, retrieved_chunks) or retrieved_chunks[0]
        summary = _summarize_from_chunk(best_chunk.text)
        if summary:
            structured.answer = f"Based on your notes, {summary}"
            structured.used_fallback = False
    if structured.used_fallback or _is_notes_miss(structured.answer):  # Hide action lanes on true misses.
        structured.used_fallback = True
        structured.citations = []
        structured.info_lane = None
        structured.quiz_lane = None
    return structured


def retrieve_note_chunks(query: str) -> list[RetrievedChunk]:
    """Return retrieved chunks for the current session and document.

    Args:
        query (str): The search query to retrieve against.

    Returns:
        list[RetrievedChunk]: Retrieved chunks from local and remote sources.
    """
    try:
        session_chunks = st.session_state.chunks
        session_id = st.session_state.session_id
        document_id = st.session_state.active_document_id
    except (AttributeError, KeyError):
        session_chunks = []
        session_id = ""
        document_id = None

    return retrieve_chunks(
        question=query,
        chunks=session_chunks,
        session_id=session_id,
        document_id=document_id,
        user_id=st.session_state.user_id if "user_id" in st.session_state else None,
    )


def get_supporting_citations(query: str) -> list[str]:
    """Return citations for the best supporting chunks for this query.

    Args:
        query (str): The search query to retrieve against.

    Returns:
        list[str]: Citations extracted from retrieved chunks.
    """
    return collect_citations(retrieve_note_chunks(query))
