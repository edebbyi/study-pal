from __future__ import annotations

import json

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import settings
from src.models import DocumentMetadata, RetrievedChunk, StudyPlan, StudyQuiz, TeachingResponse
from src.prompts import (
    build_answer_prompt,
    build_document_metadata_prompt,
    build_quiz_prompt,
    build_reteach_prompt,
    build_study_plan_prompt,
)


def _get_chat_client() -> OpenAI | None:
    if not settings.openrouter_api_key:
        return None
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[{chunk.citation}]\n{chunk.text}")
    return "\n\n".join(parts)


def _extract_message_text(response) -> str:
    return (response.choices[0].message.content or "").strip()


def generate_document_metadata(filename: str, document_excerpt: str) -> DocumentMetadata | None:
    if not document_excerpt.strip():
        return None

    client = _get_chat_client()
    if client is None:
        return None

    prompt = build_document_metadata_prompt(filename, document_excerpt)
    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            max_tokens=settings.max_chat_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise, structured metadata from study documents.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(_extract_message_text(response))
        return DocumentMetadata.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None


def answer_from_context(question: str, retrieved_chunks: list[RetrievedChunk]) -> TeachingResponse:
    if not retrieved_chunks:
        return TeachingResponse(
            answer="I couldn't find relevant support in the uploaded notes for that question yet.",
            citations=[],
            used_fallback=True,
        )

    client = _get_chat_client()
    if client is not None:
        prompt = build_answer_prompt(_build_context(retrieved_chunks), question)
        try:
            response = client.chat.completions.create(
                model=settings.chat_model,
                max_tokens=settings.max_chat_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful tutor who only answers from provided class notes.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            message = _extract_message_text(response)
            return TeachingResponse(
                answer=message.strip(),
                citations=[chunk.citation for chunk in retrieved_chunks],
                used_fallback=False,
            )
        except (APIConnectionError, APIStatusError, APITimeoutError):
            pass

    top_chunk = retrieved_chunks[0]
    answer = (
        "Based on your notes, the most relevant passage says:\n\n"
        f"{top_chunk.text}\n\n"
        "This is a retrieval-based fallback answer. Add API credentials to upgrade this to a fuller generated explanation."
    )
    return TeachingResponse(
        answer=answer,
        citations=[chunk.citation for chunk in retrieved_chunks],
        used_fallback=True,
    )


def generate_quiz_from_context(
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    num_questions: int,
    weak_concepts: list[str] | None = None,
) -> StudyQuiz | None:
    if not retrieved_chunks:
        return None

    client = _get_chat_client()
    if client is None:
        return None

    prompt = build_quiz_prompt(
        topic=topic,
        context=_build_context(retrieved_chunks),
        num_questions=num_questions,
        weak_concepts=weak_concepts,
    )
    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            max_tokens=settings.max_chat_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You create structured, note-grounded quiz content in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(_extract_message_text(response))
        return StudyQuiz.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None


def generate_remediation_from_context(
    topic: str,
    weak_concepts: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> str | None:
    if not weak_concepts or not retrieved_chunks:
        return None

    client = _get_chat_client()
    if client is None:
        return None

    prompt = build_reteach_prompt(
        topic=topic,
        weak_concepts=weak_concepts,
        context=_build_context(retrieved_chunks),
    )
    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            max_tokens=settings.max_chat_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You provide short, note-grounded reteaching explanations.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return _extract_message_text(response)
    except (APIConnectionError, APIStatusError, APITimeoutError):
        return None


def generate_study_plan_from_context(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> StudyPlan | None:
    if not retrieved_chunks:
        return None

    client = _get_chat_client()
    if client is None:
        return None

    prompt = build_study_plan_prompt(
        topic=topic,
        weak_concepts=weak_concepts,
        reviewed_concepts=reviewed_concepts,
        context=_build_context(retrieved_chunks),
    )
    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            max_tokens=settings.max_chat_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You create structured, note-grounded study plans in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(_extract_message_text(response))
        return StudyPlan.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None
