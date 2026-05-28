"""llm_client.py: LLM client utilities for answering, quizzes, and planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.core.config import settings
from src.core.openrouter_credentials import get_effective_openrouter_api_key
from src.core.models import (
    DocumentMetadata,
    RetrievedChunk,
    ReteachResponse,
    StructuredAnswer,
    StudyPlan,
    StudyQuiz,
    TeachingResponse,
)
from src.core.observability import langfuse_enabled
from src.llm.answer_generation import (
    AnswerGenerationDeps,
    STRUCTURED_ANSWER_SCHEMA as _STRUCTURED_ANSWER_SCHEMA,
    answer_from_context as _answer_from_context_impl,
    generate_follow_up as _generate_follow_up_impl,
    generate_structured_answer as _generate_structured_answer_impl,
)
from src.llm.parsing import (
    clean_json as _clean_json,
    parse_json_payload as _parse_json_payload,
    repair_book_brief_payload as _repair_book_brief_payload,
    repair_structured_answer_payload as _repair_structured_answer_payload,
    strip_answer_sources_block as _strip_answer_sources_block,
)
from src.llm.learning_generation import (
    LearningGenerationDeps,
    generate_quiz_from_context as _generate_quiz_from_context_impl,
    generate_remediation_from_context as _generate_remediation_from_context_impl,
    generate_study_plan_from_context as _generate_study_plan_from_context_impl,
)
from src.llm.publishing_generation import (
    PublishingGenerationDeps,
    generate_book_brief_from_context as _generate_book_brief_from_context_impl,
    generate_marketing_copy_from_context as _generate_marketing_copy_from_context_impl,
)
from src.llm.postprocess import (
    ensure_action_lanes as _ensure_action_lanes,
    strip_inline_citations as _strip_inline_citations,
    truncate_sentences as _truncate_sentences,
)
from src.llm.prompts import (
    build_document_metadata_prompt,
)
from src.llm.tracing import (
    extract_generation_ids as _extract_generation_ids,
    langfuse_generation as _langfuse_generation,
    update_generation as _update_generation,
)

# Re-export for tests and existing imports.
STRUCTURED_ANSWER_SCHEMA = _STRUCTURED_ANSWER_SCHEMA


@dataclass(frozen=True)
class ChatClient:
    client: OpenAI
    enable_tracing: bool



def _get_chat_client(api_key_override: str | None = None) -> ChatClient | None:
    """Return a configured chat client when an API key is available."""

    api_key = (api_key_override or "").strip() or get_effective_openrouter_api_key()
    if not api_key:
        return None
    return ChatClient(
        client=OpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
        ),
        enable_tracing=langfuse_enabled(),  # defer Langfuse wiring unless explicitly enabled
    )


def _create_chat_completion(chat_client: ChatClient, **kwargs: Any) -> Any:
    completions = cast(Any, chat_client.client.chat.completions)
    return completions.create(**kwargs)



def _build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build one context string from retrieved chunks."""

    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[{chunk.citation}]\n{chunk.text}")
    return "\n\n".join(parts)



def _extract_message_text(response) -> str:
    """Return assistant message text from a chat completion response."""

    return (response.choices[0].message.content or "").strip()


_ANSWER_GENERATION_DEPS = AnswerGenerationDeps(
    get_chat_client=_get_chat_client,
    create_chat_completion=_create_chat_completion,
    langfuse_generation=_langfuse_generation,
    update_generation=_update_generation,
    extract_generation_ids=_extract_generation_ids,
    extract_message_text=_extract_message_text,
    parse_json_payload=_parse_json_payload,
    repair_structured_answer_payload=_repair_structured_answer_payload,
    strip_answer_sources_block=_strip_answer_sources_block,
    truncate_sentences=_truncate_sentences,
    strip_inline_citations=_strip_inline_citations,
    ensure_action_lanes=_ensure_action_lanes,
    build_context=_build_context,
)



def generate_document_metadata(filename: str, document_excerpt: str) -> DocumentMetadata | None:
    """Generate document title/topic/summary from a text excerpt."""

    if not document_excerpt.strip():
        return None

    chat_client = _get_chat_client()
    if chat_client is None:
        return None

    prompt_bundle = build_document_metadata_prompt(filename, document_excerpt)
    try:
        with _langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="document_metadata",
            prompt_bundle=prompt_bundle,
            metadata={"filename": filename, "excerpt_chars": str(len(document_excerpt))},
        ) as generation:
            response = _create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=settings.max_chat_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You extract concise, structured metadata from study documents.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            _update_generation(generation, _extract_message_text(response), response)
        payload = json.loads(_clean_json(_extract_message_text(response)))
        return DocumentMetadata.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None



def answer_from_context(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    api_key_override: str | None = None,
    theme_synthesis: bool = False,
) -> TeachingResponse:
    """Answer a question using only the provided context."""
    return _answer_from_context_impl(
        question=question,
        retrieved_chunks=retrieved_chunks,
        api_key_override=api_key_override,
        theme_synthesis=theme_synthesis,
        deps=_ANSWER_GENERATION_DEPS,
    )

def generate_follow_up(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    answer: str,

    used_fallback: bool,
) -> str | None:
    """Generate a short follow-up question to keep the learner engaged."""
    return _generate_follow_up_impl(
        question=question,
        retrieved_chunks=retrieved_chunks,
        answer=answer,
        used_fallback=used_fallback,
        deps=_ANSWER_GENERATION_DEPS,
    )

def generate_structured_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    persona_name: str,
    example: str,

    chat_history: str = "",
) -> StructuredAnswer:
    """Generate structured answer."""
    return _generate_structured_answer_impl(
        question=question,
        retrieved_chunks=retrieved_chunks,
        persona_name=persona_name,
        example=example,
        chat_history=chat_history,
        deps=_ANSWER_GENERATION_DEPS,
    )


_LEARNING_GENERATION_DEPS = LearningGenerationDeps(
    get_chat_client=_get_chat_client,
    create_chat_completion=_create_chat_completion,
    langfuse_generation=_langfuse_generation,
    update_generation=_update_generation,
    extract_message_text=_extract_message_text,
    parse_json_payload=_parse_json_payload,
    build_context=_build_context,
)


def generate_quiz_from_context(
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    num_questions: int,

    weak_concepts: list[str] | None = None,
) -> StudyQuiz | None:
    """Generate a quiz from the provided context."""
    return _generate_quiz_from_context_impl(
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        num_questions=num_questions,
        weak_concepts=weak_concepts,
        deps=_LEARNING_GENERATION_DEPS,
    )


def generate_remediation_from_context(
    topic: str,
    weak_concepts_with_error: str,

    retrieved_chunks: list[RetrievedChunk],
) -> ReteachResponse | None:
    """Generate a reteach explanation from the provided context."""
    return _generate_remediation_from_context_impl(
        topic=topic,
        weak_concepts_with_error=weak_concepts_with_error,
        retrieved_chunks=retrieved_chunks,
        deps=_LEARNING_GENERATION_DEPS,
    )


def generate_study_plan_from_context(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],

    retrieved_chunks: list[RetrievedChunk],
) -> StudyPlan | None:
    """Generate a study plan from the provided context."""
    return _generate_study_plan_from_context_impl(
        topic=topic,
        weak_concepts=weak_concepts,
        reviewed_concepts=reviewed_concepts,
        retrieved_chunks=retrieved_chunks,
        deps=_LEARNING_GENERATION_DEPS,
    )


_PUBLISHING_GENERATION_DEPS = PublishingGenerationDeps(
    get_chat_client=_get_chat_client,
    create_chat_completion=_create_chat_completion,
    langfuse_generation=_langfuse_generation,
    update_generation=_update_generation,
    extract_message_text=_extract_message_text,
    parse_json_payload=_parse_json_payload,
    repair_book_brief_payload=_repair_book_brief_payload,
)


def generate_book_brief_from_context(
    *,
    retrieved_chunks: list[RetrievedChunk],
    audience: str | None = None,
    spoiler_level: str | None = None,
    notes: str | None = None,
    document_title_hint: str | None = None,
    api_key_override: str | None = None,
) -> dict[str, object] | None:
    """Generate a structured Publishing Mode book brief from retrieved context."""
    return _generate_book_brief_from_context_impl(
        retrieved_chunks=retrieved_chunks,
        audience=audience,
        spoiler_level=spoiler_level,
        notes=notes,
        document_title_hint=document_title_hint,
        api_key_override=api_key_override,
        deps=_PUBLISHING_GENERATION_DEPS,
    )


def generate_marketing_copy_from_context(
    *,
    retrieved_chunks: list[RetrievedChunk],
    output_type: str,
    tone: str | None = None,
    audience: str | None = None,
    spoiler_level: str | None = None,
    length: str | None = None,
    notes: str | None = None,
    document_title_hint: str | None = None,
    api_key_override: str | None = None,
) -> dict[str, object] | None:
    """Generate structured Publishing Mode marketing copy from retrieved context."""
    return _generate_marketing_copy_from_context_impl(
        retrieved_chunks=retrieved_chunks,
        output_type=output_type,
        tone=tone,
        audience=audience,
        spoiler_level=spoiler_level,
        length=length,
        notes=notes,
        document_title_hint=document_title_hint,
        api_key_override=api_key_override,
        deps=_PUBLISHING_GENERATION_DEPS,
    )
