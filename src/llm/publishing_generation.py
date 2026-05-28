"""publishing_generation.py: Publishing-mode generation pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.llm.parsing import JSON_OBJECT_RESPONSE_FORMAT
from src.llm.prompts import PromptBundle
from src.publishing.prompts import build_book_brief_prompt, build_marketing_copy_prompt


@dataclass(frozen=True)
class PublishingGenerationDeps:
    """Injected dependencies for publishing generators."""

    get_chat_client: Callable[[str | None], Any]
    create_chat_completion: Callable[..., Any]
    langfuse_generation: Callable[..., Any]
    update_generation: Callable[[object, str, object], None]
    extract_message_text: Callable[[Any], str]
    parse_json_payload: Callable[[str], dict[str, object]]
    repair_book_brief_payload: Callable[..., dict[str, object] | None]


def generate_book_brief_from_context(
    *,
    retrieved_chunks: list[RetrievedChunk],
    audience: str | None = None,
    spoiler_level: str | None = None,
    notes: str | None = None,
    document_title_hint: str | None = None,
    api_key_override: str | None = None,
    deps: PublishingGenerationDeps,
) -> dict[str, object] | None:
    """Generate a structured Publishing-mode book brief from retrieved context."""
    if not retrieved_chunks:
        return None

    chat_client = deps.get_chat_client(api_key_override)
    if chat_client is None:
        return None

    prompt_text = build_book_brief_prompt(
        retrieved_chunks=retrieved_chunks,
        audience=audience,
        spoiler_level=spoiler_level,
        notes=notes,
        document_title_hint=document_title_hint,
    )
    prompt_bundle = PromptBundle(
        text=prompt_text,
        prompt=None,
        name="publishing_book_brief",
    )
    for attempt in range(2):
        try:
            with deps.langfuse_generation(
                enabled=chat_client.enable_tracing,
                feature="publishing_book_brief",
                prompt_bundle=prompt_bundle,
                metadata={
                    "num_chunks": str(len(retrieved_chunks)),
                    "audience": (audience or "").strip(),
                    "spoiler_level": (spoiler_level or "").strip(),
                    "attempt": str(attempt + 1),
                },
            ) as generation:
                response = deps.create_chat_completion(
                    chat_client,
                    model=settings.chat_model,
                    max_tokens=max(settings.max_chat_tokens, 1000),
                    response_format=JSON_OBJECT_RESPONSE_FORMAT,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You create grounded publishing briefs as strict JSON. "
                                "Never invent unsupported details."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                )
                output_text = deps.extract_message_text(response)
                deps.update_generation(generation, output_text, response)
            try:
                return deps.parse_json_payload(output_text)
            except (ValueError, json.JSONDecodeError):
                repaired_payload = deps.repair_book_brief_payload(
                    chat_client=chat_client,
                    prompt_text=prompt_text,
                    raw_output=output_text,
                    create_chat_completion=deps.create_chat_completion,
                    extract_message_text=deps.extract_message_text,
                )
                if repaired_payload is not None:
                    return repaired_payload
        except (APIConnectionError, APIStatusError, APITimeoutError):
            continue
    return None


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
    deps: PublishingGenerationDeps,
) -> dict[str, object] | None:
    """Generate structured Publishing-mode marketing copy from retrieved context."""
    if not retrieved_chunks:
        return None

    chat_client = deps.get_chat_client(api_key_override)
    if chat_client is None:
        return None

    prompt_text = build_marketing_copy_prompt(
        retrieved_chunks=retrieved_chunks,
        output_type=output_type,
        tone=tone,
        audience=audience,
        spoiler_level=spoiler_level,
        length=length,
        notes=notes,
        document_title_hint=document_title_hint,
    )
    prompt_bundle = PromptBundle(
        text=prompt_text,
        prompt=None,
        name="publishing_marketing_copy",
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="publishing_marketing_copy",
            prompt_bundle=prompt_bundle,
            metadata={
                "num_chunks": str(len(retrieved_chunks)),
                "output_type": output_type,
                "audience": (audience or "").strip(),
                "spoiler_level": (spoiler_level or "").strip(),
            },
        ) as generation:
            response = deps.create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=max(settings.max_chat_tokens, 900),
                response_format=JSON_OBJECT_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create grounded publishing marketing outputs as strict JSON. "
                            "Never invent unsupported details."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
            )
            output_text = deps.extract_message_text(response)
            deps.update_generation(generation, output_text, response)
        return deps.parse_json_payload(output_text)
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, json.JSONDecodeError):
        return None
