"""parsing.py: JSON parsing and repair helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.config import settings

JSON_OBJECT_RESPONSE_FORMAT: dict[str, object] = {"type": "json_object"}


def clean_json(raw_response: str) -> str:
    """Normalize model output before JSON parsing."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def extract_answer_tag(raw_response: str) -> str:
    """Extract `<answer>` content when tag wrappers are present."""
    match = re.search(r"<answer>(.*?)</answer>", raw_response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response


def strip_answer_sources_block(text: str) -> str:
    """Remove model-generated source headings from the answer body."""
    lines = [line.rstrip() for line in text.splitlines()]
    heading_index = None
    for index, line in enumerate(lines):
        normalized = re.sub(r"[^a-z]", "", line.lower())
        if normalized in {"sources", "sourcesused", "citations", "references"}:
            heading_index = index
            break
    if heading_index is None:
        return text.strip()
    return "\n".join(lines[:heading_index]).strip()


def parse_json_payload(raw_response: str) -> dict[str, object]:
    """Parse JSON object payloads from model output."""
    cleaned = clean_json(raw_response)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    tagged_answer = extract_answer_tag(cleaned)
    if tagged_answer != cleaned:
        parsed = json.loads(clean_json(tagged_answer))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response is not a valid JSON object")


def repair_structured_answer_payload(
    *,
    chat_client: Any,
    prompt_text: str,
    raw_output: str,
    structured_answer_schema: dict[str, object],
    create_chat_completion: Callable[..., Any],
    extract_message_text: Callable[[Any], str],
) -> dict[str, object] | None:
    """Try to repair malformed structured-answer output."""
    repair_prompt = (
        "Fix the malformed JSON so it matches this schema exactly.\n"
        "Return only a valid JSON object.\n\n"
        f"Schema:\n{json.dumps(structured_answer_schema)}\n\n"
        f"Original prompt:\n{prompt_text}\n\n"
        f"Malformed output:\n{raw_output}"
    )
    try:
        response = create_chat_completion(
            chat_client,
            model=settings.chat_model,
            max_tokens=settings.max_chat_tokens,
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
            messages=[
                {
                    "role": "system",
                    "content": "You repair malformed JSON and return only valid JSON.",
                },
                {"role": "user", "content": repair_prompt},
            ],
        )
        return parse_json_payload(extract_message_text(response))
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, json.JSONDecodeError):
        return None


def repair_book_brief_payload(
    *,
    chat_client: Any,
    prompt_text: str,
    raw_output: str,
    create_chat_completion: Callable[..., Any],
    extract_message_text: Callable[[Any], str],
) -> dict[str, object] | None:
    """Repair malformed Book Brief JSON output."""
    schema_hint = {
        "title": "string|null",
        "genre": "string|null",
        "primary_audience": "string",
        "secondary_audience": "string|null",
        "reader_buyer_persona": {
            "persona_name": "string",
            "role_context": "string",
            "motivation": "string",
            "needs": "string",
            "likely_objections": "string",
            "discovery_channels": "string",
            "messaging_notes": "string",
        },
        "core_themes": ["string"],
        "audience_keywords": ["string"],
        "one_sentence_positioning": "string",
        "positioning_recommendation": "string",
        "marketing_angles": ["string"],
        "sales_use_case": "string",
        "risk_flags": ["string"],
    }
    repair_prompt = (
        "Fix malformed JSON and return only one valid JSON object.\n"
        "Do not add markdown, comments, or code fences.\n\n"
        f"Expected schema:\n{json.dumps(schema_hint)}\n\n"
        f"Original prompt:\n{prompt_text}\n\n"
        f"Malformed output:\n{raw_output}"
    )
    try:
        response = create_chat_completion(
            chat_client,
            model=settings.chat_model,
            max_tokens=max(settings.max_chat_tokens, 700),
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
            messages=[
                {
                    "role": "system",
                    "content": "You repair malformed JSON and return only valid JSON.",
                },
                {"role": "user", "content": repair_prompt},
            ],
        )
        return parse_json_payload(extract_message_text(response))
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, json.JSONDecodeError):
        return None
