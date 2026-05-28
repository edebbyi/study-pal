"""answer_generation.py: Answer/follow-up/structured-answer generation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.config import settings
from src.core.models import InfoLane, QuizLane, RetrievedChunk, StructuredAnswer, TeachingResponse
from src.llm.parsing import JSON_OBJECT_RESPONSE_FORMAT
from src.llm.prompts import build_answer_prompt, build_follow_up_prompt, build_structured_answer_prompt

STRUCTURED_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations", "topic_subject", "info_lane", "quiz_lane"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "topic_subject": {"type": "string"},
        "info_lane": {
            "type": "object",
            "additionalProperties": False,
            "required": ["button_label", "query"],
            "properties": {
                "button_label": {"type": "string"},
                "query": {"type": "string"},
            },
        },
        "quiz_lane": {
            "type": "object",
            "additionalProperties": False,
            "required": ["button_label", "intent"],
            "properties": {
                "button_label": {"type": "string"},
                "intent": {"type": "string"},
            },
        },
    },
}
STRUCTURED_ANSWER_RESPONSE_FORMAT_JSON_SCHEMA: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "structured_answer",
        "strict": True,
        "schema": STRUCTURED_ANSWER_SCHEMA,
    },
}


@dataclass(frozen=True)
class AnswerGenerationDeps:
    """Injected dependencies for answer/follow-up/structured-answer generators."""

    get_chat_client: Callable[[str | None], Any]
    create_chat_completion: Callable[..., Any]
    langfuse_generation: Callable[..., Any]
    update_generation: Callable[[object, str, object], None]
    extract_generation_ids: Callable[[object], tuple[str | None, str | None]]
    extract_message_text: Callable[[Any], str]
    parse_json_payload: Callable[[str], dict[str, object]]
    repair_structured_answer_payload: Callable[..., dict[str, object] | None]
    strip_answer_sources_block: Callable[[str], str]
    truncate_sentences: Callable[[str, int], str]
    strip_inline_citations: Callable[[str], str]
    ensure_action_lanes: Callable[[StructuredAnswer, str], None]
    build_context: Callable[[list[RetrievedChunk]], str]


def _fallback_follow_up(question: str, used_fallback: bool) -> str:
    if used_fallback:
        return "Would you like to upload more notes so I can give a deeper answer?"
    return "Would you like me to explain this with a quick example or a short quiz?"


def _normalize_follow_up(follow_up: str) -> str:
    stripped = follow_up.strip()
    lowered = stripped.lower()
    if lowered.startswith("can you explain"):
        return "Would you like me to explain" + stripped[len("can you explain") :]
    if lowered.startswith("could you explain"):
        return "Would you like me to explain" + stripped[len("could you explain") :]
    if lowered.startswith("can you walk me through"):
        return "Would you like me to walk you through" + stripped[len("can you walk me through") :]
    if lowered.startswith("could you walk me through"):
        return "Would you like me to walk you through" + stripped[len("could you walk me through") :]
    return stripped


def answer_from_context(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    api_key_override: str | None = None,
    theme_synthesis: bool = False,
    deps: AnswerGenerationDeps,
) -> TeachingResponse:
    """Answer a question using only provided context."""
    if not retrieved_chunks:
        return TeachingResponse(
            answer="I couldn't find relevant support in the uploaded notes for that question yet.",
            citations=[],
            used_fallback=True,
        )

    chat_client = deps.get_chat_client(api_key_override)
    if chat_client is not None:
        system_prompt = "You are a careful tutor who only answers from provided class notes."
        if theme_synthesis:
            system_prompt = (
                "You are an expert academic tutor. Your task is to extract and synthesize the core themes "
                "of a book using ONLY the provided class notes.\n\n"
                "Follow these strict execution rules:\n"
                "1. DO NOT simply list chapter titles, table of contents, or raw topics.\n"
                '2. If the notes explicitly mention a specific list (like "Eight Core Concepts") but do not '
                "outline them, do not stall or apologize repeatedly. Acknowledge the missing list in exactly "
                "one brief sentence, then immediately pivot to your main task.\n"
                "3. Synthesize the underlying themes by looking at the broader picture. Group the available "
                "chapter topics into 3-4 conceptual pillars (for example, how they connect human behavior, "
                "biological mechanics, or change over time).\n"
                "4. Maintain a supportive, academic, and direct tone. Never be defensive about missing data; "
                "maximize the value of the information that is present."
            )
        prompt_bundle = build_answer_prompt(
            deps.build_context(retrieved_chunks),
            question,
            theme_synthesis=theme_synthesis,
            use_langfuse_template=settings.langfuse_use_answer_prompt_template,
        )
        try:
            with deps.langfuse_generation(
                enabled=chat_client.enable_tracing,
                feature="answer_from_context",
                prompt_bundle=prompt_bundle,
                metadata={"question_chars": str(len(question)), "num_chunks": str(len(retrieved_chunks))},
            ) as generation:
                response = deps.create_chat_completion(
                    chat_client,
                    model=settings.chat_model,
                    max_tokens=settings.max_chat_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt_bundle.text},
                    ],
                )
                deps.update_generation(generation, deps.extract_message_text(response), response)
            trace_id, observation_id = deps.extract_generation_ids(generation)
            message = deps.extract_message_text(response)
            return TeachingResponse(
                answer=message.strip(),
                citations=[chunk.citation for chunk in retrieved_chunks],
                used_fallback=False,
                trace_id=trace_id,
                observation_id=observation_id,
            )
        except (APIConnectionError, APIStatusError, APITimeoutError):
            pass

    top_chunk = retrieved_chunks[0]
    answer = (
        "Based on your notes, the most relevant passage says:\n\n"
        f"{top_chunk.text}\n\n"
        "This is a retrieval-based fallback answer. Add your OpenRouter key in Settings to upgrade this answer."
    )
    return TeachingResponse(
        answer=answer,
        citations=[chunk.citation for chunk in retrieved_chunks],
        used_fallback=True,
    )


def generate_follow_up(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    answer: str,
    used_fallback: bool,
    *,
    deps: AnswerGenerationDeps,
) -> str | None:
    """Generate a short follow-up question to keep learners engaged."""
    if used_fallback:
        return None

    chat_client = deps.get_chat_client(None)
    if chat_client is None:
        return None

    prompt_bundle = build_follow_up_prompt(
        question=question,
        answer=answer,
        context=deps.build_context(retrieved_chunks),
        used_fallback=used_fallback,
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="follow_up",
            prompt_bundle=prompt_bundle,
            metadata={"question_chars": str(len(question)), "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = deps.create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=120,
                response_format=JSON_OBJECT_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate short, engaging follow-up questions for learners.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            output_text = deps.extract_message_text(response)
            deps.update_generation(generation, output_text, response)
        payload = deps.parse_json_payload(output_text)
        follow_up = str(payload.get("follow_up", "")).strip()
        if follow_up:
            return _normalize_follow_up(follow_up)
        return _fallback_follow_up(question, used_fallback)
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError):
        return _fallback_follow_up(question, used_fallback)


def generate_structured_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    persona_name: str,
    example: str,
    chat_history: str = "",
    deps: AnswerGenerationDeps,
) -> StructuredAnswer:
    """Generate a structured answer payload from grounded context."""
    chat_client = deps.get_chat_client(None)
    if chat_client is None:
        return StructuredAnswer(
            answer="Navigate to Settings and enter you OpenRouter key for chat usage.",
            citations=[],
            info_lane=InfoLane(
                button_label="🧠 Share more notes",
                query="I can help more if you upload or select additional notes.",
            ),
            quiz_lane=QuizLane(button_label="Test your knowledge on this"),
            used_fallback=True,
        )
    if not retrieved_chunks:
        return StructuredAnswer(
            answer="I couldn't find that in the notes yet.",
            citations=[],
            info_lane=InfoLane(
                button_label="🧠 Share more notes",
                query="I can help more if you upload or select additional notes.",
            ),
            quiz_lane=QuizLane(button_label="Test your knowledge on this"),
            used_fallback=True,
        )

    prompt_bundle = build_structured_answer_prompt(
        context=deps.build_context(retrieved_chunks),
        question=question,
        persona_name=persona_name,
        example=example,
        chat_history=chat_history,
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="structured_answer",
            prompt_bundle=prompt_bundle,
            metadata={"question_chars": str(len(question)), "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = None
            output_text = ""
            payload: dict[str, object] | None = None
            response_formats: list[dict[str, object]] = [
                STRUCTURED_ANSWER_RESPONSE_FORMAT_JSON_SCHEMA,
                JSON_OBJECT_RESPONSE_FORMAT,
            ]

            for response_format in response_formats:
                try:
                    response = deps.create_chat_completion(
                        chat_client,
                        model=settings.chat_model,
                        max_tokens=settings.max_chat_tokens,
                        response_format=response_format,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You return structured, note-grounded answers and action lanes in JSON. "
                                    "Do not reveal chain-of-thought."
                                ),
                            },
                            {"role": "user", "content": prompt_bundle.text},
                        ],
                    )
                    output_text = deps.extract_message_text(response)
                    payload = deps.parse_json_payload(output_text)
                    break
                except (APIConnectionError, APIStatusError, APITimeoutError):
                    continue
                except ValueError:
                    payload = deps.repair_structured_answer_payload(
                        chat_client=chat_client,
                        prompt_text=prompt_bundle.text,
                        raw_output=output_text,
                        structured_answer_schema=STRUCTURED_ANSWER_SCHEMA,
                        create_chat_completion=deps.create_chat_completion,
                        extract_message_text=deps.extract_message_text,
                    )
                    if payload is not None:
                        break
            if response is not None:
                deps.update_generation(generation, output_text, response)
        if payload is None:
            raise ValueError("Unable to parse structured answer payload")
        structured = StructuredAnswer.model_validate(payload)
        trace_id, observation_id = deps.extract_generation_ids(generation)
        sanitized_answer = deps.strip_answer_sources_block(structured.answer)
        structured.answer = deps.truncate_sentences(deps.strip_inline_citations(sanitized_answer), 4)
        deps.ensure_action_lanes(structured, question)
        if not structured.citations:
            structured.citations = [chunk.citation for chunk in retrieved_chunks]
        structured.trace_id = trace_id
        structured.observation_id = observation_id
        if structured.topic_subject:
            structured.topic_subject = structured.topic_subject.strip()
        return structured
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError):
        return StructuredAnswer(
            answer="I couldn't find that in the notes yet.",
            citations=[],
            info_lane=InfoLane(
                button_label="🧠 Share more notes",
                query="I can help more if you upload or select additional notes.",
            ),
            quiz_lane=QuizLane(button_label="Test your knowledge on this"),
            used_fallback=True,
        )
