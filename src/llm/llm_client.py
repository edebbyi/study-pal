"""llm_client.py: LLM client utilities for answering, quizzes, and planning."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, cast

from langfuse import get_client
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.core.config import settings
from src.core.models import (
    DocumentMetadata,
    InfoLane,
    QuizLane,
    RetrievedChunk,
    ReteachResponse,
    StructuredAnswer,
    StudyPlan,
    StudyQuiz,
    TeachingResponse,
)
from src.core.observability import build_langfuse_metadata, configure_langfuse_environment, langfuse_enabled
from src.llm.prompts import (
    PromptBundle,
    build_answer_prompt,
    build_document_metadata_prompt,
    build_follow_up_prompt,
    build_quiz_prompt,
    build_reteach_prompt,
    build_structured_answer_prompt,
    build_study_plan_prompt,
)

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
JSON_OBJECT_RESPONSE_FORMAT: dict[str, object] = {"type": "json_object"}


@dataclass(frozen=True)
class ChatClient:
    client: OpenAI
    enable_tracing: bool



def _get_chat_client() -> ChatClient | None:
    """Create a chat client when API credentials are available.
    
    Returns:
        ChatClient | None: Result value.
    """

    if not settings.openrouter_api_key:
        return None
    return ChatClient(
        client=OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        ),
        enable_tracing=langfuse_enabled(),  # defer Langfuse wiring unless explicitly enabled
    )



def _langfuse_kwargs(enabled: bool, feature: str, extra: dict[str, str]) -> dict[str, object]:
    """Build Langfuse tracing arguments when available.
    
    Args:
        enabled (bool): Input parameter.
        feature (str): Input parameter.
        extra (dict[str, str]): Input parameter.
    
    Returns:
        dict[str, object]: Mapping of computed results.
    """

    if not enabled:
        return {}
    return {
        "metadata": build_langfuse_metadata(feature, extra),
    }


def _langfuse_generation(
    *,
    enabled: bool,
    feature: str,
    prompt_bundle: PromptBundle,

    metadata: dict[str, str],
) -> ContextManager[Any]:
    """Langfuse generation.
    
    Args:
        enabled (bool): Input parameter.
        feature (str): Input parameter.
        prompt_bundle (PromptBundle): Input parameter.
        metadata (dict[str, str]): Input parameter.
    
    Returns:
        object: Result value.
    """

    if not enabled:
        return nullcontext()

    if not configure_langfuse_environment():
        return nullcontext()

    try:
        langfuse = cast(Any, get_client())
        return langfuse.start_as_current_observation(
            name=feature,
            as_type="generation",
            prompt=prompt_bundle.prompt,  # send prompt object so Langfuse can link template/version
            input=prompt_bundle.text,
            metadata=build_langfuse_metadata(feature, metadata),
            model=settings.chat_model,
        )
    except Exception:
        return nullcontext()


def _create_chat_completion(chat_client: ChatClient, **kwargs: Any) -> Any:
    completions = cast(Any, chat_client.client.chat.completions)
    return completions.create(**kwargs)



def _update_generation(generation: object, output_text: str, response: object) -> None:
    """Update generation.
    
    Args:
        generation (object): Input parameter.
        output_text (str): Input parameter.
        response (object): Input parameter.
    """

    if not hasattr(generation, "update"):
        return

    usage = getattr(response, "usage", None)
    usage_details = None
    if usage is not None:
        usage_details = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    try:
        generation.update(output=output_text, usage_details=usage_details)
    except Exception:
        return



def _extract_generation_ids(generation: object) -> tuple[str | None, str | None]:
    """Extract generation ids.
    
    Args:
        generation (object): Input parameter.
    
    Returns:
        tuple[str | None, str | None]: Result value.
    """

    if generation is None:
        return None, None
    trace_id = getattr(generation, "trace_id", None) or getattr(generation, "traceId", None)
    observation_id = (
        getattr(generation, "id", None)
        or getattr(generation, "observation_id", None)
        or getattr(generation, "observationId", None)
    )
    return trace_id, observation_id



def _fallback_follow_up(question: str, used_fallback: bool) -> str:
    """Fallback follow up.
    
    Args:
        question (str): The user question to answer.
        used_fallback (bool): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    if used_fallback:
        return "Would you like to upload more notes so I can give a deeper answer?"
    return "Would you like me to explain this with a quick example or a short quiz?"



def _normalize_follow_up(follow_up: str) -> str:
    """Normalize follow up.
    
    Args:
        follow_up (str): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

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



def _build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build a compact context string from retrieved chunks.
    
    Args:
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        str: Formatted text result.
    """

    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[{chunk.citation}]\n{chunk.text}")
    return "\n\n".join(parts)



def _extract_message_text(response) -> str:
    """Pull the message text out of a chat completion response.
    
    Args:
        response: Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    return (response.choices[0].message.content or "").strip()



def _clean_json(raw_response: str) -> str:
    """Normalize model output before JSON parsing.

    Args:
        raw_response (str): Raw model output text.

    Returns:
        str: Cleaned output text.
    """
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()  # strip fenced wrappers
    return cleaned


def _extract_answer_tag(raw_response: str) -> str:
    """Extract `<answer>` content when tag wrappers are present.

    Args:
        raw_response (str): Raw model output.

    Returns:
        str: Tagged answer content or original output.
    """
    match = re.search(r"<answer>(.*?)</answer>", raw_response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response


def _strip_answer_sources_block(text: str) -> str:
    """Remove model-generated source headings from the answer body.

    Args:
        text (str): Answer text that may include a sources block.

    Returns:
        str: Sanitized answer text.
    """
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


def _parse_json_payload(raw_response: str) -> dict[str, object]:
    """Parse JSON object payloads from model output.

    Args:
        raw_response (str): Raw model output.

    Returns:
        dict[str, object]: Parsed payload.
    """
    cleaned = _clean_json(raw_response)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    tagged_answer = _extract_answer_tag(cleaned)
    if tagged_answer != cleaned:
        parsed = json.loads(_clean_json(tagged_answer))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response is not a valid JSON object")


def _repair_structured_answer_payload(
    *,
    chat_client: ChatClient,
    prompt_text: str,
    raw_output: str,
) -> dict[str, object] | None:
    """Try to repair malformed structured-answer output.

    Args:
        chat_client (ChatClient): Active chat client.
        prompt_text (str): Original prompt text.
        raw_output (str): Malformed output that needs repair.

    Returns:
        dict[str, object] | None: Repaired payload when successful.
    """
    repair_prompt = (
        "Fix the malformed JSON so it matches this schema exactly.\n"
        "Return only a valid JSON object.\n\n"
        f"Schema:\n{json.dumps(STRUCTURED_ANSWER_SCHEMA)}\n\n"
        f"Original prompt:\n{prompt_text}\n\n"
        f"Malformed output:\n{raw_output}"
    )
    try:
        response = _create_chat_completion(
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
        return _parse_json_payload(_extract_message_text(response))
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, json.JSONDecodeError):
        return None


def _truncate_sentences(text: str, max_sentences: int) -> str:
    """Truncate sentences.
    
    Args:
        text (str): Input text to process.
        max_sentences (int): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    if max_sentences <= 0:
        return ""
    normalized = " ".join(text.split())
    if not normalized:
        return normalized
    sentences = re.split(r"(?<=[.!?])\s+", normalized)  # keep sentence boundary punctuation
    if len(sentences) <= max_sentences:
        return normalized
    return " ".join(sentences[:max_sentences]).strip()



def _ensure_info_lane_emoji(label: str, fallback_emoji: str = "🧠") -> str:
    """Ensure info lane emoji.
    
    Args:
        label (str): Input parameter.
        fallback_emoji (str): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    stripped = label.strip()
    if not stripped:
        return f"{fallback_emoji} Learn more"
    first_char = stripped[0]
    if first_char.isascii():
        return f"{fallback_emoji} {stripped}"
    return stripped


def _derive_subject(question: str, structured_topic_subject: str | None) -> str:
    """Derive a compact subject label for default action lanes.

    Args:
        question (str): User question text.
        structured_topic_subject (str | None): Subject label returned by the model.

    Returns:
        str: Short subject text suitable for button labels.
    """
    if structured_topic_subject and structured_topic_subject.strip():
        return structured_topic_subject.strip()

    raw = question.strip().rstrip("?.! ")
    raw = re.sub(
        r"^(what\s+is|what\s+are|who\s+is|define|explain|tell\s+me\s+about|how\s+does|how\s+do)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return "this topic"
    return raw[:60]


def _ensure_action_lanes(structured: StructuredAnswer, question: str) -> None:
    """Ensure every structured answer has usable info and quiz lanes.

    Args:
        structured (StructuredAnswer): Parsed structured payload.
        question (str): User question text.
    """
    subject = _derive_subject(question, structured.topic_subject)

    if structured.info_lane is None:
        structured.info_lane = InfoLane(
            button_label=f"🧠 Why is {subject} important?",
            query=f"Explain {subject} in more detail from the notes.",
        )
    else:
        if not structured.info_lane.button_label.strip():
            structured.info_lane.button_label = f"🧠 Why is {subject} important?"
        if not structured.info_lane.query.strip():
            structured.info_lane.query = f"Explain {subject} in more detail from the notes."
    structured.info_lane.button_label = _ensure_info_lane_emoji(structured.info_lane.button_label)

    if structured.quiz_lane is None:
        structured.quiz_lane = QuizLane(button_label=f"Test your {subject} knowledge")
    elif not structured.quiz_lane.button_label.strip():
        structured.quiz_lane.button_label = f"Test your {subject} knowledge"


def _strip_inline_citations(text: str) -> str:
    """Strip inline citations.
    
    Args:
        text (str): Input text to process.
    
    Returns:
        str: Formatted text result.
    """

    if not text:
        return text
    cleaned = re.sub(  # hide inline citations before truncating for UI
        r"\[[^\]]*(?:\.pdf|page|p\.|chapter|notes|doc)[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip()



def generate_document_metadata(filename: str, document_excerpt: str) -> DocumentMetadata | None:
    """Generate document title, topic, and summary from an excerpt.
    
    Args:
        filename (str): Filename associated with the document.
        document_excerpt (str): Input parameter.
    
    Returns:
        DocumentMetadata | None: Result value.
    """

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



def answer_from_context(question: str, retrieved_chunks: list[RetrievedChunk]) -> TeachingResponse:
    """Answer a question using only the provided context.
    
    Args:
        question (str): The user question to answer.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        TeachingResponse: Result value.
    """

    if not retrieved_chunks:
        return TeachingResponse(
            answer="I couldn't find relevant support in the uploaded notes for that question yet.",
            citations=[],
            used_fallback=True,
        )

    chat_client = _get_chat_client()
    if chat_client is not None:
        prompt_bundle = build_answer_prompt(_build_context(retrieved_chunks), question)
        try:
            with _langfuse_generation(
                enabled=chat_client.enable_tracing,
                feature="answer_from_context",
                prompt_bundle=prompt_bundle,
                metadata={"question_chars": str(len(question)), "num_chunks": str(len(retrieved_chunks))},
            ) as generation:
                response = _create_chat_completion(
                    chat_client,
                    model=settings.chat_model,
                    max_tokens=settings.max_chat_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful tutor who only answers from provided class notes.",
                        },
                        {"role": "user", "content": prompt_bundle.text},
                    ],
                )
                _update_generation(generation, _extract_message_text(response), response)
            trace_id, observation_id = _extract_generation_ids(generation)
            message = _extract_message_text(response)
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
        "This is a retrieval-based fallback answer. Add API credentials to upgrade this to a fuller generated explanation."
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
) -> str | None:
    """Generate a short follow-up question to keep the learner engaged.
    
    Args:
        question (str): The user question to answer.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
        answer (str): Input parameter.
        used_fallback (bool): Input parameter.
    
    Returns:
        str | None: Result value.
    """

    if used_fallback:
        return None

    chat_client = _get_chat_client()
    if chat_client is None:
        return None

    prompt_bundle = build_follow_up_prompt(
        question=question,
        answer=answer,
        context=_build_context(retrieved_chunks),
        used_fallback=used_fallback,
    )
    try:
        with _langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="follow_up",
            prompt_bundle=prompt_bundle,
            metadata={"question_chars": str(len(question)), "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = _create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=120,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You generate short, engaging follow-up questions for learners.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            output_text = _extract_message_text(response)
            _update_generation(generation, output_text, response)
        payload = json.loads(_clean_json(output_text))
        follow_up = str(payload.get("follow_up", "")).strip()
        if follow_up:
            return _normalize_follow_up(follow_up)
        return _fallback_follow_up(question, used_fallback)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return _fallback_follow_up(question, used_fallback)


def generate_structured_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    persona_name: str,
    example: str,

    chat_history: str = "",
) -> StructuredAnswer:
    """Generate structured answer.
    
    Args:
        question (str): The user question to answer.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
        persona_name (str): Input parameter.
        example (str): Input parameter.
        chat_history (str): Input parameter.
    
    Returns:
        StructuredAnswer: Result value.
    """

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

    chat_client = _get_chat_client()
    if chat_client is None:
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
        context=_build_context(retrieved_chunks),
        question=question,
        persona_name=persona_name,
        example=example,
        chat_history=chat_history,
    )
    try:
        with _langfuse_generation(
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
                    response = _create_chat_completion(
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
                    output_text = _extract_message_text(response)
                    payload = _parse_json_payload(output_text)
                    break
                except (APIConnectionError, APIStatusError, APITimeoutError):
                    # Try the next response format strategy.
                    continue
                except (ValueError, json.JSONDecodeError):
                    # Try to repair malformed JSON once before falling back.
                    payload = _repair_structured_answer_payload(
                        chat_client=chat_client,
                        prompt_text=prompt_bundle.text,
                        raw_output=output_text,
                    )
                    if payload is not None:
                        break
            if response is not None:
                _update_generation(generation, output_text, response)
        if payload is None:
            raise ValueError("Unable to parse structured answer payload")
        structured = StructuredAnswer.model_validate(payload)
        trace_id, observation_id = _extract_generation_ids(generation)
        sanitized_answer = _strip_answer_sources_block(structured.answer)
        structured.answer = _truncate_sentences(_strip_inline_citations(sanitized_answer), 4)
        _ensure_action_lanes(structured, question)
        if not structured.citations:
            structured.citations = [chunk.citation for chunk in retrieved_chunks]
        structured.trace_id = trace_id
        structured.observation_id = observation_id
        if structured.topic_subject:
            structured.topic_subject = structured.topic_subject.strip()
        return structured
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
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


def generate_quiz_from_context(
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    num_questions: int,

    weak_concepts: list[str] | None = None,
) -> StudyQuiz | None:
    """Generate a quiz from the provided context.
    
    Args:
        topic (str): Topic label for the request.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
        num_questions (int): Input parameter.
        weak_concepts (list[str] | None): Input parameter.
    
    Returns:
        StudyQuiz | None: Result value.
    """

    if not retrieved_chunks:
        return None

    chat_client = _get_chat_client()
    if chat_client is None:
        return None

    prompt_bundle = build_quiz_prompt(
        topic=topic,
        context=_build_context(retrieved_chunks),
        num_questions=num_questions,
        weak_concepts=weak_concepts,
    )
    try:
        with _langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="quiz_generation",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = _create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=settings.max_chat_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You create structured, note-grounded quiz content in JSON.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            _update_generation(generation, _extract_message_text(response), response)
        payload = json.loads(_clean_json(_extract_message_text(response)))
        return StudyQuiz.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None


def generate_remediation_from_context(
    topic: str,
    weak_concepts_with_error: str,

    retrieved_chunks: list[RetrievedChunk],
) -> ReteachResponse | None:
    """Generate a reteach explanation from the provided context.
    
    Args:
        topic (str): Topic label for the request.
        weak_concepts_with_error (str): Input parameter.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        ReteachResponse | None: Result value.
    """

    if not weak_concepts_with_error or not retrieved_chunks:
        return None

    chat_client = _get_chat_client()
    if chat_client is None:
        return None

    prompt_bundle = build_reteach_prompt(
        topic=topic,
        weak_concepts_with_error=weak_concepts_with_error,
        context=_build_context(retrieved_chunks),
    )
    try:
        with _langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="remediation",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = _create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=settings.max_chat_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You provide short, note-grounded reteaching explanations in JSON.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            output = _extract_message_text(response)
            _update_generation(generation, output, response)
            payload = json.loads(_clean_json(output))
            response_payload = ReteachResponse.model_validate(payload)
            if response_payload.mini_check_answer:
                response_payload.mini_check_answer = response_payload.mini_check_answer.strip().lower()
            return response_payload
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None


def generate_study_plan_from_context(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],

    retrieved_chunks: list[RetrievedChunk],
) -> StudyPlan | None:
    """Generate a study plan from the provided context.
    
    Args:
        topic (str): Topic label for the request.
        weak_concepts (list[str]): Input parameter.
        reviewed_concepts (list[str]): Input parameter.
        retrieved_chunks (list[RetrievedChunk]): Chunks returned from retrieval.
    
    Returns:
        StudyPlan | None: Result value.
    """

    if not retrieved_chunks:
        return None

    chat_client = _get_chat_client()
    if chat_client is None:
        return None

    prompt_bundle = build_study_plan_prompt(
        topic=topic,
        weak_concepts=weak_concepts,
        reviewed_concepts=reviewed_concepts,
        context=_build_context(retrieved_chunks),
    )
    try:
        with _langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="study_plan",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = _create_chat_completion(
                chat_client,
                model=settings.chat_model,
                max_tokens=settings.max_chat_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You create structured, note-grounded study plans in JSON.",
                    },
                    {"role": "user", "content": prompt_bundle.text},
                ],
            )
            _update_generation(generation, _extract_message_text(response), response)
        payload = json.loads(_clean_json(_extract_message_text(response)))
        return StudyPlan.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, json.JSONDecodeError, ValueError):
        return None
