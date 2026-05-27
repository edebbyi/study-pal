"""learning_generation.py: Quiz/remediation/study-plan generation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.config import settings
from src.core.models import RetrievedChunk, ReteachResponse, StudyPlan, StudyQuiz
from src.llm.prompts import build_quiz_prompt, build_reteach_prompt, build_study_plan_prompt


@dataclass(frozen=True)
class LearningGenerationDeps:
    """Injected dependencies for learning-mode generators."""

    get_chat_client: Callable[[str | None], Any]
    create_chat_completion: Callable[..., Any]
    langfuse_generation: Callable[..., Any]
    update_generation: Callable[[object, str, object], None]
    extract_message_text: Callable[[Any], str]
    parse_json_payload: Callable[[str], dict[str, object]]
    build_context: Callable[[list[RetrievedChunk]], str]


def generate_quiz_from_context(
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    num_questions: int,
    *,
    weak_concepts: list[str] | None = None,
    deps: LearningGenerationDeps,
) -> StudyQuiz | None:
    """Generate a quiz from grounded context."""
    if not retrieved_chunks:
        return None

    chat_client = deps.get_chat_client(None)
    if chat_client is None:
        return None

    prompt_bundle = build_quiz_prompt(
        topic=topic,
        context=deps.build_context(retrieved_chunks),
        num_questions=num_questions,
        weak_concepts=weak_concepts,
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="quiz_generation",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = deps.create_chat_completion(
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
            output_text = deps.extract_message_text(response)
            deps.update_generation(generation, output_text, response)
        payload = deps.parse_json_payload(output_text)
        return StudyQuiz.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError):
        return None


def generate_remediation_from_context(
    topic: str,
    weak_concepts_with_error: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    deps: LearningGenerationDeps,
) -> ReteachResponse | None:
    """Generate a reteach explanation from grounded context."""
    if not weak_concepts_with_error or not retrieved_chunks:
        return None

    chat_client = deps.get_chat_client(None)
    if chat_client is None:
        return None

    prompt_bundle = build_reteach_prompt(
        topic=topic,
        weak_concepts_with_error=weak_concepts_with_error,
        context=deps.build_context(retrieved_chunks),
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="remediation",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = deps.create_chat_completion(
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
            output = deps.extract_message_text(response)
            deps.update_generation(generation, output, response)
            payload = deps.parse_json_payload(output)
            response_payload = ReteachResponse.model_validate(payload)
            if response_payload.mini_check_answer:
                response_payload.mini_check_answer = response_payload.mini_check_answer.strip().lower()
            return response_payload
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError):
        return None


def generate_study_plan_from_context(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],
    retrieved_chunks: list[RetrievedChunk],
    *,
    deps: LearningGenerationDeps,
) -> StudyPlan | None:
    """Generate a study plan from grounded context."""
    if not retrieved_chunks:
        return None

    chat_client = deps.get_chat_client(None)
    if chat_client is None:
        return None

    prompt_bundle = build_study_plan_prompt(
        topic=topic,
        weak_concepts=weak_concepts,
        reviewed_concepts=reviewed_concepts,
        context=deps.build_context(retrieved_chunks),
    )
    try:
        with deps.langfuse_generation(
            enabled=chat_client.enable_tracing,
            feature="study_plan",
            prompt_bundle=prompt_bundle,
            metadata={"topic": topic, "num_chunks": str(len(retrieved_chunks))},
        ) as generation:
            response = deps.create_chat_completion(
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
            output_text = deps.extract_message_text(response)
            deps.update_generation(generation, output_text, response)
        payload = deps.parse_json_payload(output_text)
        return StudyPlan.model_validate(payload)
    except (APIConnectionError, APIStatusError, APITimeoutError, ValueError):
        return None
