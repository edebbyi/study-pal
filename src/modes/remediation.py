"""remediation.py: Generate reteach guidance for weak concepts."""

from __future__ import annotations

from src.core.models import QuizResult
from src.core.utils import humanize_label
from src.llm.llm_client import generate_remediation_from_context
from src.notes.notes_answering import retrieve_note_chunks


def _fallback_remediation(topic: str, weak_concepts: list[str]) -> str:
    """Create a simple reteach message when generation is unavailable.

    Args:
        topic (str): Current study topic.
        weak_concepts (list[str]): Concepts still weak.

    Returns:
        str: A short reteach message.
    """
    concept_list = ", ".join(humanize_label(concept) for concept in weak_concepts)
    return (
        f"Let's reteach the parts of {humanize_label(topic)} that still look shaky: {concept_list}.\n\n"
        "Focus on the main idea, connect it back to the notes, and try the reinforcement quiz next."
    )


def is_valid_remediation_message(message: str | None) -> bool:
    """Check whether a remediation message is long enough to be useful.

    Args:
        message (str | None): Message to validate.

    Returns:
        bool: True when the message is detailed enough.
    """
    if message is None:
        return False
    normalized_message = " ".join(message.split())
    return len(normalized_message) >= 40


def _build_error_context(
    weak_concepts: list[str],
    quiz_result: QuizResult | None,
) -> str:
    """Build a detailed error context for reteach prompts.

    Args:
        weak_concepts (list[str]): Weak concept tags.
        quiz_result (QuizResult | None): Latest quiz result, if available.

    Returns:
        str: Error context that captures misunderstandings.
    """
    if quiz_result is None:
        return ", ".join(weak_concepts)

    lines: list[str] = []
    weak_set = {concept.casefold() for concept in weak_concepts}
    for feedback in quiz_result.feedback:
        if feedback.is_correct:
            continue
        if feedback.concept_tag.casefold() not in weak_set:
            continue
        user_answer = feedback.user_answer or "No answer"
        lines.append(
            f"Concept: {feedback.concept_tag}. User thought: '{user_answer}'. "
            f"Correct fact: '{feedback.correct_answer}'."
        )
    if not lines:
        return ", ".join(weak_concepts)
    return "\n".join(lines)


def generate_remediation_message(
    topic: str,
    weak_concepts: list[str],
    quiz_result: QuizResult | None = None,
) -> tuple[str, dict[str, str] | None]:
    """Generate a short reteach message for weak concepts.

    Args:
        topic (str): Current study topic.
        weak_concepts (list[str]): Concepts still weak.
        quiz_result (QuizResult | None): Latest quiz result, if available.

    Returns:
        tuple[str, dict[str, str] | None]: Reteach message and optional payload.
    """
    if not weak_concepts:
        return (
            f"You answered everything correctly for {humanize_label(topic)}. You're ready to move on.",
            None,
        )

    cleaned_weak_concepts = [humanize_label(concept) for concept in weak_concepts]
    query = f"{topic} {' '.join(cleaned_weak_concepts)}".strip()
    retrieved_chunks = retrieve_note_chunks(query)
    weak_concepts_with_error = _build_error_context(cleaned_weak_concepts, quiz_result)  # Surface misconceptions.
    remediation = generate_remediation_from_context(topic, weak_concepts_with_error, retrieved_chunks)
    if remediation is not None:
        parts: list[str] = []
        if remediation.concept:
            parts.append(f"**{remediation.concept.strip()}**")
        if remediation.contrast:
            parts.append(remediation.contrast.strip())
        if remediation.explanation:
            parts.append(remediation.explanation.strip())
        if remediation.mini_check:
            parts.append(f"Mini-check: {remediation.mini_check.strip()}")
        explanation = "\n\n".join(part for part in parts if part)
        if is_valid_remediation_message(explanation):
            payload = remediation.model_dump()
            return explanation, payload
    return _fallback_remediation(topic, cleaned_weak_concepts), None
