from __future__ import annotations

from src.llm_client import generate_remediation_from_context
from src.notes_answering import retrieve_note_chunks
from src.utils import humanize_label


def _fallback_remediation(topic: str, weak_concepts: list[str]) -> str:
    concept_list = ", ".join(humanize_label(concept) for concept in weak_concepts)
    return (
        f"Let's reteach the parts of {humanize_label(topic)} that still look shaky: {concept_list}.\n\n"
        "Focus on the main idea, connect it back to the notes, and try the reinforcement quiz next."
    )


def is_valid_remediation_message(message: str | None) -> bool:
    if message is None:
        return False
    normalized_message = " ".join(message.split())
    return len(normalized_message) >= 40


def generate_remediation_message(topic: str, weak_concepts: list[str]) -> str:
    if not weak_concepts:
        return f"You answered everything correctly for {humanize_label(topic)}. You're ready to move on."

    cleaned_weak_concepts = [humanize_label(concept) for concept in weak_concepts]
    query = f"{topic} {' '.join(cleaned_weak_concepts)}".strip()
    retrieved_chunks = retrieve_note_chunks(query)
    remediation = generate_remediation_from_context(topic, cleaned_weak_concepts, retrieved_chunks)
    if is_valid_remediation_message(remediation):
        return remediation
    return _fallback_remediation(topic, cleaned_weak_concepts)
