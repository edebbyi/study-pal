from __future__ import annotations

from src.llm_client import generate_study_plan_from_context
from src.models import StudyPlan
from src.notes_answering import retrieve_note_chunks


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped_items: list[str] = []
    for item in items:
        normalized_item = item.strip()
        if not normalized_item:
            continue
        key = normalized_item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped_items.append(normalized_item)
    return deduped_items


def _fallback_study_plan(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],
) -> StudyPlan:
    reviewed_topics = _dedupe_preserving_order(reviewed_concepts) or [topic]
    weak_areas = weak_concepts

    if weak_areas:
        recommended_order = _dedupe_preserving_order(weak_areas + reviewed_topics)
        suggested_next_steps = [
            f"Review the weak concepts first: {', '.join(weak_areas)}.",
            f"Reconnect those concepts to the larger topic of {topic}.",
            "Retake a short quiz after reviewing the weak areas.",
        ]
    else:
        recommended_order = reviewed_topics
        suggested_next_steps = [
            f"Do one final summary review of {reviewed_topics[0]}.",
            f"Explain how {reviewed_topics[0]} connects to the other reviewed ideas without looking at your notes.",
            "Come back later for a spaced repetition check on these same concepts.",
        ]

    return StudyPlan(
        topic=topic,
        reviewed_topics=reviewed_topics,
        weak_areas=weak_areas,
        recommended_order=recommended_order,
        suggested_next_steps=suggested_next_steps,
    )


def is_valid_study_plan(study_plan: StudyPlan, expected_topic: str) -> bool:
    if study_plan.topic.strip() != expected_topic.strip():
        return False
    if not study_plan.reviewed_topics:
        return False
    if not study_plan.recommended_order:
        return False
    if not study_plan.suggested_next_steps:
        return False
    if any(not item.strip() for item in study_plan.reviewed_topics):
        return False
    if any(not item.strip() for item in study_plan.recommended_order):
        return False
    if any(not item.strip() for item in study_plan.suggested_next_steps):
        return False
    return True


def build_study_plan(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str] | None = None,
) -> StudyPlan:
    focused_reviewed_concepts = _dedupe_preserving_order(reviewed_concepts or [])
    query_parts = focused_reviewed_concepts or [topic]
    if weak_concepts:
        query_parts.extend(weak_concepts)
    query = " ".join(query_parts).strip()
    retrieved_chunks = retrieve_note_chunks(query)
    study_plan = generate_study_plan_from_context(
        topic,
        weak_concepts,
        focused_reviewed_concepts,
        retrieved_chunks,
    )
    if study_plan is not None and is_valid_study_plan(study_plan, topic):
        return study_plan
    return _fallback_study_plan(topic, weak_concepts, focused_reviewed_concepts)
