"""planning.py: Generate and normalize study plan wrap-ups."""

from __future__ import annotations

from src.core.models import StudyPlan
from src.llm.llm_client import generate_study_plan_from_context
from src.notes.notes_answering import retrieve_note_chunks


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """Remove duplicates while keeping the original order.

    Args:
        items (list[str]): Items to dedupe.

    Returns:
        list[str]: Deduped list in original order.
    """
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


def _estimate_mastery_score(reviewed: list[str], weak: list[str]) -> str:
    """Estimate a mastery score when generation is unavailable.

    Args:
        reviewed (list[str]): Concepts reviewed during the session.
        weak (list[str]): Concepts still weak.

    Returns:
        str: Estimated mastery percentage as a string.
    """
    reviewed_count = max(len(reviewed), 1)  # Avoid divide-by-zero in empty sessions.
    weak_count = len(weak)
    score = max(0, int(round((1 - (weak_count / reviewed_count)) * 100)))
    return f"{score}%"


def _dedupe_casefold(items: list[str]) -> list[str]:
    """Deduplicate strings while treating case as equivalent.

    Args:
        items (list[str]): Items to dedupe.

    Returns:
        list[str]: Deduped, normalized items.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _normalize_wrap_up(
    plan: StudyPlan,
    *,
    topic: str,
    reviewed_concepts: list[str],
    weak_concepts: list[str],
) -> StudyPlan:
    """Normalize a generated study plan to ensure clarity and consistency.

    Args:
        plan (StudyPlan): Raw generated study plan.
        topic (str): Current study topic.
        reviewed_concepts (list[str]): Concepts reviewed in this session.
        weak_concepts (list[str]): Concepts still weak.

    Returns:
        StudyPlan: Cleaned study plan with fallbacks applied.
    """
    reviewed = _dedupe_casefold(reviewed_concepts) or [topic]
    weak = _dedupe_casefold(weak_concepts)
    strengths = _dedupe_casefold(plan.strengths)

    weak_set = {item.casefold() for item in weak}
    strengths = [item for item in strengths if item.casefold() not in weak_set]

    if not strengths:
        strengths = [item for item in reviewed if item.casefold() not in weak_set]

    if not strengths:
        strengths = ["More practice needed"]

    if not weak and plan.weak_areas:
        weak = _dedupe_casefold(plan.weak_areas)

    summary = plan.summary.strip() if plan.summary else ""
    if weak and (not summary or strengths == ["More practice needed"]):
        summary = (
            f"We reviewed {', '.join(reviewed)} during this session. "
            f"Focus next on {weak[0]} to solidify the core ideas."
        )

    return StudyPlan(
        topic=topic,
        reviewed_topics=reviewed,
        recommended_order=weak + [item for item in reviewed if item.casefold() not in weak_set],
        suggested_next_steps=[plan.next_step_lane.query] if plan.next_step_lane else [],
        mastery_score=plan.mastery_score or _estimate_mastery_score(reviewed, weak),
        summary=summary or f"We reviewed {', '.join(reviewed)} during this session.",
        strengths=strengths,
        weak_areas=weak,
        next_step_lane=plan.next_step_lane,
    )


def _fallback_study_plan(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],
) -> StudyPlan:
    """Build a simple study plan when generation fails.

    Args:
        topic (str): Current study topic.
        weak_concepts (list[str]): Concepts still weak.
        reviewed_concepts (list[str]): Concepts reviewed in this session.

    Returns:
        StudyPlan: Fallback study plan.
    """
    reviewed_topics = _dedupe_preserving_order(reviewed_concepts) or [topic]
    weak_areas = _dedupe_preserving_order(weak_concepts)
    strengths = [item for item in reviewed_topics if item not in weak_areas]
    weakest_area = weak_areas[0] if weak_areas else reviewed_topics[0]
    summary = (
        f"We reviewed {', '.join(reviewed_topics)} during this session. "
        f"Keep reinforcing {weakest_area} to lock in the core ideas."
    )
    if not strengths:
        strengths = ["More practice needed"]

    return StudyPlan(
        topic=topic,
        reviewed_topics=reviewed_topics,
        recommended_order=weak_areas + [item for item in reviewed_topics if item not in weak_areas],
        suggested_next_steps=[f"Review {weakest_area} with one worked example."],
        mastery_score=_estimate_mastery_score(reviewed_topics, weak_areas),
        summary=summary,
        strengths=strengths,
        weak_areas=weak_areas,
        next_step_lane={
            "button_label": f"Deep dive into {weakest_area}",
            "query": f"Explain {weakest_area} in more detail.",
        },
    )


def is_valid_study_plan(study_plan: StudyPlan, expected_topic: str) -> bool:
    """Check that a study plan is complete and on-topic.

    Args:
        study_plan (StudyPlan): Generated study plan.
        expected_topic (str): Topic that should match the plan.

    Returns:
        bool: True when the plan passes validation.
    """
    if not study_plan.mastery_score.strip():
        return False
    if not study_plan.summary.strip():
        return False
    if not study_plan.strengths:
        return False
    if any(not item.strip() for item in study_plan.strengths):
        return False
    if any(not item.strip() for item in study_plan.weak_areas):
        return False
    if study_plan.next_step_lane is None:
        return False
    if not study_plan.next_step_lane.button_label.strip():
        return False
    if not study_plan.next_step_lane.query.strip():
        return False
    return True


def build_study_plan(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str] | None = None,
) -> StudyPlan:
    """Generate a study plan grounded in retrieved notes.

    Args:
        topic (str): Current study topic.
        weak_concepts (list[str]): Concepts still weak.
        reviewed_concepts (list[str] | None): Concepts already reviewed.

    Returns:
        StudyPlan: A validated or fallback study plan.
    """
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
        return _normalize_wrap_up(
            study_plan,
            topic=topic,
            reviewed_concepts=focused_reviewed_concepts,
            weak_concepts=weak_concepts,
        )
    return _fallback_study_plan(topic, weak_concepts, focused_reviewed_concepts)
