"""test_planning_generation.py: Tests for test_planning_generation.py."""

from src.core.models import StudyPlan
from src.modes.planning import _fallback_study_plan, is_valid_study_plan


def test_fallback_study_plan_prioritizes_weak_concepts() -> None:
    """Test fallback study plan prioritizes weak concepts.
    """

    study_plan = _fallback_study_plan(
        "Photosynthesis",
        ["light reactions", "glucose production"],
        ["light reactions", "Calvin cycle"],
    )

    assert study_plan.recommended_order[0] == "light reactions"
    assert study_plan.suggested_next_steps
    assert study_plan.reviewed_topics == ["light reactions", "Calvin cycle"]


def test_fallback_study_plan_handles_no_weak_concepts() -> None:
    """Test fallback study plan handles no weak concepts.
    """

    study_plan = _fallback_study_plan(
        "Photosynthesis",
        [],
        ["light reactions", "Calvin cycle"],
    )

    assert study_plan.recommended_order == ["light reactions", "Calvin cycle"]
    assert study_plan.suggested_next_steps


def test_study_plan_validator_rejects_empty_sections_and_topic_mismatch() -> None:
    """Test study plan validator rejects empty sections and topic mismatch.
    """

    invalid_plan = StudyPlan(
        topic="Cell Respiration",
        reviewed_topics=[" "],
        strengths=[""],
        weak_areas=[],
        recommended_order=[],
        suggested_next_steps=["Review it"],
    )

    assert not is_valid_study_plan(invalid_plan, "Photosynthesis")
