from src.planning import build_study_plan


def test_build_study_plan_prioritizes_weak_concepts() -> None:
    study_plan = build_study_plan(
        "Photosynthesis",
        ["light reactions", "glucose production"],
        ["light reactions", "glucose production", "Calvin cycle"],
    )

    assert study_plan.topic == "Photosynthesis"
    assert study_plan.reviewed_topics == ["light reactions", "glucose production", "Calvin cycle"]
    assert study_plan.weak_areas == ["light reactions", "glucose production"]
    assert study_plan.recommended_order[0] == "light reactions"
    assert "Retake a short quiz" in study_plan.suggested_next_steps[2]


def test_build_study_plan_handles_perfect_mastery() -> None:
    study_plan = build_study_plan(
        "Photosynthesis",
        [],
        ["light reactions", "Calvin cycle"],
    )

    assert study_plan.weak_areas == []
    assert study_plan.recommended_order == ["light reactions", "Calvin cycle"]
    assert "spaced repetition" in study_plan.suggested_next_steps[2]
