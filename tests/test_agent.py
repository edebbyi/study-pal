from src.agent import advance_mastery_loop, stop_mastery_loop
from src.models import QuizFeedback, QuizResult


def test_advance_mastery_loop_builds_plan_on_completion() -> None:
    quiz_result = QuizResult(
        score=2,
        total=2,
        weak_concepts=[],
        feedback=[
            QuizFeedback(
                question="Q1",
                user_answer="A",
                correct_answer="A",
                is_correct=True,
                concept_tag="topic",
            )
        ],
    )

    progress = advance_mastery_loop("Photosynthesis", quiz_result, current_round=1)

    assert progress.status == "completed"
    assert progress.study_plan is not None
    assert progress.study_plan.topic == "Photosynthesis"


def test_stop_mastery_loop_builds_stopped_plan() -> None:
    progress = stop_mastery_loop("Photosynthesis", ["light reactions"])

    assert progress.status == "stopped"
    assert progress.study_plan is not None
    assert progress.study_plan.weak_areas == ["light reactions"]
