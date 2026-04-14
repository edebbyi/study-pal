"""test_grading.py: Tests for test_grading.py."""

from src.mastery import advance_mastery_progress
from src.grading import grade_quiz
from src.models import QuizQuestion, StudyQuiz


def test_grade_quiz_scores_and_collects_weak_concepts() -> None:
    """Test grade quiz scores and collects weak concepts.
    """

    quiz = StudyQuiz(
        title="Checkpoint Quiz: Photosynthesis",
        topic="Photosynthesis",
        questions=[
            QuizQuestion(
                prompt="What powers photosynthesis?",
                options=["Light energy", "Sound waves"],
                correct_answer="Light energy",
                concept_tag="energy source",
            ),
            QuizQuestion(
                prompt="What molecule do plants make?",
                options=["Glucose", "Helium"],
                correct_answer="Glucose",
                concept_tag="glucose production",
            ),
        ],
    )

    result = grade_quiz(quiz, ["Light energy", "Helium"])

    assert result.score == 1
    assert result.total == 2
    assert result.weak_concepts == ["glucose production"]
    assert result.feedback[1].is_correct is False


def test_advance_mastery_progress_builds_reinforcement_round() -> None:
    """Test advance mastery progress builds reinforcement round.
    """

    quiz = StudyQuiz(
        title="Checkpoint Quiz: Photosynthesis",
        topic="Photosynthesis",
        questions=[
            QuizQuestion(
                prompt="What powers photosynthesis?",
                options=["Light energy", "Sound waves"],
                correct_answer="Light energy",
                concept_tag="energy source",
            ),
            QuizQuestion(
                prompt="What molecule do plants make?",
                options=["Glucose", "Helium"],
                correct_answer="Glucose",
                concept_tag="glucose production",
            ),
        ],
    )

    result = grade_quiz(quiz, ["Light energy", "Helium"])
    progress = advance_mastery_progress("Photosynthesis", result, current_round=1)

    assert progress.status == "in_progress"
    assert progress.remediation_message is not None
    assert progress.next_quiz is not None
    assert progress.next_quiz_round == 2
    assert progress.next_quiz.title == "Reinforcement Quiz: Photosynthesis"


def test_advance_mastery_progress_marks_completion_on_perfect_score() -> None:
    """Test advance mastery progress marks completion on perfect score.
    """

    quiz = StudyQuiz(
        title="Checkpoint Quiz: Photosynthesis",
        topic="Photosynthesis",
        questions=[
            QuizQuestion(
                prompt="What powers photosynthesis?",
                options=["Light energy", "Sound waves"],
                correct_answer="Light energy",
                concept_tag="energy source",
            )
        ],
    )

    result = grade_quiz(quiz, ["Light energy"])
    progress = advance_mastery_progress("Photosynthesis", result, current_round=1)

    assert progress.status == "completed"
    assert progress.remediation_message is None
    assert progress.next_quiz is None
    assert progress.next_quiz_round is None
