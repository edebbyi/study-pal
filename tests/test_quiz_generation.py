"""test_quiz_generation.py: Tests for test_quiz_generation.py."""

from src.core.config import settings
from src.modes.quiz import _fallback_quiz, _sanitize_study_quiz, is_valid_study_quiz
from src.core.models import QuizQuestion, StudyQuiz


def test_fallback_quiz_has_expected_question_count_and_matching_answers() -> None:
    """Test fallback quiz has expected question count and matching answers.
    """

    quiz = _fallback_quiz("Photosynthesis", title_prefix="Checkpoint Quiz")

    assert quiz.title == "Checkpoint Quiz: Photosynthesis"
    assert len(quiz.questions) == settings.quiz_questions_per_round
    assert all(question.correct_answer in question.options for question in quiz.questions)


def test_reinforcement_fallback_quiz_uses_weak_concept() -> None:
    """Test reinforcement fallback quiz uses weak concept.
    """

    quiz = _fallback_quiz(
        "Photosynthesis",
        title_prefix="Reinforcement Quiz",
        weak_concepts=["light reactions"],
        quiz_round=2,
    )

    assert quiz.title == "Reinforcement Quiz: Photosynthesis"
    assert "light reactions" in quiz.questions[0].prompt.lower()
    assert "round 2" in quiz.questions[1].prompt
    assert "light reactions" in quiz.questions[2].concept_tag.lower() or "photosynthesis" in quiz.questions[2].concept_tag.lower()


def test_generated_quiz_validator_rejects_mismatched_topic_and_bad_answers() -> None:
    """Test generated quiz validator rejects mismatched topic and bad answers.
    """

    invalid_quiz = StudyQuiz(
        title="Quiz",
        topic="Cell Respiration",
        questions=[
            QuizQuestion(
                prompt="Question",
                options=["A", "B"],
                correct_answer="C",
                concept_tag="concept",
            ),
            QuizQuestion(
                prompt="Question 2",
                options=["A", "B"],
                correct_answer="A",
                concept_tag="concept 2",
            ),
        ],
    )

    assert not is_valid_study_quiz(invalid_quiz, "Photosynthesis", settings.quiz_questions_per_round)


def test_sanitize_study_quiz_humanizes_concept_tags_and_generic_topics() -> None:
    """Test sanitize study quiz humanizes concept tags and generic topics.
    """

    generated_quiz = StudyQuiz(
        title="Generate A Quiz Please",
        topic="Generate A Quiz Please",
        questions=[
            QuizQuestion(
                prompt="Question 1",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                concept_tag="animal_models",
            ),
            QuizQuestion(
                prompt="Question 2",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                concept_tag="brain_power_usage",
            ),
            QuizQuestion(
                prompt="Question 3",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                concept_tag="medulla_function",
            ),
        ],
    )

    sanitized_quiz = _sanitize_study_quiz(
        generated_quiz,
        expected_topic="Medulla",
        title_prefix="Checkpoint Quiz",
    )

    assert sanitized_quiz.topic == "Medulla"
    assert sanitized_quiz.questions[0].concept_tag == "Animal models"
    assert sanitized_quiz.questions[1].concept_tag == "Brain power usage"
