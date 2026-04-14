"""quiz.py: Generate quizzes and sanitize quiz payloads."""

from __future__ import annotations

import streamlit as st

from src.core.config import settings
from src.core.models import QuizQuestion, StudyQuiz
from src.core.utils import humanize_label
from src.data.retrieval import retrieve_chunks
from src.llm.llm_client import generate_quiz_from_context
from src.modes.mode_router import is_generic_mastery_topic


default_quiz_size = settings.quiz_questions_per_round
minimum_options_per_question = 2


def _get_session_chunks() -> list[object]:
    """Read the current session's chunks safely.

    Returns:
        list[object]: Session chunks or an empty list.
    """
    try:
        return st.session_state.chunks
    except (AttributeError, KeyError):
        return []


def _get_session_id() -> str:
    """Read the current session ID safely.

    Returns:
        str: Session identifier or empty string.
    """
    try:
        return st.session_state.session_id
    except (AttributeError, KeyError):
        return ""


def _fallback_quiz(
    topic: str,
    *,
    title_prefix: str,
    weak_concepts: list[str] | None = None,
    quiz_round: int | None = None,
) -> StudyQuiz:
    """Build a safe fallback quiz when generation fails.

    Args:
        topic (str): Quiz topic.
        title_prefix (str): Prefix to use for the quiz title.
        weak_concepts (list[str] | None): Weak concepts to emphasize.
        quiz_round (int | None): Current round for labeling.

    Returns:
        StudyQuiz: A simple quiz with generic questions.
    """
    display_topic = humanize_label(topic)
    cleaned_weak_concepts = [humanize_label(concept) for concept in weak_concepts or []]
    reinforcement_focus = cleaned_weak_concepts[0] if cleaned_weak_concepts else display_topic
    secondary_prompt = (
        f"Which review move is best for strengthening {display_topic} in round {quiz_round}?"
        if quiz_round
        else f"What should you focus on first when reviewing {display_topic}?"
    )
    secondary_correct = (
        f"Revisit the weak concept and connect it back to {display_topic}"
        if quiz_round
        else f"The main concept and vocabulary of {display_topic}"
    )
    tertiary_focus = cleaned_weak_concepts[1] if len(cleaned_weak_concepts) > 1 else display_topic
    tertiary_prompt = (
        f"Which follow-up question would best check your understanding of {tertiary_focus}?"
    )
    tertiary_correct = (
        f"Ask how {tertiary_focus} connects to the broader idea of {display_topic}"
    )

    return StudyQuiz(
        title=f"{title_prefix}: {display_topic}",
        topic=display_topic,
        questions=[
            QuizQuestion(
                prompt=f"Which option best matches the core idea of {reinforcement_focus}?",
                options=[
                    f"A high-level explanation of {reinforcement_focus}",
                    "An unrelated definition from another topic",
                    "A random memorization fact with no context",
                    "A formatting instruction instead of content",
                ],
                correct_answer=f"A high-level explanation of {reinforcement_focus}",
                concept_tag=reinforcement_focus,
            ),
            QuizQuestion(
                prompt=secondary_prompt,
                options=[
                    secondary_correct,
                    "Skip directly to a new topic",
                    "Ignore the missed questions",
                    "Memorize answer positions instead of ideas",
                ],
                correct_answer=secondary_correct,
                concept_tag=display_topic,
            ),
            QuizQuestion(
                prompt=tertiary_prompt,
                options=[
                    tertiary_correct,
                    "Switch to an unrelated chapter before checking comprehension",
                    "Judge understanding only by speed, not accuracy",
                    "Memorize isolated words without linking them to the topic",
                ],
                correct_answer=tertiary_correct,
                concept_tag=tertiary_focus,
            ),
        ],
    )


def _normalize_topic_label(raw_topic: str, expected_topic: str) -> str:
    """Normalize a quiz topic label to match the expected topic.

    Args:
        raw_topic (str): Raw topic returned by the model.
        expected_topic (str): Expected topic label.

    Returns:
        str: Normalized topic label.
    """
    cleaned_topic = humanize_label(raw_topic)
    if is_generic_mastery_topic(cleaned_topic):
        return humanize_label(expected_topic)
    return cleaned_topic


def _sanitize_study_quiz(
    quiz: StudyQuiz,
    *,
    expected_topic: str,
    title_prefix: str,
) -> StudyQuiz:
    """Clean up quiz fields for consistency and display.

    Args:
        quiz (StudyQuiz): Quiz returned by the model.
        expected_topic (str): Expected topic label.
        title_prefix (str): Title prefix to use when titles are generic.

    Returns:
        StudyQuiz: Sanitized quiz ready for display.
    """
    normalized_topic = _normalize_topic_label(quiz.topic, expected_topic)
    normalized_title = humanize_label(quiz.title)
    if not normalized_title or is_generic_mastery_topic(normalized_title):
        normalized_title = f"{title_prefix}: {normalized_topic}"

    return StudyQuiz(
        title=normalized_title,
        topic=normalized_topic,
        questions=[
            QuizQuestion(
                prompt=question.prompt.strip(),
                options=[option.strip() for option in question.options],
                correct_answer=question.correct_answer.strip(),
                concept_tag=humanize_label(question.concept_tag),
            )
            for question in quiz.questions
        ],
    )


def _has_valid_question_shape(question: QuizQuestion) -> bool:
    """Check that a quiz question has the required shape.

    Args:
        question (QuizQuestion): Question to validate.

    Returns:
        bool: True when the question is valid.
    """
    return (
        bool(question.prompt.strip())
        and bool(question.concept_tag.strip())
        and len(question.options) >= minimum_options_per_question
        and all(option.strip() for option in question.options)
        and question.correct_answer.strip() in question.options
    )


def is_valid_study_quiz(quiz: StudyQuiz, expected_topic: str, num_questions: int) -> bool:
    """Validate that a quiz is complete and on-topic.

    Args:
        quiz (StudyQuiz): Quiz to validate.
        expected_topic (str): Expected topic label.
        num_questions (int): Required number of questions.

    Returns:
        bool: True when the quiz is valid.
    """
    if quiz.topic.strip() != humanize_label(expected_topic):
        return False
    if not quiz.title.strip():
        return False
    if len(quiz.questions) != num_questions:
        return False
    return all(_has_valid_question_shape(question) for question in quiz.questions)


def generate_quiz(topic: str, num_questions: int = default_quiz_size) -> StudyQuiz:
    """Generate a checkpoint quiz for a topic.

    Args:
        topic (str): Quiz topic.
        num_questions (int): Number of questions to generate.

    Returns:
        StudyQuiz: Generated or fallback quiz.
    """
    session_chunks = _get_session_chunks()
    session_id = _get_session_id()
    retrieved_chunks = retrieve_chunks(
        question=topic,
        chunks=session_chunks,
        session_id=session_id,
    )
    quiz = generate_quiz_from_context(
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        num_questions=num_questions,
    )
    if quiz is not None:
        sanitized_quiz = _sanitize_study_quiz(
            quiz,
            expected_topic=topic,
            title_prefix="Checkpoint Quiz",
        )
        if is_valid_study_quiz(sanitized_quiz, topic, num_questions):
            return sanitized_quiz
    return _fallback_quiz(topic, title_prefix="Checkpoint Quiz")  # Keep the quiz flow usable on model failure.


def generate_reinforcement_quiz(
    topic: str,
    weak_concepts: list[str],
    quiz_round: int,
    num_questions: int = default_quiz_size,
) -> StudyQuiz:
    """Generate a reinforcement quiz focused on weak concepts.

    Args:
        topic (str): Current topic.
        weak_concepts (list[str]): Concepts to reinforce.
        quiz_round (int): Current round number.
        num_questions (int): Number of questions to generate.

    Returns:
        StudyQuiz: Generated or fallback quiz.
    """
    cleaned_weak_concepts = [humanize_label(concept) for concept in weak_concepts]
    query = f"{topic} {' '.join(cleaned_weak_concepts)}".strip()
    session_chunks = _get_session_chunks()
    session_id = _get_session_id()
    retrieved_chunks = retrieve_chunks(
        question=query,
        chunks=session_chunks,
        session_id=session_id,
    )
    quiz = generate_quiz_from_context(
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        num_questions=num_questions,
        weak_concepts=cleaned_weak_concepts,
    )
    if quiz is not None:
        sanitized_quiz = _sanitize_study_quiz(
            quiz,
            expected_topic=topic,
            title_prefix="Reinforcement Quiz",
        )
        if is_valid_study_quiz(sanitized_quiz, topic, num_questions):
            return sanitized_quiz
    return _fallback_quiz(
        topic,
        title_prefix="Reinforcement Quiz",
        weak_concepts=cleaned_weak_concepts,
        quiz_round=quiz_round,
    )
