from __future__ import annotations

import streamlit as st

from src.config import settings
from src.llm_client import generate_quiz_from_context
from src.mode_router import is_generic_mastery_topic
from src.models import QuizQuestion, StudyQuiz
from src.retrieval import retrieve_chunks
from src.utils import humanize_label


default_quiz_size = settings.quiz_questions_per_round
minimum_options_per_question = 2


def _get_session_chunks():
    try:
        return st.session_state.chunks
    except (AttributeError, KeyError):
        return []


def _get_session_id() -> str:
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
                    f"An unrelated definition from another topic",
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
    return (
        bool(question.prompt.strip())
        and bool(question.concept_tag.strip())
        and len(question.options) >= minimum_options_per_question
        and all(option.strip() for option in question.options)
        and question.correct_answer.strip() in question.options
    )


def is_valid_study_quiz(quiz: StudyQuiz, expected_topic: str, num_questions: int) -> bool:
    if quiz.topic.strip() != humanize_label(expected_topic):
        return False
    if not quiz.title.strip():
        return False
    if len(quiz.questions) != num_questions:
        return False
    return all(_has_valid_question_shape(question) for question in quiz.questions)


def generate_quiz(topic: str, num_questions: int = default_quiz_size) -> StudyQuiz:
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
    return _fallback_quiz(topic, title_prefix="Checkpoint Quiz")


def generate_reinforcement_quiz(
    topic: str,
    weak_concepts: list[str],
    quiz_round: int,
    num_questions: int = default_quiz_size,
) -> StudyQuiz:
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
