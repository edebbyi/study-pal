from __future__ import annotations

from src.config import settings
from src.mastery import (
    advance_mastery_progress,
    start_mastery_session,
)
from src.models import MasteryProgress, MasterySession, QuizResult
from src.planning import build_study_plan
from src.quiz import generate_quiz


def start_mastery_loop(
    question: str,
    fallback_topic: str | None = None,
) -> tuple[MasterySession, MasteryProgress]:
    mastery_session = start_mastery_session(question, fallback_topic)
    initial_quiz = generate_quiz(mastery_session.topic)
    initial_progress = MasteryProgress(
        quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
        remediation_message=None,
        next_quiz=initial_quiz,
        next_quiz_round=1,
        study_plan=None,
        status="in_progress",
    )
    return mastery_session, initial_progress


def advance_mastery_loop(
    topic: str,
    quiz_result: QuizResult,
    current_round: int,
) -> MasteryProgress:
    bounded_progress = advance_mastery_progress(topic, quiz_result, current_round)
    reviewed_concepts = [feedback.concept_tag for feedback in quiz_result.feedback]

    if bounded_progress.status == "completed":
        bounded_progress.study_plan = build_study_plan(
            topic,
            quiz_result.weak_concepts,
            reviewed_concepts,
        )
        return bounded_progress

    if current_round >= settings.max_quiz_rounds:
        bounded_progress.status = "stopped"
        bounded_progress.next_quiz = None
        bounded_progress.next_quiz_round = None
        bounded_progress.study_plan = build_study_plan(
            topic,
            quiz_result.weak_concepts,
            reviewed_concepts,
        )
        return bounded_progress

    return bounded_progress


def stop_mastery_loop(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str] | None = None,
) -> MasteryProgress:
    return MasteryProgress(
        quiz_result=QuizResult(score=0, total=0, weak_concepts=weak_concepts, feedback=[]),
        remediation_message=None,
        next_quiz=None,
        next_quiz_round=None,
        study_plan=build_study_plan(topic, weak_concepts, reviewed_concepts),
        status="stopped",
    )
