"""agent.py: Orchestrate mastery sessions across quiz rounds."""

from __future__ import annotations

from src.core.config import settings
from src.core.models import MasteryProgress, MasterySession, QuizResult
from src.core.observability import log_langfuse_event
from src.modes.mastery import advance_mastery_progress, start_mastery_session
from src.modes.planning import build_study_plan
from src.modes.quiz import generate_quiz


def start_mastery_loop(
    question: str,
    fallback_topic: str | None = None,
    session_id: str | None = None,
) -> tuple[MasterySession, MasteryProgress]:
    """Start a mastery session and prepare the first quiz.

    Args:
        question (str): The user's initial mastery prompt.
        fallback_topic (str | None): Topic to use if the prompt is vague.
        session_id (str | None): Session identifier for observability.

    Returns:
        tuple[MasterySession, MasteryProgress]: Session details and initial progress.
    """
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
    log_langfuse_event(
        "mastery_start",
        session_id=session_id,
        metadata={
            "topic": mastery_session.topic,
            "quiz_created": bool(initial_quiz),
        },
    )
    return mastery_session, initial_progress


def advance_mastery_loop(
    topic: str,
    quiz_result: QuizResult,
    current_round: int,
    session_id: str | None = None,
) -> MasteryProgress:
    """Advance the mastery loop based on the latest quiz result.

    Args:
        topic (str): Current study topic.
        quiz_result (QuizResult): Results from the latest quiz.
        current_round (int): Current quiz round number.
        session_id (str | None): Session identifier for observability.

    Returns:
        MasteryProgress: Updated mastery progress for the session.
    """
    bounded_progress = advance_mastery_progress(topic, quiz_result, current_round)
    reviewed_concepts = [feedback.concept_tag for feedback in quiz_result.feedback]

    if bounded_progress.status == "completed":
        bounded_progress.study_plan = build_study_plan(
            topic,
            quiz_result.weak_concepts,
            reviewed_concepts,
        )
        log_langfuse_event(
            "mastery_completed",
            session_id=session_id,
            metadata={
                "topic": topic,
                "round": current_round,
                "weak_concepts": len(quiz_result.weak_concepts),
            },
        )
        return bounded_progress

    if current_round >= settings.max_quiz_rounds:  # Enforce a hard cap so sessions don't loop forever.
        bounded_progress.status = "stopped"
        bounded_progress.next_quiz = None
        bounded_progress.next_quiz_round = None
        bounded_progress.study_plan = build_study_plan(
            topic,
            quiz_result.weak_concepts,
            reviewed_concepts,
        )
        log_langfuse_event(
            "mastery_max_rounds",
            session_id=session_id,
            metadata={
                "topic": topic,
                "round": current_round,
                "max_rounds": settings.max_quiz_rounds,
            },
        )
        return bounded_progress

    log_langfuse_event(
        "mastery_advance",
        session_id=session_id,
        metadata={
            "topic": topic,
            "round": current_round,
            "weak_concepts": len(quiz_result.weak_concepts),
        },
    )
    return bounded_progress


def stop_mastery_loop(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str] | None = None,
    session_id: str | None = None,
) -> MasteryProgress:
    """Stop the mastery loop and build a wrap-up study plan.

    Args:
        topic (str): Current study topic.
        weak_concepts (list[str]): Concepts still needing practice.
        reviewed_concepts (list[str] | None): Concepts already reviewed.
        session_id (str | None): Session identifier for observability.

    Returns:
        MasteryProgress: Finalized progress with a study plan.
    """
    progress = MasteryProgress(
        quiz_result=QuizResult(score=0, total=0, weak_concepts=weak_concepts, feedback=[]),
        remediation_message=None,
        next_quiz=None,
        next_quiz_round=None,
        study_plan=build_study_plan(topic, weak_concepts, reviewed_concepts),
        status="stopped",
    )
    log_langfuse_event(
        "mastery_stopped",
        session_id=session_id,
        metadata={
            "topic": topic,
            "weak_concepts": len(weak_concepts),
        },
    )
    return progress
