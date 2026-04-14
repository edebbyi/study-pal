"""mastery.py: Build mastery sessions and determine next study steps."""

from __future__ import annotations

from src.core.models import MasteryProgress, MasterySession, QuizResult
from src.modes.mode_router import extract_study_topic
from src.modes.quiz import generate_reinforcement_quiz
from src.modes.remediation import generate_remediation_message
from src.notes.notes_answering import build_answer_response, get_supporting_citations


def start_mastery_session(question: str, fallback_topic: str | None = None) -> MasterySession:
    """Create a mastery session from an initial user prompt.

    Args:
        question (str): Initial mastery prompt.
        fallback_topic (str | None): Topic to use if the prompt is vague.

    Returns:
        MasterySession: Session metadata and intro message.
    """
    topic = extract_study_topic(question, fallback_topic)
    response = build_answer_response(topic)
    intro_message = response.answer
    return MasterySession(
        topic=topic,
        intro_message=intro_message,
        citations=get_supporting_citations(topic),
        trace_id=response.trace_id,
        observation_id=response.observation_id,
    )


def advance_mastery_progress(topic: str, quiz_result: QuizResult, current_round: int) -> MasteryProgress:
    """Decide the next mastery step based on quiz results.

    Args:
        topic (str): Current study topic.
        quiz_result (QuizResult): Results from the latest quiz.
        current_round (int): Current quiz round.

    Returns:
        MasteryProgress: Updated mastery progress.
    """
    if quiz_result.score == quiz_result.total:
        return MasteryProgress(
            quiz_result=quiz_result,
            remediation_message=None,
            next_quiz=None,
            next_quiz_round=None,
            status="completed",
        )

    next_round = current_round + 1
    remediation_message, remediation_payload = generate_remediation_message(
        topic,
        quiz_result.weak_concepts,
        quiz_result,
    )
    next_quiz = generate_reinforcement_quiz(topic, quiz_result.weak_concepts, next_round)
    return MasteryProgress(
        quiz_result=quiz_result,
        remediation_message=remediation_message,
        remediation_payload=remediation_payload,
        next_quiz=next_quiz,
        next_quiz_round=next_round,
        status="in_progress",
    )
