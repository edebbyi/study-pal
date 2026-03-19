from __future__ import annotations

from src.mode_router import extract_study_topic
from src.models import MasteryProgress, MasterySession, QuizResult
from src.notes_answering import build_answer_message, get_supporting_citations
from src.quiz import generate_quiz, generate_reinforcement_quiz
from src.remediation import generate_remediation_message


def start_mastery_session(question: str, fallback_topic: str | None = None) -> MasterySession:
    topic = extract_study_topic(question, fallback_topic)
    intro_message = build_answer_message(topic)
    return MasterySession(
        topic=topic,
        intro_message=intro_message,
        citations=get_supporting_citations(topic),
    )

def advance_mastery_progress(topic: str, quiz_result: QuizResult, current_round: int) -> MasteryProgress:
    if quiz_result.score == quiz_result.total:
        return MasteryProgress(
            quiz_result=quiz_result,
            remediation_message=None,
            next_quiz=None,
            next_quiz_round=None,
            status="completed",
        )

    next_round = current_round + 1
    remediation_message = generate_remediation_message(topic, quiz_result.weak_concepts)
    next_quiz = generate_reinforcement_quiz(topic, quiz_result.weak_concepts, next_round)
    return MasteryProgress(
        quiz_result=quiz_result,
        remediation_message=remediation_message,
        next_quiz=next_quiz,
        next_quiz_round=next_round,
        status="in_progress",
    )
