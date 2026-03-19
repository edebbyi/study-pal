from __future__ import annotations

from src.models import QuizFeedback, QuizResult, StudyQuiz
from src.utils import dedupe_preserve_order


def grade_quiz(quiz: StudyQuiz, answers: list[str | None]) -> QuizResult:
    score = 0
    weak_concepts: list[str] = []
    feedback: list[QuizFeedback] = []

    for question, user_answer in zip(quiz.questions, answers, strict=False):
        is_correct = user_answer == question.correct_answer
        if is_correct:
            score += 1
        else:
            weak_concepts.append(question.concept_tag)

        feedback.append(
            QuizFeedback(
                question=question.prompt,
                user_answer=user_answer,
                correct_answer=question.correct_answer,
                is_correct=is_correct,
                concept_tag=question.concept_tag,
            )
        )

    return QuizResult(
        score=score,
        total=len(quiz.questions),
        weak_concepts=dedupe_preserve_order(weak_concepts),
        feedback=feedback,
    )
