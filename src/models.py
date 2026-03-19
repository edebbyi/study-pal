from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["pdf", "txt", "md"]
AppMode = Literal["ask", "mastery"]
MasteryStatus = Literal["idle", "in_progress", "completed", "stopped"]
FeedbackRating = Literal["Very helpful", "Somewhat helpful", "Not helpful"]


class Page(BaseModel):
    page_number: int
    text: str


class Document(BaseModel):
    filename: str
    session_id: str
    source_type: SourceType
    pages: list[Page]


class DocumentMetadata(BaseModel):
    document_title: str
    document_topic: str
    document_summary: str


class Chunk(BaseModel):
    id: str
    text: str
    filename: str
    page: int
    chunk_id: int
    session_id: str
    citation: str
    source_type: SourceType
    document_id: str | None = None
    document_title: str | None = None
    document_summary: str | None = None
    topic: str | None = None
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    filename: str
    page: int
    citation: str
    score: float
    chunk_id: int
    chapter: str | None = None
    topic: str | None = None


class TeachingResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class ResponseFeedback(BaseModel):
    message_id: str
    session_id: str
    document_id: str | None = None
    filename: str | None = None
    query: str
    response: str
    rating: FeedbackRating
    feedback_text: str | None = None
    topic: str | None = None
    mode: AppMode
    citations: list[str] = Field(default_factory=list)
    created_at: str


class QuizQuestion(BaseModel):
    prompt: str
    options: list[str]
    correct_answer: str
    concept_tag: str


class StudyQuiz(BaseModel):
    title: str
    topic: str
    questions: list[QuizQuestion]


class QuizFeedback(BaseModel):
    question: str
    user_answer: str | None
    correct_answer: str
    is_correct: bool
    concept_tag: str


class QuizResult(BaseModel):
    score: int
    total: int
    weak_concepts: list[str]
    feedback: list[QuizFeedback]


class StudyPlan(BaseModel):
    topic: str
    reviewed_topics: list[str]
    weak_areas: list[str]
    recommended_order: list[str]
    suggested_next_steps: list[str]


class MasterySession(BaseModel):
    topic: str
    intro_message: str
    citations: list[str] = Field(default_factory=list)
    status: MasteryStatus = "in_progress"


class MasteryProgress(BaseModel):
    quiz_result: QuizResult
    remediation_message: str | None = None
    next_quiz: StudyQuiz | None = None
    next_quiz_round: int | None = None
    study_plan: StudyPlan | None = None
    status: MasteryStatus = "in_progress"
