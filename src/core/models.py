"""models.py: Pydantic models and shared types for Study Pal."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["pdf", "txt", "md"]
AppMode = Literal["ask", "mastery"]
MasteryStatus = Literal["idle", "in_progress", "completed", "stopped"]
FeedbackRating = Literal["Very helpful", "Somewhat helpful", "Not helpful"]


class Page(BaseModel):
    """Represent a single document page."""

    page_number: int
    text: str


class Document(BaseModel):
    """Represent an ingested document and its pages."""

    filename: str
    session_id: str
    source_type: SourceType
    pages: list[Page]


class DocumentMetadata(BaseModel):
    """Store extracted metadata for a document."""

    document_title: str
    document_topic: str
    document_summary: str
    key_hooks: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """Represent a chunk of text with source metadata."""

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
    """Represent a retrieved chunk and its relevance score."""

    text: str
    filename: str
    page: int
    citation: str
    score: float
    chunk_id: int
    chapter: str | None = None
    topic: str | None = None


class TeachingResponse(BaseModel):
    """Capture a grounded answer and its metadata."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    follow_up: str | None = None
    trace_id: str | None = None
    observation_id: str | None = None


class InfoLane(BaseModel):
    """Describe the info lane button payload."""

    button_label: str
    query: str


class QuizLane(BaseModel):
    """Describe the quiz lane button payload."""

    button_label: str
    intent: str = "START_QUIZ_LOOP"


class StructuredAnswer(BaseModel):
    """Return a structured answer with action lanes."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    info_lane: InfoLane | None = None
    quiz_lane: QuizLane | None = None
    used_fallback: bool = False
    trace_id: str | None = None
    observation_id: str | None = None
    topic_subject: str | None = None


class ReteachResponse(BaseModel):
    """Capture reteach response fields from the model."""

    concept: str | None = None
    explanation: str
    contrast: str | None = None
    mini_check: str | None = None
    mini_check_answer: str | None = None


class ResponseFeedback(BaseModel):
    """Persist a user feedback record for an answer."""

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
    """Represent a quiz question and its choices."""

    prompt: str
    options: list[str]
    correct_answer: str
    concept_tag: str


class StudyQuiz(BaseModel):
    """Represent a quiz with multiple questions."""

    title: str
    topic: str
    questions: list[QuizQuestion]


class QuizFeedback(BaseModel):
    """Record feedback for a single quiz question."""

    question: str
    user_answer: str | None
    correct_answer: str
    is_correct: bool
    concept_tag: str


class QuizResult(BaseModel):
    """Summarize the score and weak concepts for a quiz."""

    score: int
    total: int
    weak_concepts: list[str]
    feedback: list[QuizFeedback]


class StudyPlan(BaseModel):
    """Summarize a session wrap-up and next step."""

    mastery_score: str
    summary: str
    strengths: list[str]
    weak_areas: list[str]
    next_step_lane: InfoLane | None = None


class MasterySession(BaseModel):
    """Represent a mastery session kickoff and intro message."""

    topic: str
    intro_message: str
    citations: list[str] = Field(default_factory=list)
    status: MasteryStatus = "in_progress"
    trace_id: str | None = None
    observation_id: str | None = None


class MasteryProgress(BaseModel):
    """Track progress through a mastery loop."""

    quiz_result: QuizResult
    remediation_message: str | None = None
    remediation_payload: dict[str, str] | None = None
    next_quiz: StudyQuiz | None = None
    next_quiz_round: int | None = None
    study_plan: StudyPlan | None = None
    status: MasteryStatus = "in_progress"
