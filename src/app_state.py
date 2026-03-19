from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import streamlit as st

from src.models import AppMode, Chunk, MasterySession, MasteryStatus, QuizResult, StudyPlan, StudyQuiz
from src.utils import generate_message_id, generate_session_id


def _empty_workspace_state() -> dict[str, object]:
    return {
        "messages": [],
        "message_feedback": {},
        "current_mode": "ask",
        "conversation_topic": None,
        "study_topic": None,
        "mastery_intro": None,
        "mastery_intro_citations": [],
        "remediation_message": None,
        "remediation_citations": [],
        "mastery_status": "idle",
        "current_quiz": None,
        "quiz_round": 0,
        "last_quiz_result": None,
        "weak_concepts": [],
        "study_plan": None,
        "study_plan_citations": [],
    }


@dataclass
class SessionStateDefaults:
    session_id: str = field(default_factory=generate_session_id)
    messages: list[dict[str, object]] = field(default_factory=list)
    message_feedback: dict[str, dict[str, object]] = field(default_factory=dict)
    uploaded_sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    document_library: list[dict[str, object]] = field(default_factory=list)
    active_document_id: str | None = None
    library_status_message: str | None = None
    current_mode: AppMode = "ask"
    conversation_topic: str | None = None
    study_topic: str | None = None
    mastery_intro: str | None = None
    mastery_intro_citations: list[str] = field(default_factory=list)
    remediation_message: str | None = None
    remediation_citations: list[str] = field(default_factory=list)
    mastery_status: MasteryStatus = "idle"
    current_quiz: StudyQuiz | None = None
    quiz_round: int = 0
    last_quiz_result: QuizResult | None = None
    weak_concepts: list[str] = field(default_factory=list)
    study_plan: StudyPlan | None = None
    study_plan_citations: list[str] = field(default_factory=list)
    observability_enabled: bool = False


@dataclass
class IndexedDocument:
    document_id: str
    session_id: str
    filename: str
    document_title: str
    document_topic: str
    document_summary: str
    chunks: list[Chunk]
    size_mb: float
    last_conversation_topic: str | None = None
    last_opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def initialize_session_state() -> None:
    # Keep session defaults in one place so new UI modes can reuse them consistently.
    defaults = SessionStateDefaults()
    for key, value in asdict(defaults).items():
        st.session_state.setdefault(key, value)


def _build_workspace(indexed_document: IndexedDocument) -> dict[str, object]:
    workspace = {
        "document_id": indexed_document.document_id,
        "session_id": indexed_document.session_id,
        "filename": indexed_document.filename,
        "document_title": indexed_document.document_title,
        "document_topic": indexed_document.document_topic,
        "document_summary": indexed_document.document_summary,
        "chunks": indexed_document.chunks,
        "size_mb": indexed_document.size_mb,
        "chunk_count": len(indexed_document.chunks),
        "last_conversation_topic": indexed_document.last_conversation_topic,
        "last_opened_at": indexed_document.last_opened_at,
    }
    workspace.update(_empty_workspace_state())
    return workspace


def build_workspace_from_session() -> dict[str, object] | None:
    if not st.session_state.uploaded_sources or not st.session_state.chunks:
        return None

    filename = st.session_state.uploaded_sources[0]
    workspace = {
        "document_id": f"session-{st.session_state.session_id}",
        "session_id": st.session_state.session_id,
        "filename": filename,
        "document_title": filename.rsplit(".", 1)[0].replace("_", " ").title(),
        "document_topic": filename.rsplit(".", 1)[0].replace("_", " ").title(),
        "document_summary": f"Recovered active workspace for {filename}.",
        "chunks": list(st.session_state.chunks),
        "size_mb": 0.0,
        "chunk_count": len(st.session_state.chunks),
        "last_conversation_topic": st.session_state.conversation_topic,
        "last_opened_at": datetime.now(timezone.utc).isoformat(),
    }
    workspace.update(_workspace_state_from_session())
    return workspace


def _workspace_state_from_session() -> dict[str, object]:
    return {
        "messages": list(st.session_state.messages),
        "message_feedback": dict(st.session_state.message_feedback),
        "current_mode": st.session_state.current_mode,
        "conversation_topic": st.session_state.conversation_topic,
        "last_conversation_topic": st.session_state.conversation_topic,
        "study_topic": st.session_state.study_topic,
        "mastery_intro": st.session_state.mastery_intro,
        "mastery_intro_citations": list(st.session_state.mastery_intro_citations),
        "remediation_message": st.session_state.remediation_message,
        "remediation_citations": list(st.session_state.remediation_citations),
        "mastery_status": st.session_state.mastery_status,
        "current_quiz": st.session_state.current_quiz,
        "quiz_round": st.session_state.quiz_round,
        "last_quiz_result": st.session_state.last_quiz_result,
        "weak_concepts": list(st.session_state.weak_concepts),
        "study_plan": st.session_state.study_plan,
        "study_plan_citations": list(st.session_state.study_plan_citations),
    }


def save_active_document_workspace() -> None:
    active_document_id = st.session_state.active_document_id
    if active_document_id is None:
        return

    for workspace in st.session_state.document_library:
        if workspace["document_id"] == active_document_id:
            workspace.update(_workspace_state_from_session())
            return


def _load_workspace_state(workspace: dict[str, object]) -> None:
    st.session_state.session_id = workspace["session_id"]
    st.session_state.uploaded_sources = [workspace["filename"]]
    st.session_state.chunks = workspace["chunks"]
    st.session_state.messages = list(workspace["messages"])
    st.session_state.message_feedback = dict(workspace.get("message_feedback", {}))
    st.session_state.current_mode = workspace["current_mode"]
    st.session_state.conversation_topic = workspace["conversation_topic"]
    st.session_state.study_topic = workspace["study_topic"]
    st.session_state.mastery_intro = workspace["mastery_intro"]
    st.session_state.mastery_intro_citations = list(workspace["mastery_intro_citations"])
    st.session_state.remediation_message = workspace["remediation_message"]
    st.session_state.remediation_citations = list(workspace["remediation_citations"])
    st.session_state.mastery_status = workspace["mastery_status"]
    st.session_state.current_quiz = workspace["current_quiz"]
    st.session_state.quiz_round = workspace["quiz_round"]
    st.session_state.last_quiz_result = workspace["last_quiz_result"]
    st.session_state.weak_concepts = list(workspace["weak_concepts"])
    st.session_state.study_plan = workspace["study_plan"]
    st.session_state.study_plan_citations = list(workspace["study_plan_citations"])
    workspace["last_opened_at"] = datetime.now(timezone.utc).isoformat()


def activate_document_workspace(document_id: str) -> None:
    save_active_document_workspace()
    for workspace in st.session_state.document_library:
        if workspace["document_id"] == document_id:
            st.session_state.active_document_id = document_id
            _load_workspace_state(workspace)
            return


def set_document_library(
    document_library: list[dict[str, object]],
    active_document_id: str | None,
) -> None:
    st.session_state.document_library = document_library
    st.session_state.active_document_id = active_document_id
    if active_document_id is not None:
        activate_document_workspace(active_document_id)


def store_indexed_document(indexed_document: IndexedDocument) -> None:
    workspace = _build_workspace(indexed_document)
    st.session_state.document_library = [
        existing_workspace
        for existing_workspace in st.session_state.document_library
        if existing_workspace["document_id"] != indexed_document.document_id
    ]
    st.session_state.document_library.insert(0, workspace)
    st.session_state.active_document_id = indexed_document.document_id
    _load_workspace_state(workspace)
    st.success(
        f"Indexed {indexed_document.filename} into {len(indexed_document.chunks)} chunks."
    )


def store_message(
    role: str,
    content: str,
    citations: list[str] | None = None,
    topic: str | None = None,
    query: str | None = None,
    mode: AppMode | None = None,
) -> None:
    message: dict[str, object] = {"role": role, "content": content}
    if role == "assistant":
        message["id"] = generate_message_id()
    if citations:
        message["citations"] = citations
    if topic:
        message["topic"] = topic
    if query:
        message["query"] = query
    if mode:
        message["mode"] = mode
    st.session_state.messages.append(message)
    save_active_document_workspace()


def store_message_feedback(message_id: str, feedback: dict[str, object]) -> None:
    st.session_state.message_feedback[message_id] = feedback
    save_active_document_workspace()


def set_current_mode(mode: AppMode) -> None:
    st.session_state.current_mode = mode
    save_active_document_workspace()


def ensure_current_mode(mode: AppMode) -> None:
    st.session_state.setdefault("current_mode", mode)


def set_conversation_topic(topic: str | None) -> None:
    st.session_state.conversation_topic = topic
    save_active_document_workspace()


def store_mastery_session(mastery_session: MasterySession) -> None:
    st.session_state.study_topic = mastery_session.topic
    st.session_state.mastery_intro = mastery_session.intro_message
    st.session_state.mastery_intro_citations = mastery_session.citations
    st.session_state.remediation_message = None
    st.session_state.remediation_citations = []
    st.session_state.mastery_status = mastery_session.status
    st.session_state.last_quiz_result = None
    st.session_state.weak_concepts = []
    st.session_state.study_plan = None
    st.session_state.study_plan_citations = []
    save_active_document_workspace()


def store_current_quiz(
    quiz: StudyQuiz,
    quiz_round: int = 1,
    *,
    clear_previous_result: bool = True,
) -> None:
    st.session_state.current_quiz = quiz
    st.session_state.quiz_round = quiz_round
    if clear_previous_result:
        st.session_state.last_quiz_result = None
    save_active_document_workspace()


def clear_current_quiz() -> None:
    st.session_state.current_quiz = None
    st.session_state.quiz_round = 0
    save_active_document_workspace()


def store_quiz_result(quiz_result: QuizResult) -> None:
    st.session_state.last_quiz_result = quiz_result
    st.session_state.weak_concepts = quiz_result.weak_concepts
    st.session_state.mastery_status = (
        "completed" if quiz_result.score == quiz_result.total else "in_progress"
    )
    save_active_document_workspace()


def store_remediation_message(message: str | None) -> None:
    st.session_state.remediation_message = message
    save_active_document_workspace()


def store_remediation_citations(citations: list[str]) -> None:
    st.session_state.remediation_citations = citations
    save_active_document_workspace()


def store_study_plan(study_plan: StudyPlan, status: MasteryStatus) -> None:
    st.session_state.study_plan = study_plan
    st.session_state.mastery_status = status
    save_active_document_workspace()


def store_study_plan_citations(citations: list[str]) -> None:
    st.session_state.study_plan_citations = citations
    save_active_document_workspace()


def clear_mastery_session() -> None:
    st.session_state.conversation_topic = None
    st.session_state.study_topic = None
    st.session_state.mastery_intro = None
    st.session_state.mastery_intro_citations = []
    st.session_state.remediation_message = None
    st.session_state.remediation_citations = []
    st.session_state.mastery_status = "idle"
    st.session_state.current_quiz = None
    st.session_state.quiz_round = 0
    st.session_state.last_quiz_result = None
    st.session_state.weak_concepts = []
    st.session_state.study_plan = None
    st.session_state.study_plan_citations = []
    save_active_document_workspace()
