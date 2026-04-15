"""app_state.py: Manage Streamlit session state and workspace persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import streamlit as st

from src.core.models import AppMode, Chunk, MasterySession, MasteryStatus, QuizResult, StudyPlan, StudyQuiz
from src.core.observability import log_langfuse_event
from src.core.utils import generate_message_id, generate_session_id


def _empty_workspace_state() -> dict[str, object]:
    """Return the default state for a document workspace.

    Returns:
        dict[str, object]: Default workspace state payload.
    """
    return {
        "messages": [],
        "message_feedback": {},
        "current_mode": "ask",
        "conversation_topic": None,
        "quiz_goal": None,
        "study_topic": None,
        "mastery_intro": None,
        "mastery_intro_citations": [],
        "remediation_message": None,
        "remediation_citations": [],
        "remediation_payload": None,
        "mastery_status": "idle",
        "current_quiz": None,
        "quiz_round": 0,
        "last_quiz_result": None,
        "weak_concepts": [],
        "study_plan": None,
        "study_plan_citations": [],
        "quiz_history": [],
        "quiz_view_round": None,
        "last_info_lane_label": None,
        "info_lane_variant_index": 0,
        "reindex_request": None,
    }


@dataclass
class SessionStateDefaults:
    """Provide defaults for Streamlit session state keys."""

    session_id: str = field(default_factory=generate_session_id)
    user_email: str | None = None
    user_id: str | None = None
    auth_email: str = ""
    auth_code_sent: bool = False
    auth_error: str | None = None
    messages: list[dict[str, object]] = field(default_factory=list)
    message_feedback: dict[str, dict[str, object]] = field(default_factory=dict)
    uploaded_sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    document_library: list[dict[str, object]] = field(default_factory=list)
    active_document_id: str | None = None
    library_status_message: str | None = None
    current_mode: AppMode = "ask"
    conversation_topic: str | None = None
    quiz_goal: str | None = None
    study_topic: str | None = None
    mastery_intro: str | None = None
    mastery_intro_citations: list[str] = field(default_factory=list)
    remediation_message: str | None = None
    remediation_citations: list[str] = field(default_factory=list)
    remediation_payload: dict[str, str] | None = None
    mastery_status: MasteryStatus = "idle"
    current_quiz: StudyQuiz | None = None
    quiz_round: int = 0
    last_quiz_result: QuizResult | None = None
    weak_concepts: list[str] = field(default_factory=list)
    study_plan: StudyPlan | None = None
    study_plan_citations: list[str] = field(default_factory=list)
    quiz_history: list[dict[str, object]] = field(default_factory=list)
    quiz_view_round: int | None = None
    observability_enabled: bool = False
    reindex_request: str | None = None
    last_info_lane_label: str | None = None
    info_lane_variant_index: int = 0


@dataclass
class IndexedDocument:
    """Represent an indexed document and its associated chunks."""

    document_id: str
    session_id: str
    filename: str
    document_title: str
    document_topic: str
    document_summary: str
    key_hooks: list[str]
    chunks: list[Chunk]
    size_mb: float
    user_id: str | None = None
    last_conversation_topic: str | None = None
    last_opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def initialize_session_state() -> None:
    """Ensure Streamlit session state has all expected defaults."""
    defaults = SessionStateDefaults()  # Keep defaults centralized for consistent UI behavior.
    for key, value in asdict(defaults).items():
        st.session_state.setdefault(key, value)


def _build_workspace(indexed_document: IndexedDocument) -> dict[str, object]:
    """Build a new workspace record from an indexed document.

    Args:
        indexed_document (IndexedDocument): Document metadata and chunks.

    Returns:
        dict[str, object]: Workspace payload for the library.
    """
    workspace = {
        "document_id": indexed_document.document_id,
        "session_id": indexed_document.session_id,
        "user_id": indexed_document.user_id,
        "filename": indexed_document.filename,
        "document_title": indexed_document.document_title,
        "document_topic": indexed_document.document_topic,
        "document_summary": indexed_document.document_summary,
        "key_hooks": indexed_document.key_hooks,
        "chunks": indexed_document.chunks,
        "size_mb": indexed_document.size_mb,
        "chunk_count": len(indexed_document.chunks),
        "last_conversation_topic": indexed_document.last_conversation_topic,
        "last_opened_at": indexed_document.last_opened_at,
    }
    workspace.update(_empty_workspace_state())
    return workspace


def build_workspace_from_session() -> dict[str, object] | None:
    """Create a workspace record from the current session data.

    Returns:
        dict[str, object] | None: Workspace payload or None if no data exists.
    """
    if not st.session_state.uploaded_sources or not st.session_state.chunks:
        return None

    filename = st.session_state.uploaded_sources[0]
    workspace = {
        "document_id": f"session-{st.session_state.session_id}",
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "filename": filename,
        "document_title": filename.rsplit(".", 1)[0].replace("_", " ").title(),
        "document_topic": filename.rsplit(".", 1)[0].replace("_", " ").title(),
        "document_summary": f"Recovered active workspace for {filename}.",
        "key_hooks": [],
        "chunks": list(st.session_state.chunks),
        "size_mb": 0.0,
        "chunk_count": len(st.session_state.chunks),
        "last_conversation_topic": st.session_state.conversation_topic,
        "last_opened_at": datetime.now(timezone.utc).isoformat(),
    }
    workspace.update(_workspace_state_from_session())
    return workspace


def _workspace_state_from_session() -> dict[str, object]:
    """Capture the current session state into a workspace payload.

    Returns:
        dict[str, object]: Workspace state snapshot.
    """
    return {
        "messages": list(st.session_state.messages),
        "message_feedback": dict(st.session_state.message_feedback),
        "current_mode": st.session_state.current_mode,
        "conversation_topic": st.session_state.conversation_topic,
        "last_conversation_topic": st.session_state.conversation_topic,
        "quiz_goal": st.session_state.quiz_goal,
        "study_topic": st.session_state.study_topic,
        "mastery_intro": st.session_state.mastery_intro,
        "mastery_intro_citations": list(st.session_state.mastery_intro_citations),
        "remediation_message": st.session_state.remediation_message,
        "remediation_citations": list(st.session_state.remediation_citations),
        "remediation_payload": st.session_state.remediation_payload,
        "mastery_status": st.session_state.mastery_status,
        "current_quiz": st.session_state.current_quiz,
        "quiz_round": st.session_state.quiz_round,
        "last_quiz_result": st.session_state.last_quiz_result,
        "weak_concepts": list(st.session_state.weak_concepts),
        "study_plan": st.session_state.study_plan,
        "study_plan_citations": list(st.session_state.study_plan_citations),
        "quiz_history": list(st.session_state.quiz_history),
        "quiz_view_round": st.session_state.quiz_view_round,
        "last_info_lane_label": st.session_state.last_info_lane_label,
        "info_lane_variant_index": st.session_state.info_lane_variant_index,
        "reindex_request": st.session_state.reindex_request,
    }


def save_active_document_workspace() -> None:
    """Persist changes back into the active workspace record."""
    active_document_id = st.session_state.active_document_id
    if active_document_id is None:
        return

    for workspace in st.session_state.document_library:
        if workspace["document_id"] == active_document_id:
            workspace.update(_workspace_state_from_session())
            return


def _load_workspace_state(workspace: dict[str, object]) -> None:
    """Load a workspace record into the active session state.

    Args:
        workspace (dict[str, object]): Workspace payload to load.
    """
    st.session_state.session_id = workspace["session_id"]
    st.session_state.uploaded_sources = [workspace["filename"]]
    st.session_state.chunks = workspace["chunks"]
    st.session_state.messages = list(workspace["messages"])
    st.session_state.message_feedback = dict(workspace.get("message_feedback", {}))
    st.session_state.current_mode = workspace["current_mode"]
    st.session_state.conversation_topic = workspace["conversation_topic"]
    st.session_state.quiz_goal = workspace.get("quiz_goal")
    st.session_state.study_topic = workspace["study_topic"]
    st.session_state.mastery_intro = workspace["mastery_intro"]
    st.session_state.mastery_intro_citations = list(workspace["mastery_intro_citations"])
    st.session_state.remediation_message = workspace["remediation_message"]
    st.session_state.remediation_citations = list(workspace["remediation_citations"])
    st.session_state.remediation_payload = workspace.get("remediation_payload")
    st.session_state.mastery_status = workspace["mastery_status"]
    st.session_state.current_quiz = workspace["current_quiz"]
    st.session_state.quiz_round = workspace["quiz_round"]
    st.session_state.last_quiz_result = workspace["last_quiz_result"]
    st.session_state.weak_concepts = list(workspace["weak_concepts"])
    st.session_state.study_plan = workspace["study_plan"]
    st.session_state.study_plan_citations = list(workspace["study_plan_citations"])
    st.session_state.quiz_history = list(workspace.get("quiz_history", []))
    st.session_state.quiz_view_round = workspace.get("quiz_view_round")
    st.session_state.last_info_lane_label = workspace.get("last_info_lane_label")
    st.session_state.info_lane_variant_index = workspace.get("info_lane_variant_index", 0)
    st.session_state.reindex_request = workspace.get("reindex_request")
    workspace["last_opened_at"] = datetime.now(timezone.utc).isoformat()


def activate_document_workspace(document_id: str) -> None:
    """Switch the session to a chosen workspace.

    Args:
        document_id (str): Workspace document identifier.
    """
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
    """Replace the document library and optionally activate one workspace.

    Args:
        document_library (list[dict[str, object]]): New document library.
        active_document_id (str | None): Active workspace identifier.
    """
    st.session_state.document_library = document_library
    st.session_state.active_document_id = active_document_id
    if active_document_id is not None:
        activate_document_workspace(active_document_id)


def store_indexed_document(indexed_document: IndexedDocument) -> None:
    """Add a newly indexed document to the library and activate it.

    Args:
        indexed_document (IndexedDocument): Indexed document payload.
    """
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
    info_lane: dict[str, str] | None = None,
    quiz_lane: dict[str, str] | None = None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    topic_subject: str | None = None,
) -> None:
    """Append a user or assistant message to the conversation history.

    Args:
        role (str): Message role (user or assistant).
        content (str): Message content.
        citations (list[str] | None): Optional citations for assistant messages.
        topic (str | None): Conversation topic label.
        query (str | None): Original query string.
        mode (AppMode | None): Mode associated with the message.
        info_lane (dict[str, str] | None): Info lane payload.
        quiz_lane (dict[str, str] | None): Quiz lane payload.
        trace_id (str | None): Langfuse trace identifier.
        observation_id (str | None): Langfuse observation identifier.
        topic_subject (str | None): Quiz topic subject.
    """
    message: dict[str, object] = {"role": role, "content": content}
    if role == "assistant":  # Only assistants need persistent IDs for feedback tracking.
        message["id"] = generate_message_id()
    if citations:
        message["citations"] = citations
    if topic:
        message["topic"] = topic
    if query:
        message["query"] = query
    if mode:
        message["mode"] = mode
    if info_lane:
        message["info_lane"] = info_lane
    if quiz_lane:
        message["quiz_lane"] = quiz_lane
    if trace_id:
        message["trace_id"] = trace_id
    if observation_id:
        message["observation_id"] = observation_id
    if topic_subject:
        message["topic_subject"] = topic_subject
    st.session_state.messages.append(message)
    save_active_document_workspace()


def store_message_feedback(message_id: str, feedback: dict[str, object]) -> None:
    """Save feedback for a specific assistant message.

    Args:
        message_id (str): Message identifier.
        feedback (dict[str, object]): Feedback payload.
    """
    st.session_state.message_feedback[message_id] = feedback
    save_active_document_workspace()


def set_current_mode(mode: AppMode) -> None:
    """Update the current app mode.

    Args:
        mode (AppMode): New app mode.
    """
    previous_mode = st.session_state.current_mode
    st.session_state.current_mode = mode
    save_active_document_workspace()
    if previous_mode != mode:
        log_langfuse_event(
            "mode_switch",
            session_id=st.session_state.session_id,
            metadata={"from": previous_mode, "to": mode},
        )


def ensure_current_mode(mode: AppMode) -> None:
    """Ensure a default app mode exists in session state.

    Args:
        mode (AppMode): Default mode to set if missing.
    """
    st.session_state.setdefault("current_mode", mode)


def set_conversation_topic(topic: str | None) -> None:
    """Update the current conversation topic.

    Args:
        topic (str | None): New topic label.
    """
    st.session_state.conversation_topic = topic
    save_active_document_workspace()


def store_mastery_session(mastery_session: MasterySession) -> None:
    """Store a new mastery session and clear old mastery data.

    Args:
        mastery_session (MasterySession): Session metadata to store.
    """
    st.session_state.study_topic = mastery_session.topic
    st.session_state.mastery_intro = mastery_session.intro_message
    st.session_state.mastery_intro_citations = mastery_session.citations
    st.session_state.remediation_message = None
    st.session_state.remediation_citations = []
    st.session_state.remediation_payload = None
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
    """Store the current quiz and its round number.

    Args:
        quiz (StudyQuiz): Quiz payload.
        quiz_round (int): Round number for the quiz.
        clear_previous_result (bool): Whether to clear prior results.
    """
    st.session_state.current_quiz = quiz
    st.session_state.quiz_round = quiz_round
    if clear_previous_result:
        st.session_state.last_quiz_result = None
    save_active_document_workspace()


def clear_current_quiz() -> None:
    """Clear the active quiz from session state."""
    st.session_state.current_quiz = None
    st.session_state.quiz_round = 0
    save_active_document_workspace()


def store_quiz_result(quiz_result: QuizResult) -> None:
    """Store the latest quiz result and update mastery status.

    Args:
        quiz_result (QuizResult): Latest quiz result.
    """
    st.session_state.last_quiz_result = quiz_result
    st.session_state.weak_concepts = quiz_result.weak_concepts
    st.session_state.mastery_status = (
        "completed" if quiz_result.score == quiz_result.total else "in_progress"
    )
    save_active_document_workspace()


def store_remediation_message(message: str | None) -> None:
    """Save the reteach message for the current mastery round.

    Args:
        message (str | None): Reteach message text.
    """
    st.session_state.remediation_message = message
    save_active_document_workspace()


def store_remediation_payload(payload: dict[str, str] | None) -> None:
    """Save reteach payload metadata for the current mastery round.

    Args:
        payload (dict[str, str] | None): Reteach payload data.
    """
    st.session_state.remediation_payload = payload
    save_active_document_workspace()


def store_remediation_citations(citations: list[str]) -> None:
    """Save citations associated with the reteach message.

    Args:
        citations (list[str]): Citation strings to store.
    """
    st.session_state.remediation_citations = citations
    save_active_document_workspace()


def store_study_plan(study_plan: StudyPlan, status: MasteryStatus) -> None:
    """Store the final study plan and mastery status.

    Args:
        study_plan (StudyPlan): Study plan payload.
        status (MasteryStatus): Mastery status to persist.
    """
    st.session_state.study_plan = study_plan
    st.session_state.mastery_status = status
    save_active_document_workspace()


def store_study_plan_citations(citations: list[str]) -> None:
    """Save citations associated with the study plan.

    Args:
        citations (list[str]): Citation strings to store.
    """
    st.session_state.study_plan_citations = citations
    save_active_document_workspace()


def clear_mastery_session() -> None:
    """Reset mastery-related state back to idle."""
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
    st.session_state.quiz_goal = None
    st.session_state.quiz_history = []
    st.session_state.quiz_view_round = None
    save_active_document_workspace()
