from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.agent import advance_mastery_loop, start_mastery_loop, stop_mastery_loop
from src.feedback_store import load_recent_feedback, save_response_feedback
from src.index_cache import persist_document_library, restore_document_library
from src.mode_router import detect_app_mode, extract_conversation_topic, is_generic_mastery_topic
from src.notes_answering import build_answer_response, get_supporting_citations
from src.notes_upload import index_uploaded_file
from src.app_state import (
    activate_document_workspace,
    build_workspace_from_session,
    clear_current_quiz,
    clear_mastery_session,
    ensure_current_mode,
    initialize_session_state,
    save_active_document_workspace,
    set_current_mode,
    set_conversation_topic,
    set_document_library,
    store_current_quiz,
    store_mastery_session,
    store_message_feedback,
    store_indexed_document,
    store_message,
    store_quiz_result,
    store_remediation_citations,
    store_remediation_message,
    store_study_plan,
    store_study_plan_citations,
)
from src.config import settings
from src.grading import grade_quiz
from src.models import AppMode, FeedbackRating, QuizResult, ResponseFeedback, StudyPlan, StudyQuiz
from src.observability import initialize_observability
from src.utils import humanize_label
from src.vector_store import rebuild_document_library_from_remote


def render_hero() -> None:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.25rem 0;">
            <div style="display: inline-block; padding: 0.35rem 0.75rem; border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px; font-size: 0.9rem; color: #9ec5ff; background: rgba(50, 110, 190, 0.14);">
                Notes in, study loop out
            </div>
            <h1 style="margin: 1rem 0 0.4rem 0; font-size: 4rem; line-height: 0.95;">Study Pal</h1>
            <p style="max-width: 46rem; font-size: 1.15rem; color: #c7ccd6; margin: 0;">
                Upload your class notes, ask grounded questions, or switch into a guided mastery loop
                that quizzes you, reteaches weak spots, and finishes with a study plan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_overview() -> None:
    ask_column, mastery_column, outcome_column = st.columns(3)
    ask_column.markdown(
        """
        <div style="padding: 1rem; border-radius: 1rem; background: rgba(255,255,255,0.04); min-height: 10rem;">
            <div style="font-size: 0.85rem; color: #8fb7ff; text-transform: uppercase; letter-spacing: 0.08em;">Ask Mode</div>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;">Get cited answers fast</div>
            <div style="margin-top: 0.6rem; color: #c7ccd6;">Ask about definitions, examples, formulas, or confusing passages and get note-grounded responses.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mastery_column.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 1rem; background: rgba(255,255,255,0.04); min-height: 10rem;">
            <div style="font-size: 0.85rem; color: #8fb7ff; text-transform: uppercase; letter-spacing: 0.08em;">Mastery Mode</div>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;">Study with feedback loops</div>
            <div style="margin-top: 0.6rem; color: #c7ccd6;">Start with a topic, take a {settings.quiz_questions_per_round}-question checkpoint quiz, review weak concepts, and keep going until you reach a stopping point.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    outcome_column.markdown(
        """
        <div style="padding: 1rem; border-radius: 1rem; background: linear-gradient(180deg, rgba(66,153,225,0.14), rgba(255,255,255,0.04)); min-height: 10rem;">
            <div style="font-size: 0.85rem; color: #8fb7ff; text-transform: uppercase; letter-spacing: 0.08em;">Outcome</div>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;">Leave with a plan</div>
            <div style="margin-top: 0.6rem; color: #c7ccd6;">Every mastery session can end with a study plan built from the topic, your quiz results, and the uploaded notes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_prompts() -> None:
    st.markdown("#### Try prompts like these")
    prompt_column, mastery_prompt_column = st.columns(2)
    prompt_column.code("What does the notes packet say about eigenvalues?", language="text")
    mastery_prompt_column.code("Help me study photosynthesis for my quiz tomorrow.", language="text")


def render_upload_panel() -> None:
    uploaded_file = st.file_uploader(
        "Upload notes",
        type=list(settings.allowed_file_types),
        help=f"Accepted file types: {', '.join(settings.allowed_file_types).upper()} | Recommended max size: {settings.max_file_size_mb}MB",
    )
    if uploaded_file is not None:
        indexed_document = index_uploaded_file(uploaded_file)
        store_indexed_document(indexed_document)
        st.session_state.library_status_message = (
            f"Saved {indexed_document.filename} to your study library."
        )
        persist_document_library(
            st.session_state.document_library,
            st.session_state.active_document_id,
        )
        st.rerun()


def render_document_library() -> None:
    st.markdown("### Your study library")
    if st.session_state.library_status_message:
        st.info(st.session_state.library_status_message)
    if not st.session_state.document_library:
        st.caption("No saved note workspaces yet. Upload a file to create one.")
        return

    columns = st.columns(2)
    for index, workspace in enumerate(st.session_state.document_library):
        column = columns[index % 2]
        with column:
            with st.container(border=True):
                st.markdown(f"#### {workspace.get('document_title') or workspace['filename']}")
                st.caption(workspace["filename"])
                st.caption(
                    f"{workspace['chunk_count']} chunks | {workspace['size_mb']}MB"
                )
                st.write(f"Document topic: {workspace.get('document_topic') or 'Unknown'}")
                summary = workspace.get("document_summary")
                if summary:
                    st.write(summary)
                topic = workspace.get("last_conversation_topic") or "No topic yet"
                st.write(f"Last conversation topic: {topic}")
                last_opened_at = workspace.get("last_opened_at")
                if last_opened_at:
                    st.caption(f"Last opened: {last_opened_at}")
                status = str(workspace.get("mastery_status", "idle")).replace("_", " ").title()
                st.write(f"Status: {status}")
                button_label = "Continue chat" if st.session_state.active_document_id == workspace["document_id"] else "Open chat"
                if st.button(button_label, key=f"open_workspace_{workspace['document_id']}"):
                    activate_document_workspace(str(workspace["document_id"]))
                    st.session_state.library_status_message = (
                        f"Opened {workspace['filename']} from your study library."
                    )
                    persist_document_library(
                        st.session_state.document_library,
                        st.session_state.active_document_id,
                    )
                    st.rerun()


def _render_message_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_source_list(message.get("citations", []), caption="Sources used")
            if message["role"] == "assistant":
                render_message_feedback_form(message)


def render_message_feedback_form(message: dict[str, object]) -> None:
    message_id = message.get("id")
    if not isinstance(message_id, str):
        return

    existing_feedback = st.session_state.message_feedback.get(message_id)
    if existing_feedback is not None:
        st.caption(f"Feedback saved: {existing_feedback['rating']}")
        if existing_feedback.get("feedback_text"):
            st.caption(f"Comment: {existing_feedback['feedback_text']}")
        return

    with st.expander("How was this response?", expanded=False):
        with st.form(f"feedback_form_{message_id}"):
            rating = st.radio(
                "Rate this response",
                options=["Very helpful", "Somewhat helpful", "Not helpful"],
                horizontal=True,
            )
            feedback_text = st.text_area(
                "Anything else we should know?",
                placeholder="Optional: tell us what was helpful, confusing, or missing.",
            )
            submitted = st.form_submit_button("Save feedback")

        if submitted:
            _submit_response_feedback(
                message=message,
                rating=rating,
                feedback_text=feedback_text,
            )


def _get_recent_topic_context() -> str | None:
    explicit_topic = st.session_state.study_topic or st.session_state.conversation_topic
    if explicit_topic and not is_generic_mastery_topic(explicit_topic):
        return explicit_topic

    for message in reversed(st.session_state.messages):
        topic = message.get("topic")
        if isinstance(topic, str) and topic and not is_generic_mastery_topic(topic):
            return topic
    return None


def _handle_question(question: str) -> None:
    detected_mode = detect_app_mode(question)
    previous_topic = _get_recent_topic_context()
    conversation_topic = extract_conversation_topic(question, fallback_topic=previous_topic)
    set_current_mode(detected_mode)
    set_conversation_topic(conversation_topic)
    store_message("user", question, topic=conversation_topic)
    if detected_mode == "mastery":
        mastery_session, mastery_progress = start_mastery_loop(
            question,
            fallback_topic=previous_topic,
        )
        set_conversation_topic(mastery_session.topic)
        store_mastery_session(mastery_session)
        if mastery_progress.next_quiz and mastery_progress.next_quiz_round:
            store_current_quiz(mastery_progress.next_quiz, mastery_progress.next_quiz_round)
        answer_message = mastery_session.intro_message
        answer_citations: list[str] = []
        conversation_topic = mastery_session.topic
    else:
        clear_mastery_session()
        set_conversation_topic(conversation_topic)
        answer_response = build_answer_response(question)
        answer_message = answer_response.answer
        answer_citations = answer_response.citations
    store_message(
        "assistant",
        answer_message,
        citations=answer_citations,
        topic=conversation_topic,
        query=question,
        mode=detected_mode,
    )
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
    )
    st.rerun()


def render_chat() -> None:
    _render_message_history()
    if st.session_state.current_mode == "mastery":
        render_mastery_inline()
    question = st.chat_input("Ask a question about your uploaded notes")
    if question:
        _handle_question(question)


def _submit_response_feedback(
    *,
    message: dict[str, object],
    rating: FeedbackRating,
    feedback_text: str,
) -> None:
    message_id = str(message["id"])
    feedback_record = ResponseFeedback(
        message_id=message_id,
        session_id=st.session_state.session_id,
        document_id=st.session_state.active_document_id,
        filename=st.session_state.uploaded_sources[0] if st.session_state.uploaded_sources else None,
        query=str(message.get("query") or ""),
        response=str(message["content"]),
        rating=rating,
        feedback_text=feedback_text.strip() or None,
        topic=str(message.get("topic")) if message.get("topic") else None,
        mode=message.get("mode", st.session_state.current_mode),
        citations=[str(citation) for citation in message.get("citations", [])],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_response_feedback(feedback_record)
    store_message_feedback(message_id, feedback_record.model_dump())
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
    )
    st.success("Thanks. Your feedback was saved.")


def _render_active_mode() -> None:
    render_chat()


def render_mastery_inline() -> None:
    if (
        st.session_state.mastery_status == "idle"
        and not st.session_state.current_quiz
        and not st.session_state.last_quiz_result
        and not st.session_state.study_plan
    ):
        return

    if _should_show_stop_mastery_button():
        control_column, status_column = st.columns([1, 4])
        with control_column:
            if st.button("Stop and build study plan", key="stop_mastery"):
                stop_mastery_session()
        with status_column:
            st.caption(_build_mastery_context_line())
    else:
        st.caption(_build_mastery_context_line())

    _render_mastery_status_banner()
    if st.session_state.mastery_status in {"completed", "stopped"}:
        st.caption("Start a new mastery prompt in chat if you want to run another round on a new focus area.")
    if st.session_state.last_quiz_result:
        render_quiz_result(st.session_state.last_quiz_result)
    if st.session_state.remediation_message:
        with st.expander("Reteach focus", expanded=True):
            st.markdown(st.session_state.remediation_message)
            render_source_list(st.session_state.remediation_citations)
    if st.session_state.current_quiz:
        render_quiz_panel(st.session_state.current_quiz, st.session_state.quiz_round)
    if st.session_state.study_plan:
        render_study_plan(st.session_state.study_plan)


def render_quiz_panel(quiz: StudyQuiz, quiz_round: int) -> None:
    st.markdown(f"### {quiz.title}")
    st.caption(_build_round_caption(quiz_round))
    with st.form(f"quiz_form_round_{quiz_round}"):
        for index, question in enumerate(quiz.questions, start=1):
            st.markdown(f"**Question {index}.** {question.prompt}")
            st.radio(
                label=f"Select an answer for question {index}",
                options=question.options,
                index=None,
                key=f"quiz_question_{quiz_round}_{index}",
                label_visibility="collapsed",
            )

        # Use a form so all quiz answers are submitted and graded together.
        submitted = st.form_submit_button("Grade quiz")

    if submitted:
        answers = [
            st.session_state.get(f"quiz_question_{quiz_round}_{index}")
            for index in range(1, len(quiz.questions) + 1)
        ]
        _submit_quiz_answers(quiz, quiz_round, answers)


def _submit_quiz_answers(quiz: StudyQuiz, quiz_round: int, answers: list[str | None]) -> None:
    # Keep quiz submission state changes together so the flow is easier to review and test.
    quiz_result = grade_quiz(quiz, answers)
    store_quiz_result(quiz_result)
    mastery_progress = advance_mastery_loop(quiz.topic, quiz_result, quiz_round)
    citation_query = f"{quiz.topic} {' '.join(quiz_result.weak_concepts)}".strip()
    store_remediation_message(mastery_progress.remediation_message)
    store_remediation_citations(get_supporting_citations(citation_query))
    if mastery_progress.next_quiz and mastery_progress.next_quiz_round:
        store_current_quiz(
            mastery_progress.next_quiz,
            mastery_progress.next_quiz_round,
            clear_previous_result=False,
        )
    elif mastery_progress.status in {"completed", "stopped"}:
        clear_current_quiz()
        if mastery_progress.study_plan:
            store_study_plan(mastery_progress.study_plan, mastery_progress.status)
            store_study_plan_citations(get_supporting_citations(citation_query))
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
    )
    st.rerun()


def render_quiz_result(quiz_result: QuizResult) -> None:
    st.markdown("### Latest checkpoint")
    st.metric("Score", f"{quiz_result.score}/{quiz_result.total}")
    if quiz_result.weak_concepts:
        st.warning(
            "Focus next on: "
            + ", ".join(humanize_label(concept) for concept in quiz_result.weak_concepts)
        )
        with st.expander("Answer review", expanded=False):
            for index, feedback in enumerate(quiz_result.feedback, start=1):
                result_label = "Correct" if feedback.is_correct else "Needs review"
                st.markdown(
                    f"**Question {index}** ({result_label})  \n"
                    f"Your answer: {feedback.user_answer or 'No answer'}  \n"
                    f"Correct answer: {feedback.correct_answer}  \n"
                    f"Concept: {humanize_label(feedback.concept_tag)}"
                )
    else:
        st.success("Perfect score. Mastery reached for this checkpoint.")


def stop_mastery_session() -> None:
    if not st.session_state.study_topic:
        return
    reviewed_concepts = _get_recent_mastery_concepts()
    mastery_progress = stop_mastery_loop(
        st.session_state.study_topic,
        st.session_state.weak_concepts,
        reviewed_concepts,
    )
    clear_current_quiz()
    store_remediation_message(None)
    if mastery_progress.study_plan:
        store_study_plan(mastery_progress.study_plan, mastery_progress.status)
        store_study_plan_citations(
            get_supporting_citations(
                f"{st.session_state.study_topic} {' '.join(st.session_state.weak_concepts)}".strip()
            )
        )
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
    )
    st.rerun()


def render_study_plan(study_plan: StudyPlan) -> None:
    st.subheader("Study plan")
    st.write("Reviewed topics:", ", ".join(study_plan.reviewed_topics))
    if study_plan.weak_areas:
        st.write("Weak areas:", ", ".join(study_plan.weak_areas))
    else:
        st.write("Weak areas: none")
    st.write("Recommended review order:")
    for item in study_plan.recommended_order:
        st.markdown(f"- {item}")
    st.write("Suggested next steps:")
    for item in study_plan.suggested_next_steps:
        st.markdown(f"- {item}")
    render_source_list(st.session_state.study_plan_citations)

def render_source_list(citations: list[str], caption: str = "Sources") -> None:
    if not citations:
        return
    st.caption(caption)
    for citation in citations:
        st.markdown(f"- {citation}")


def render_empty_state() -> None:
    if st.session_state.document_library:
        st.markdown("### Open a saved workspace")
        st.write("Select a document from your study library above to continue that chat.")
        return

    st.markdown("### Start here")
    step_column, prompt_column, plan_column = st.columns(3)
    step_column.markdown(
        """
        **1. Upload notes**  
        Add a class handout, reading notes, or a study guide in `pdf`, `txt`, or `md`.
        """
    )
    prompt_column.markdown(
        f"""
        **2. Ask or study**  
        Ask a direct question or start with a mastery prompt like `Help me study metabolism` to launch a {settings.quiz_questions_per_round}-question checkpoint.
        """
    )
    plan_column.markdown(
        """
        **3. Review the results**  
        Use cited answers, quiz feedback, and the final study plan to guide your review.
        """
    )
    render_sample_prompts()


def render_document_workspace_header() -> None:
    header_column, action_column = st.columns([5, 1])
    with header_column:
        active_workspace = next(
            (
                workspace
                for workspace in st.session_state.document_library
                if workspace["document_id"] == st.session_state.active_document_id
            ),
            None,
        )
        workspace_title = (
            active_workspace.get("document_title")
            if active_workspace is not None
            else st.session_state.uploaded_sources[0]
        )
        st.markdown(f"## {workspace_title}")
        st.caption(_build_workspace_context_line(active_workspace))
    with action_column:
        if st.button("Back to library", key="back_to_library"):
            save_active_document_workspace()
            st.session_state.active_document_id = None
            st.session_state.uploaded_sources = []
            st.session_state.chunks = []
            st.session_state.messages = []
            st.session_state.conversation_topic = None
            clear_mastery_session()
            persist_document_library(
                st.session_state.document_library,
                st.session_state.active_document_id,
            )
            st.rerun()


def render_feedback_admin() -> None:
    st.title("Feedback Admin")
    st.caption("Browse recent response feedback captured from the app.")

    limit = st.selectbox("Rows to load", options=[25, 50, 100, 200], index=1)
    feedback_items = load_recent_feedback(limit=limit)

    if not feedback_items:
        st.info("No feedback records found yet.")
        return

    very_helpful = sum(1 for item in feedback_items if item.rating == "Very helpful")
    somewhat_helpful = sum(1 for item in feedback_items if item.rating == "Somewhat helpful")
    not_helpful = sum(1 for item in feedback_items if item.rating == "Not helpful")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Loaded records", str(len(feedback_items)))
    metric_columns[1].metric("Very helpful", str(very_helpful))
    metric_columns[2].metric("Somewhat helpful", str(somewhat_helpful))
    metric_columns[3].metric("Not helpful", str(not_helpful))

    rows = [
        {
            "created_at": item.created_at,
            "rating": item.rating,
            "topic": item.topic or "",
            "mode": item.mode,
            "filename": item.filename or "",
            "query": item.query,
            "response": item.response,
            "feedback_text": item.feedback_text or "",
        }
        for item in feedback_items
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_mastery_status_banner() -> None:
    status = st.session_state.mastery_status
    if status == "completed":
        st.success("Mastery checkpoint completed. Your study plan is ready below.")
    elif status == "stopped":
        st.info("Mastery session stopped. A study plan was created from your latest progress.")
    elif status == "in_progress":
        st.info("Work through the current checkpoint, then review the feedback before the next round.")
    else:
        st.info("Start a mastery request to begin a guided study loop.")


def _build_round_caption(quiz_round: int) -> str:
    if quiz_round <= 1:
        return "Round 1 of the mastery loop"
    return f"Round {quiz_round} of the mastery loop"


def _get_recent_mastery_concepts() -> list[str]:
    if st.session_state.last_quiz_result is not None:
        return [feedback.concept_tag for feedback in st.session_state.last_quiz_result.feedback]
    if st.session_state.current_quiz is not None:
        return [question.concept_tag for question in st.session_state.current_quiz.questions]
    return []


def _should_show_stop_mastery_button() -> bool:
    return (
        st.session_state.mastery_status == "in_progress"
        and st.session_state.current_quiz is not None
    )


def _build_mastery_context_line() -> str:
    topic = st.session_state.study_topic or st.session_state.conversation_topic or "No topic selected"
    status = st.session_state.mastery_status.replace("_", " ").title()
    if st.session_state.current_quiz:
        return (
            f"Mastery session | Topic: {topic} | Status: {status} | "
            f"{_build_round_caption(st.session_state.quiz_round)}"
        )
    return f"Mastery session | Topic: {topic} | Status: {status}"


def _build_workspace_context_line(active_workspace: dict[str, object] | None) -> str:
    document_topic = (
        str(active_workspace.get("document_topic"))
        if active_workspace is not None and active_workspace.get("document_topic")
        else None
    )
    document_title = (
        str(active_workspace.get("document_title"))
        if active_workspace is not None and active_workspace.get("document_title")
        else None
    )
    context_parts = []
    if document_topic and document_topic.casefold() != (document_title or "").casefold():
        context_parts.append(f"Document topic: {document_topic}")
    context_parts.append(f"Mode: {st.session_state.current_mode.title()}")
    if st.session_state.conversation_topic:
        context_parts.append(f"Conversation topic: {st.session_state.conversation_topic}")
    return " | ".join(context_parts)


def main() -> None:
    st.set_page_config(page_title=settings.app_title, page_icon="📚", layout="wide")
    initialize_session_state()
    ensure_current_mode("ask")
    page = st.sidebar.radio("Page", options=["Study Workspace", "Feedback Admin"])
    if not st.session_state.document_library:
        document_library, active_document_id = restore_document_library()
        set_document_library(document_library, active_document_id)
        if document_library:
            st.session_state.library_status_message = (
                f"Restored {len(document_library)} saved workspace"
                f"{'' if len(document_library) == 1 else 's'} from local cache."
            )
    if (
        not st.session_state.document_library
        and st.session_state.uploaded_sources
        and st.session_state.chunks
    ):
        recovered_workspace = build_workspace_from_session()
        if recovered_workspace is not None:
            set_document_library([recovered_workspace], str(recovered_workspace["document_id"]))
            st.session_state.library_status_message = (
                f"Recovered {recovered_workspace['filename']} from the active session."
            )
            persist_document_library(
                st.session_state.document_library,
                st.session_state.active_document_id,
            )
    if not st.session_state.document_library:
        remote_document_library = rebuild_document_library_from_remote()
        if remote_document_library:
            active_document_id = str(remote_document_library[0]["document_id"])
            set_document_library(remote_document_library, active_document_id)
            st.session_state.library_status_message = (
                f"Recovered {len(remote_document_library)} workspace"
                f"{'' if len(remote_document_library) == 1 else 's'} from Pinecone."
            )
            persist_document_library(
                st.session_state.document_library,
                st.session_state.active_document_id,
            )
    if not st.session_state.observability_enabled:
        st.session_state.observability_enabled = initialize_observability()

    if page == "Feedback Admin":
        render_feedback_admin()
        return

    if st.session_state.active_document_id and st.session_state.uploaded_sources:
        render_document_workspace_header()
        _render_active_mode()
    else:
        render_hero()
        render_mode_overview()
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_upload_panel()
        render_document_library()
        render_empty_state()


if __name__ == "__main__":
    main()
