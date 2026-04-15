"""app.py: Streamlit app entrypoint and UI rendering for Study Pal."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.modes.agent import advance_mastery_loop, start_mastery_loop, stop_mastery_loop
from src.feedback.feedback_store import load_recent_feedback, save_response_feedback
from src.data.index_cache import persist_document_library, restore_document_library
from src.modes.mode_router import detect_app_mode, extract_conversation_topic, is_generic_mastery_topic
from src.notes.notes_answering import build_structured_answer_response, get_supporting_citations
from src.notes.notes_upload import index_uploaded_file
from src.core.app_state import (
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
    store_remediation_payload,
    store_study_plan,
    store_study_plan_citations,
)
from src.auth.supabase_auth import (
    complete_sign_in_from_callback,
    is_valid_email_address,
    send_magic_link,
    supabase_enabled,
)
from src.core.config import settings
from src.modes.grading import grade_quiz
from src.core.models import Chunk, FeedbackRating, QuizResult, ResponseFeedback, StudyPlan, StudyQuiz
from src.core.observability import initialize_observability, log_langfuse_event
from src.core.observability import log_langfuse_score
from src.core.utils import humanize_label
from src.data.embeddings import embed_texts, is_embedding_vector
from src.data.vector_store import rebuild_document_library_from_remote, upsert_remote_chunks



def _trim_text(text: str, limit: int = 220) -> str:
    """Trim long metadata fields for clean cards.
    
    Args:
        text (str): Input text to process.
        limit (int): Maximum number of items to return.
    
    Returns:
        str: Formatted text result.
    """

    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."



def _format_document_topic(workspace: dict[str, object]) -> str:
    """Decide how to display the document topic.
    
    Args:
        workspace (dict[str, object]): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    filename = str(workspace.get("filename") or "")
    filename_title = filename.rsplit(".", 1)[0].replace("_", " ").title() if filename else ""
    document_title = str(workspace.get("document_title") or "").strip()
    topic = str(workspace.get("document_topic") or "").strip()
    if not topic:
        return "Not generated yet"
    if topic.lower() in {filename_title.lower(), document_title.lower()}:
        return "Not generated yet"
    return topic



def render_hero() -> None:
    """Render the landing hero section.
    """

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


def _render_auth_panel() -> bool:
    """Render the Supabase magic-link login panel.
    
    Returns:
        bool: True when the user is authenticated or auth is disabled.
    """

    if not supabase_enabled():
        return True

    def _first_query_param(name: str) -> str:
        """Read a Streamlit query param as a single normalized string."""
        raw_value = st.query_params.get(name, "")
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else ""
        return str(raw_value or "").strip()

    def _auth_env() -> str:
        """Return a lightweight environment label for auth tracing."""
        redirect = settings.supabase_redirect_url.strip().lower()
        if "localhost" in redirect or "127.0.0.1" in redirect:
            return "local"
        if ".streamlit.app" in redirect:
            return "streamlit_cloud"
        return "custom"

    callback_shape = "none"
    if _first_query_param("code"):
        callback_shape = "pkce_code"
    elif _first_query_param("token_hash"):
        callback_shape = "token_hash"
    elif _first_query_param("token"):
        callback_shape = "token"
    elif _first_query_param("error") or _first_query_param("error_description"):
        callback_shape = "error"

    # Handle Supabase callback URLs before rendering sign-in controls.
    if not st.session_state.user_id:
        callback_user, callback_error, callback_handled = complete_sign_in_from_callback(st.query_params)
        if callback_shape != "none":
            log_langfuse_event(
                "auth_callback_detected",
                session_id=st.session_state.session_id,
                metadata={
                    "callback_shape": callback_shape,
                    "env": _auth_env(),
                },
            )
        if callback_handled:
            if callback_error:
                st.session_state.auth_error = callback_error
                if not st.session_state.auth_email:
                    st.session_state.auth_email = _first_query_param("email")
                log_langfuse_event(
                    "auth_callback_failure",
                    session_id=st.session_state.session_id,
                    metadata={
                        "callback_shape": callback_shape,
                        "env": _auth_env(),
                        "has_email_hint": bool(st.session_state.auth_email),
                    },
                )
            elif callback_user:
                callback_email = str(callback_user.get("email") or "").strip()
                callback_user_id = str(callback_user.get("id") or callback_email).strip().lower()
                st.session_state.user_email = callback_email or None
                st.session_state.user_id = callback_user_id or None
                st.session_state.auth_email = ""
                st.session_state.auth_code_sent = False
                st.session_state.auth_error = None
                log_langfuse_event(
                    "auth_callback_success",
                    session_id=st.session_state.session_id,
                    metadata={
                        "callback_shape": callback_shape,
                        "env": _auth_env(),
                        "has_email": bool(callback_email),
                    },
                )
            st.query_params.clear()
            st.rerun()

    if st.session_state.user_id and st.session_state.user_email:
        with st.sidebar:
            st.markdown("### Account")
            st.caption(st.session_state.user_email)
            if st.button("Sign out", key="auth_sign_out"):
                st.session_state.user_email = None
                st.session_state.user_id = None
                st.session_state.auth_email = ""
                st.session_state.auth_code_sent = False
                st.session_state.auth_error = None
                st.rerun()
        return True

    with st.sidebar:
        st.markdown("### Authentication")
        st.caption("Sign in on the main panel to continue.")

    st.markdown(
        """
        <style>
          .auth-page-shell {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            background: radial-gradient(circle at 15% 20%, rgba(70,120,220,0.22), rgba(15,20,30,0.95));
            padding: 1.2rem 1.2rem;
            margin-top: 4.5rem;
            margin-bottom: 1.25rem;
          }
          .auth-kicker {
            display: inline-block;
            font-size: 0.78rem;
            color: #bcd8ff;
            border: 1px solid rgba(147, 197, 253, 0.35);
            border-radius: 999px;
            padding: 0.20rem 0.55rem;
            background: rgba(59,130,246,0.16);
            margin-bottom: 0.5rem;
          }
          .auth-subtle {
            color: #c8d1de;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, auth_center, _ = st.columns([1.05, 1.1, 1.05], gap="small")
    with auth_center:
        st.markdown(
            """
            <div class="auth-page-shell">
              <div class="auth-kicker">Private Workspace Access</div>
              <h2 style="margin: 0.2rem 0 0.55rem 0;">Welcome Back</h2>
              <p class="auth-subtle" style="margin: 0 0 0.8rem 0;">
                Sign in with your email magic link to access your saved study library and feedback history.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        email = st.text_input(
            "Email",
            value=st.session_state.auth_email,
            key="auth_email_input",
            placeholder="you@example.com",
        )
        cleaned_email = email.strip()
        email_is_valid = is_valid_email_address(cleaned_email) if cleaned_email else False
        if cleaned_email and not email_is_valid:
            st.error("Enter a valid email address, like name@example.com.")
        if st.button(
            "Send magic link",
            disabled=not cleaned_email or not email_is_valid,
            key="auth_send_link",
            use_container_width=True,
        ):
            st.session_state.auth_email = cleaned_email
            ok, err = send_magic_link(st.session_state.auth_email)
            if ok:
                st.session_state.auth_code_sent = True
                st.session_state.auth_error = None
                log_langfuse_event(
                    "auth_magic_link_sent",
                    session_id=st.session_state.session_id,
                    metadata={"env": _auth_env()},
                )
            else:
                st.session_state.auth_error = err or "Unable to send the sign-in email."
                log_langfuse_event(
                    "auth_magic_link_send_failure",
                    session_id=st.session_state.session_id,
                    metadata={"env": _auth_env()},
                )

    if cleaned_email != st.session_state.auth_email:
        st.session_state.auth_email = cleaned_email
        st.session_state.auth_code_sent = False
        if st.session_state.auth_error:
            st.session_state.auth_error = None

    with auth_center:
        if st.session_state.auth_code_sent:
            st.success("Magic link sent. Open your email and click the Log In button.")

        if st.session_state.auth_error:
            st.error(f"Sign-in issue: {st.session_state.auth_error}")
            if st.button(
                "Resend magic link",
                disabled=(
                    not st.session_state.auth_email
                    or not is_valid_email_address(st.session_state.auth_email)
                ),
                key="auth_resend_link",
                use_container_width=True,
            ):
                ok, err = send_magic_link(st.session_state.auth_email)
                if ok:
                    st.session_state.auth_code_sent = True
                    st.session_state.auth_error = None
                    st.success("New magic link sent.")
                    log_langfuse_event(
                        "auth_magic_link_resent",
                        session_id=st.session_state.session_id,
                        metadata={"env": _auth_env()},
                    )
                else:
                    st.session_state.auth_error = err or "Unable to resend the sign-in email."
                    log_langfuse_event(
                        "auth_magic_link_resend_failure",
                        session_id=st.session_state.session_id,
                        metadata={"env": _auth_env()},
                    )

    return False



def render_mode_overview() -> None:
    """Show the high-level summary of app modes.
    """

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
    """Display example prompts to help users get started.
    """

    st.markdown("#### Try prompts like these")
    prompt_column, mastery_prompt_column = st.columns(2)
    prompt_column.code("What does the notes packet say about eigenvalues?", language="text")
    mastery_prompt_column.code("Help me study photosynthesis for my quiz tomorrow.", language="text")



def render_upload_panel() -> None:
    """Provide the file upload panel and indexing trigger.
    """

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
            user_id=st.session_state.user_id,
        )
        st.rerun()



def _coerce_chunks(raw_chunks: list[object]) -> list[Chunk]:
    """Normalize stored chunks into Chunk models.
    
    Args:
        raw_chunks (list[object]): Input parameter.
    
    Returns:
        list[Chunk]: List of results.
    """

    normalized: list[Chunk] = []
    for chunk in raw_chunks:
        if isinstance(chunk, Chunk):
            normalized.append(chunk)
        else:
            normalized.append(Chunk.model_validate(chunk))
    return normalized



def _reindex_workspace(workspace: dict[str, object]) -> bool:
    """Reindex a workspace's chunks into Pinecone.
    
    Args:
        workspace (dict[str, object]): Input parameter.
    
    Returns:
        bool: True when the check succeeds; otherwise False.
    """

    raw_chunks = workspace.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        st.error("No chunks found to reindex.")
        return False

    chunks = _coerce_chunks(raw_chunks)
    embeddings = embed_texts([chunk.text for chunk in chunks])
    remote_embeddings = [embedding for embedding in embeddings if is_embedding_vector(embedding)]
    if len(remote_embeddings) != len(chunks):
        st.error("Reindexing failed. Embeddings were not generated for every chunk.")
        return False

    upsert_remote_chunks(chunks, remote_embeddings)
    return True



def _run_reindex_request() -> None:
    """Handle queued reindex requests at the top of the library view.
    """

    document_id = st.session_state.get("reindex_request")
    if not document_id:
        return

    workspace = next(
        (item for item in st.session_state.document_library if item["document_id"] == document_id),
        None,
    )
    st.session_state.reindex_request = None

    if workspace is None:
        st.error("Reindexing failed. Could not find that document in the library.")
        return

    filename = workspace.get("filename", "document")
    banner = st.empty()
    banner.info(f"Reindexing {filename} to Pinecone...")
    if _reindex_workspace(workspace):
        banner.success("Reindexing complete.")
    else:
        banner.error("Reindexing failed. Check your embedding settings and try again.")



def render_document_library() -> None:
    """List saved workspaces and let the user open one.
    """

    _run_reindex_request()
    st.markdown("### Your study library")
    st.write("Select a document from your study library below to continue that chat.")
    if st.session_state.library_status_message:
        st.toast(st.session_state.library_status_message)
        st.session_state.library_status_message = None
    if not st.session_state.document_library:
        st.caption("No saved note workspaces yet. Upload a file to create one.")
        return

    columns = st.columns(2)
    for index, workspace in enumerate(st.session_state.document_library):
        column = columns[index % 2]
        with column:
            with st.container(border=True):
                st.markdown(
                    f"#### {workspace.get('document_title') or workspace['filename']}"
                )
                st.caption(workspace["filename"])
                st.caption(
                    f"{workspace['chunk_count']} chunks | {workspace['size_mb']}MB"
                )
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
                        user_id=st.session_state.user_id,
                    )
                    st.rerun()
                if st.button(
                    "Reindex to Pinecone",
                    key=f"reindex_workspace_{workspace['document_id']}",
                ):
                    st.session_state.reindex_request = workspace["document_id"]
                    st.rerun()



def _render_message_history() -> None:
    """Render the chat transcript with citations and feedback tools.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_action_lanes(message)
            render_source_list(message.get("citations", []), caption="Sources used")
            if message["role"] == "assistant":
                render_message_feedback_form(message)



def _render_action_lanes(message: dict[str, object]) -> None:
    """Render info/quiz lane buttons for a structured assistant response.
    
    Args:
        message (dict[str, object]): Input parameter.
    """

    info_lane = message.get("info_lane")
    quiz_lane = message.get("quiz_lane")
    if not isinstance(info_lane, dict) and not isinstance(quiz_lane, dict):
        return

    info_label = None
    info_query = None
    if isinstance(info_lane, dict):
        info_label = str(info_lane.get("button_label") or "").strip()
        info_query = str(info_lane.get("query") or "").strip()

    quiz_label = None
    if isinstance(quiz_lane, dict):
        quiz_label = str(quiz_lane.get("button_label") or "").strip()

    if info_label and quiz_label:
        info_column, quiz_column = st.columns(2)
        with info_column:
            if st.button(info_label, key=f"info_lane_{message.get('id')}"):
                _handle_question(info_query or info_label)
        with quiz_column:
            if st.button(quiz_label, key=f"quiz_lane_{message.get('id')}"):
                _start_mastery_from_lane(message)
        return

    if info_label:
        if st.button(info_label, key=f"info_lane_{message.get('id')}"):
            _handle_question(info_query or info_label)
        return

    if quiz_label:
        if st.button(quiz_label, key=f"quiz_lane_{message.get('id')}"):
            _start_mastery_from_lane(message)



def _rotate_info_lane_label(structured) -> None:
    """Rotate info-lane labels so they don't repeat back-to-back.
    
    Args:
        structured: Input parameter.
    """

    if structured.info_lane is None:
        return

    label = (structured.info_lane.button_label or "").strip()
    if not label:
        return

    last_label = (st.session_state.last_info_lane_label or "").strip()
    if not last_label:
        st.session_state.last_info_lane_label = label
        return
        """Strip emoji.
        
        Args:
            text (str): Input text to process.
        
        Returns:
            str: Formatted text result.
        """


    def _strip_emoji(text: str) -> str:
        """Strip emoji.
        
        Args:
            text (str): Input text to process.
        
        Returns:
            str: Formatted text result.
        """

        parts = text.split(" ", 1)
        if len(parts) == 2 and len(parts[0]) <= 3:
            return parts[1].strip()
        return text.strip()

    normalized_label = _strip_emoji(label).lower()
    normalized_last = _strip_emoji(last_label).lower()
    if normalized_label and normalized_label == normalized_last:
        base_text = _strip_emoji(label)
        emoji_cycle = ["🧠", "🔍", "🧩", "⚡", "🧭", "🌀", "📌"]
        template_cycle = [
            "{emoji} In action: {text}",
            "{emoji} Why it matters: {text}",
            "{emoji} Quick intuition: {text}",
            "{emoji} A closer look at {text}",
            "{emoji} The hidden angle: {text}",
        ]
        emoji_index = st.session_state.info_lane_variant_index % len(emoji_cycle)
        template_index = st.session_state.info_lane_variant_index % len(template_cycle)
        emoji = emoji_cycle[emoji_index]
        template = template_cycle[template_index]
        structured.info_lane.button_label = template.format(emoji=emoji, text=base_text)
        st.session_state.info_lane_variant_index += 1
        st.session_state.last_info_lane_label = structured.info_lane.button_label
        return

    st.session_state.last_info_lane_label = label



def render_message_feedback_form(message: dict[str, object]) -> None:
    """Show the feedback form under an assistant message.
    
    Args:
        message (dict[str, object]): Input parameter.
    """

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
    """Pick a reasonable topic based on recent conversation context.
    
    Returns:
        str | None: Result value.
    """

    explicit_topic = st.session_state.study_topic or st.session_state.conversation_topic
    if explicit_topic and not is_generic_mastery_topic(explicit_topic):
        return explicit_topic

    for message in reversed(st.session_state.messages):
        topic = message.get("topic")
        if isinstance(topic, str) and topic and not is_generic_mastery_topic(topic):
            return topic
    return None



def _handle_question(question: str) -> None:
    """Route a new user question into Ask or Mastery mode.
    
    Args:
        question (str): The user question to answer.
    """

    detected_mode = detect_app_mode(question)
    previous_topic = _get_recent_topic_context()
    conversation_topic = extract_conversation_topic(question, fallback_topic=previous_topic)
    log_langfuse_event(
        "question_received",
        session_id=st.session_state.session_id,
        metadata={
            "detected_mode": detected_mode,
            "question_chars": len(question),
            "topic": conversation_topic or "",
        },
    )
    if detected_mode == "mastery":
        _enter_mastery_mode(question, previous_topic)
        return

    clear_mastery_session()
    set_current_mode(detected_mode)
    set_conversation_topic(conversation_topic)
    store_message("user", question, topic=conversation_topic)
    structured = build_structured_answer_response(question)
    _rotate_info_lane_label(structured)
    answer_message = structured.answer
    answer_citations = structured.citations
    store_message(
        "assistant",
        answer_message,
        citations=answer_citations,
        topic=conversation_topic,
        query=question,
        mode=detected_mode,
        info_lane=structured.info_lane.model_dump() if structured.info_lane else None,
        quiz_lane=structured.quiz_lane.model_dump() if structured.quiz_lane else None,
        trace_id=structured.trace_id,
        observation_id=structured.observation_id,
        topic_subject=structured.topic_subject,
    )
    st.session_state.conversation_topic = conversation_topic
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
        user_id=st.session_state.user_id,
    )
    st.rerun()


def _enter_mastery_mode(
    question: str,
    fallback_topic: str | None,

    goal_text: str | None = None,
) -> None:
    """Start the mastery loop and store the initial messages.
    
    Args:
        question (str): The user question to answer.
        fallback_topic (str | None): Input parameter.
        goal_text (str | None): Input parameter.
    """

    try:
        mastery_session, mastery_progress = start_mastery_loop(
            question,
            fallback_topic=fallback_topic,
            session_id=st.session_state.session_id,
        )
    except TypeError:
        # Backward-compatibility path for older test doubles or helper signatures.
        mastery_session, mastery_progress = start_mastery_loop(
            question,
            fallback_topic=fallback_topic,
        )
    set_current_mode("mastery")
    set_conversation_topic(mastery_session.topic)
    st.session_state.quiz_goal = goal_text or mastery_session.topic
    store_message("user", question, topic=mastery_session.topic)
    store_mastery_session(mastery_session)
    st.session_state.quiz_history = []
    st.session_state.quiz_view_round = None
    if mastery_progress.next_quiz and mastery_progress.next_quiz_round:
        store_current_quiz(mastery_progress.next_quiz, mastery_progress.next_quiz_round)
        st.session_state.quiz_view_round = mastery_progress.next_quiz_round
    store_message(
        "assistant",
        mastery_session.intro_message,
        citations=[],
        topic=mastery_session.topic,
        query=question,
        mode="mastery",
        trace_id=mastery_session.trace_id,
        observation_id=mastery_session.observation_id,
    )
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
        user_id=st.session_state.user_id,
    )
    st.rerun()



def _start_mastery_from_lane(message: dict[str, object]) -> None:
    """Start mastery from lane.
    
    Args:
        message (dict[str, object]): Input parameter.
    """

    topic = (
        message.get("topic_subject")
        or message.get("topic")
        or _get_recent_topic_context()
    )
    if not isinstance(topic, str) or not topic.strip():
        return
    question = f"Quiz me on {topic}"
    _enter_mastery_mode(question, topic, goal_text=topic)



def render_chat() -> None:
    """Render the chat UI and accept new questions.
    """

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
    """Persist feedback submitted for an assistant response.
    
    Args:
        message (dict[str, object]): Input parameter.
        rating (FeedbackRating): Input parameter.
        feedback_text (str): Input parameter.
    """

    message_id = str(message["id"])
    feedback_record = ResponseFeedback(
        message_id=message_id,
        session_id=st.session_state.session_id,
        user_id=st.session_state.user_id,
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
    _log_feedback_score(message, rating, feedback_text)
    persist_document_library(
        st.session_state.document_library,
        st.session_state.active_document_id,
        user_id=st.session_state.user_id,
    )
    st.success("Thanks. Your feedback was saved.")



def _log_feedback_score(message: dict[str, object], rating: FeedbackRating, feedback_text: str) -> None:
    """Log feedback score.
    
    Args:
        message (dict[str, object]): Input parameter.
        rating (FeedbackRating): Input parameter.
        feedback_text (str): Input parameter.
    """

    rating_map = {
        "Very helpful": 2,
        "Somewhat helpful": 1,
        "Not helpful": 0,
    }
    score = rating_map.get(rating)
    if score is None:
        return
    trace_id = message.get("trace_id")
    observation_id = message.get("observation_id")
    if not isinstance(trace_id, str):
        trace_id = None
    if not isinstance(observation_id, str):
        observation_id = None
    log_langfuse_score(
        name="user_feedback",
        value=score,
        trace_id=trace_id,
        observation_id=observation_id,
        session_id=st.session_state.session_id,
        comment=feedback_text.strip() or None,
        metadata={
            "rating_label": rating,
            "message_id": str(message.get("id") or ""),
            "mode": str(message.get("mode") or st.session_state.current_mode),
        },
    )



def _render_active_mode() -> None:
    """Render the current mode's main content area.
    """

    render_chat()



def render_mastery_inline() -> None:
    """Render the mastery UI elements within the chat view.
    """

    if (
        st.session_state.mastery_status == "idle"
        and not st.session_state.current_quiz
        and not st.session_state.last_quiz_result
        and not st.session_state.study_plan
    ):
        return

    _render_quiz_goal_line()

    _render_mastery_status_banner()
    if st.session_state.mastery_status in {"completed", "stopped"}:
        st.caption("Start a new mastery prompt in chat if you want to run another round on a new focus area.")
    if st.session_state.current_quiz or st.session_state.quiz_history:
        render_quiz_card()
    if st.session_state.study_plan:
        render_study_plan(st.session_state.study_plan)



def render_quiz_panel(quiz: StudyQuiz, quiz_round: int) -> None:
    """Render a quiz form for the current mastery round.
    
    Args:
        quiz (StudyQuiz): Quiz payload to process.
        quiz_round (int): Input parameter.
    """

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



def _coerce_quiz_result(raw_result: object) -> QuizResult | None:
    """Coerce quiz result.
    
    Args:
        raw_result (object): Input parameter.
    
    Returns:
        QuizResult | None: Result value.
    """

    if raw_result is None:
        return None
    if isinstance(raw_result, QuizResult):
        return raw_result
    try:
        return QuizResult.model_validate(raw_result)
    except Exception:
        return None



def _coerce_study_quiz(raw_quiz: object) -> StudyQuiz | None:
    """Coerce study quiz.
    
    Args:
        raw_quiz (object): Input parameter.
    
    Returns:
        StudyQuiz | None: Result value.
    """

    if raw_quiz is None:
        return None
    if isinstance(raw_quiz, StudyQuiz):
        return raw_quiz
    try:
        return StudyQuiz.model_validate(raw_quiz)
    except Exception:
        return None


def _upsert_quiz_history(
    *,
    quiz_round: int,
    quiz: StudyQuiz,
    quiz_result: QuizResult,
    remediation_message: str | None,
    remediation_payload: dict[str, str] | None,

    remediation_citations: list[str] | None,
) -> None:
    """Upsert quiz history.
    
    Args:
        quiz_round (int): Input parameter.
        quiz (StudyQuiz): Quiz payload to process.
        quiz_result (QuizResult): Input parameter.
        remediation_message (str | None): Input parameter.
        remediation_payload (dict[str, str] | None): Input parameter.
        remediation_citations (list[str] | None): Input parameter.
    """

    history = list(st.session_state.quiz_history)
    entry = {
        "round": quiz_round,
        "quiz": quiz,
        "result": quiz_result,
        "remediation_message": remediation_message,
        "remediation_payload": remediation_payload,
        "remediation_citations": remediation_citations or [],
    }
    for index, existing in enumerate(history):
        if existing.get("round") == quiz_round:
            history[index] = entry
            break
    else:
        history.append(entry)
    history.sort(key=lambda item: int(item.get("round", 0)))
    st.session_state.quiz_history = history



def _available_quiz_rounds() -> list[int]:
    """Available quiz rounds.
    
    Returns:
        list[int]: List of results.
    """

    rounds = [
        int(entry.get("round"))
        for entry in st.session_state.quiz_history
        if isinstance(entry.get("round"), int)
    ]
    if st.session_state.current_quiz and st.session_state.quiz_round:
        rounds.append(int(st.session_state.quiz_round))
    return sorted(set(rounds))



def _resolve_quiz_view_round(rounds: list[int]) -> int | None:
    """Resolve quiz view round.
    
    Args:
        rounds (list[int]): Input parameter.
    
    Returns:
        int | None: Result value.
    """

    if not rounds:
        return None
    selected = st.session_state.quiz_view_round
    if isinstance(selected, int) and selected in rounds:
        return selected
    if st.session_state.current_quiz and st.session_state.quiz_round in rounds:
        return int(st.session_state.quiz_round)
    return rounds[-1]



def _get_quiz_history_entry(round_number: int) -> dict[str, object] | None:
    """Get quiz history entry.
    
    Args:
        round_number (int): Input parameter.
    
    Returns:
        dict[str, object] | None: Mapping of computed results.
    """

    for entry in st.session_state.quiz_history:
        if entry.get("round") == round_number:
            return entry
    return None



def render_quiz_card() -> None:
    """Render quiz card.
    """

    rounds = _available_quiz_rounds()
    selected_round = _resolve_quiz_view_round(rounds)
    if selected_round is None:
        return

    st.session_state.quiz_view_round = selected_round
    entry = _get_quiz_history_entry(selected_round)
    quiz = _coerce_study_quiz(entry.get("quiz") if entry else None)
    result = _coerce_quiz_result(entry.get("result") if entry else None)
    remediation_message = entry.get("remediation_message") if entry else None
    remediation_payload = entry.get("remediation_payload") if entry else None
    remediation_citations = entry.get("remediation_citations") if entry else []

    if selected_round == st.session_state.quiz_round and st.session_state.current_quiz:
        quiz = st.session_state.current_quiz

    header_left, header_center, header_right = st.columns([2, 6, 3])
    with header_left:
        arrow_left, arrow_right = st.columns(2)
        with arrow_left:
            if st.button("◀", key="quiz_round_prev", disabled=selected_round == rounds[0]):
                st.session_state.quiz_view_round = rounds[max(0, rounds.index(selected_round) - 1)]
                st.rerun()
        with arrow_right:
            if st.button("▶", key="quiz_round_next", disabled=selected_round == rounds[-1]):
                st.session_state.quiz_view_round = rounds[min(len(rounds) - 1, rounds.index(selected_round) + 1)]
                st.rerun()
    with header_center:
        score_label = "Pending"
        if result:
            score_label = f"{result.score}/{result.total}"
        st.markdown(
            f"<div style='text-align:center;'><h3>Round {selected_round} • Score {score_label}</h3></div>",
            unsafe_allow_html=True,
        )
    with header_right:
        if _should_show_stop_mastery_button():
            if st.button("Stop and build study plan", key="stop_mastery"):
                stop_mastery_session()

    with st.container(border=True):
        if quiz is None:
            st.caption("Quiz details will appear once this round starts.")
            return
        _render_quiz_questions(quiz, selected_round, result)
        if remediation_message:
            should_expand = bool(result and result.score < result.total)
            with st.expander("Reteach", expanded=should_expand):
                st.markdown(remediation_message)
                _render_mini_check(remediation_payload, selected_round)
                render_source_list(remediation_citations, caption="Sources used")


def _render_quiz_questions(
    quiz: StudyQuiz,
    quiz_round: int,

    quiz_result: QuizResult | None,
) -> None:
    """Render quiz questions.
    
    Args:
        quiz (StudyQuiz): Quiz payload to process.
        quiz_round (int): Input parameter.
        quiz_result (QuizResult | None): Input parameter.
    """

    st.caption(quiz.title)
    if quiz_result and quiz_result.feedback:
        for index, question in enumerate(quiz.questions, start=1):
            feedback = quiz_result.feedback[index - 1] if index - 1 < len(quiz_result.feedback) else None
            result_icon = "✅" if feedback and feedback.is_correct else "❌"
            st.markdown(f"{result_icon} **Question {index}.** {question.prompt}")
            if feedback and feedback.user_answer:
                st.markdown(f"Your answer: {feedback.user_answer}")
            if feedback:
                st.markdown(f"Correct answer: {feedback.correct_answer}")
        return

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

        submitted = st.form_submit_button("Grade quiz")

    if submitted:
        answers = [
            st.session_state.get(f"quiz_question_{quiz_round}_{index}")
            for index in range(1, len(quiz.questions) + 1)
        ]
        _submit_quiz_answers(quiz, quiz_round, answers)



def _render_mini_check(remediation_payload: dict[str, str] | None, round_number: int) -> None:
    """Render mini check.
    
    Args:
        remediation_payload (dict[str, str] | None): Input parameter.
        round_number (int): Input parameter.
    """

    if not remediation_payload:
        return
    mini_check = remediation_payload.get("mini_check")
    mini_check_answer = remediation_payload.get("mini_check_answer")
    if not mini_check or not mini_check_answer:
        return
    normalized_answer = str(mini_check_answer).strip().lower()
    if normalized_answer not in {"yes", "no"}:
        return

    st.markdown(f"**Mini-check:** {mini_check}")
    response_key = f"mini_check_response_{round_number}"
    correct_key = f"mini_check_correct_{round_number}"
    col_yes, col_no = st.columns(2)
    is_correct = st.session_state.get(correct_key) is True
    with col_yes:
        if st.button("Yes", key=f"mini_check_yes_{round_number}", disabled=is_correct):
            st.session_state[response_key] = "yes"
    with col_no:
        if st.button("No", key=f"mini_check_no_{round_number}", disabled=is_correct):
            st.session_state[response_key] = "no"

    user_choice = st.session_state.get(response_key)
    if user_choice:
        if user_choice == normalized_answer:
            st.success("Correct. Nice work!")
            st.session_state[correct_key] = True
        else:
            st.warning("Not quite. Re-read the reteach and try again.")



def _submit_quiz_answers(quiz: StudyQuiz, quiz_round: int, answers: list[str | None]) -> None:
    # Keep quiz submission state changes together so the flow is easier to review and test.
    """Grade quiz answers and advance mastery state.
    
    Args:
        quiz (StudyQuiz): Quiz payload to process.
        quiz_round (int): Input parameter.
        answers (list[str | None]): User-provided answers.
    """

    quiz_result = grade_quiz(quiz, answers)
    store_quiz_result(quiz_result)
    try:
        mastery_progress = advance_mastery_loop(
            quiz.topic,
            quiz_result,
            quiz_round,
            session_id=st.session_state.session_id,
        )
    except TypeError:
        mastery_progress = advance_mastery_loop(
            quiz.topic,
            quiz_result,
            quiz_round,
        )
    citation_query = f"{quiz.topic} {' '.join(quiz_result.weak_concepts)}".strip()
    store_remediation_message(mastery_progress.remediation_message)
    store_remediation_payload(mastery_progress.remediation_payload)
    store_remediation_citations(get_supporting_citations(citation_query))
    _upsert_quiz_history(
        quiz_round=quiz_round,
        quiz=quiz,
        quiz_result=quiz_result,
        remediation_message=mastery_progress.remediation_message,
        remediation_payload=mastery_progress.remediation_payload,
        remediation_citations=st.session_state.remediation_citations,
    )
    st.session_state.quiz_view_round = quiz_round
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
        user_id=st.session_state.user_id,
    )
    st.rerun()



def render_quiz_result(quiz_result: QuizResult) -> None:
    """Show the latest quiz score and feedback.
    
    Args:
        quiz_result (QuizResult): Input parameter.
    """

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
    """Stop the mastery loop and generate a study plan.
    """

    if not st.session_state.study_topic:
        return
    reviewed_concepts = _get_recent_mastery_concepts()
    try:
        mastery_progress = stop_mastery_loop(
            st.session_state.study_topic,
            st.session_state.weak_concepts,
            reviewed_concepts,
            session_id=st.session_state.session_id,
        )
    except TypeError:
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
        user_id=st.session_state.user_id,
    )
    st.rerun()



def render_study_plan(study_plan: StudyPlan) -> None:
    """Render the final study plan and citations.
    
    Args:
        study_plan (StudyPlan): Input parameter.
    """

    st.subheader("Session wrap-up")
    st.metric("Mastery score", study_plan.mastery_score)
    st.write(study_plan.summary)
    st.write("Strengths:", ", ".join(study_plan.strengths))
    if study_plan.weak_areas:
        st.write("Weak areas:", ", ".join(study_plan.weak_areas))
    else:
        st.write("Weak areas: none")
    if study_plan.next_step_lane:
        if st.button(
            study_plan.next_step_lane.button_label,
            key="study_plan_next_step",
        ):
            _handle_question(study_plan.next_step_lane.query)
    render_source_list(st.session_state.study_plan_citations)


def render_source_list(citations: list[str], caption: str = "Sources") -> None:
    """Render a list of citations with an optional caption.
    
    Args:
        citations (list[str]): Citation strings to format or display.
        caption (str): Input parameter.
    """

    if not citations:
        return
    st.caption(caption)
    for citation in citations:
        st.markdown(f"- {citation}")



def render_empty_state() -> None:
    """Render onboarding content when no workspace is loaded.
    """

    if st.session_state.document_library:
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
    """Render the active document header and navigation controls.
    """

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
                user_id=st.session_state.user_id,
            )
            st.rerun()



def render_feedback_admin() -> None:
    """Render the feedback admin page for browsing saved feedback.
    """

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



def _latest_quiz_round() -> int | None:
    """Show the current mastery status banner.
    
    Returns:
        int | None: Result value.
    """

    selected = st.session_state.quiz_view_round
    if isinstance(selected, int):
        return selected
    rounds = [
        int(entry.get("round"))
        for entry in st.session_state.quiz_history
        if isinstance(entry.get("round"), int)
    ]
    if rounds:
        return max(rounds)
    if st.session_state.quiz_round:
        return int(st.session_state.quiz_round)
    return None



def _render_mastery_status_banner() -> None:
    """Render mastery status banner.
    """

    status = st.session_state.mastery_status
    if status == "completed":
        latest_round = _latest_quiz_round() or 1
        if latest_round > 1:
            st.success("Checkpoint complete - you worked through all rounds.")
            st.markdown("Ask another question to start a fresh checkpoint.")
        else:
            st.success("Perfect score - you're done with this checkpoint.")
            st.markdown(
                "Want to go deeper or try a new focus area? Click the deep dive button below or ask another question."
            )
    elif status == "stopped":
        st.info("Mastery session stopped. A study plan was created from your latest progress.")
    elif status == "in_progress":
        st.info("Work through this round, review feedback, and see progress.")
    else:
        st.info("Start a mastery request to begin a guided study loop.")



def _build_round_caption(quiz_round: int) -> str:
    """Build the human-friendly label for the quiz round.
    
    Args:
        quiz_round (int): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    if quiz_round <= 1:
        return "Round 1 of the mastery loop"
    return f"Round {quiz_round} of the mastery loop"



def _get_recent_mastery_concepts() -> list[str]:
    """Collect the most recent mastery concepts for summarizing progress.
    
    Returns:
        list[str]: List of results.
    """

    if st.session_state.last_quiz_result is not None:
        return [feedback.concept_tag for feedback in st.session_state.last_quiz_result.feedback]
    if st.session_state.current_quiz is not None:
        return [question.concept_tag for question in st.session_state.current_quiz.questions]
    return []



def _should_show_stop_mastery_button() -> bool:
    """Decide if the stop-mastery button should be shown.
    
    Returns:
        bool: True when the check succeeds; otherwise False.
    """

    return (
        st.session_state.mastery_status == "in_progress"
        and st.session_state.current_quiz is not None
    )



def _render_quiz_goal_line() -> None:
    """Render the quiz goal line at the top of the mastery view.
    """

    goal = (
        st.session_state.quiz_goal
        or st.session_state.study_topic
        or st.session_state.conversation_topic
    )
    if not goal:
        return
    verb_prefixes = ("learn", "recall", "explore", "review", "master", "practice")
    normalized_goal = str(goal).strip()
    if normalized_goal.lower().startswith(verb_prefixes):
        formatted_goal = normalized_goal
    else:
        formatted_goal = f"Recall {normalized_goal}"
    st.markdown(f"**Quiz Goal: {formatted_goal}**")



def _build_workspace_context_line(active_workspace: dict[str, object] | None) -> str:
    """Build the workspace context line shown under the header.
    
    Args:
        active_workspace (dict[str, object] | None): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

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
    """Entry point for the Streamlit app.
    """

    st.set_page_config(page_title=settings.app_title, page_icon="📚", layout="wide")
    initialize_session_state()
    if not _render_auth_panel():
        return
    ensure_current_mode("ask")
    page = st.sidebar.radio("Page", options=["Study Workspace", "Feedback Admin"])
    if not st.session_state.document_library:
        document_library, active_document_id = restore_document_library(
            user_id=st.session_state.user_id
        )
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
                user_id=st.session_state.user_id,
            )
    if not st.session_state.document_library:
        remote_document_library = rebuild_document_library_from_remote(
            user_id=st.session_state.user_id
        )
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
                user_id=st.session_state.user_id,
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
