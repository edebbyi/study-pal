from __future__ import annotations

import app
import src.agent as agent_module
import src.app_state as app_state_module
from src.models import (
    Chunk,
    MasteryProgress,
    MasterySession,
    QuizQuestion,
    QuizResult,
    StudyPlan,
    StudyQuiz,
    TeachingResponse,
)


class SessionState(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


def _build_quiz(title: str, topic: str) -> StudyQuiz:
    return StudyQuiz(
        title=title,
        topic=topic,
        questions=[
            QuizQuestion(
                prompt="What is the key idea?",
                options=["Correct concept", "Wrong concept"],
                correct_answer="Correct concept",
                concept_tag=topic,
            ),
            QuizQuestion(
                prompt="What should you review next?",
                options=["Core vocabulary", "Ignore the topic"],
                correct_answer="Core vocabulary",
                concept_tag=f"{topic} review",
            ),
        ],
    )


def _initialize_fake_session_state(monkeypatch) -> SessionState:
    state = SessionState()
    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(app.st, "rerun", lambda: None)
    monkeypatch.setattr(app_state_module.st, "session_state", state, raising=False)
    monkeypatch.setattr(app_state_module.st, "success", lambda *args, **kwargs: None)
    app_state_module.initialize_session_state()
    return state


def test_handle_question_routes_ask_query_and_clears_mastery_state(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.study_topic = "Old topic"
    state.mastery_intro = "Old intro"
    state.mastery_status = "in_progress"
    state.weak_concepts = ["old concept"]
    state.current_quiz = _build_quiz("Old Quiz", "Old topic")

    monkeypatch.setattr(
        app,
        "build_answer_response",
        lambda question: TeachingResponse(
            answer=f"Answer: {question}",
            citations=["notes.pdf p.1"],
        ),
    )

    app._handle_question("What does this definition mean?")

    assert state.current_mode == "ask"
    assert state.conversation_topic == "What Does This Definition Mean"
    assert state.study_topic is None
    assert state.current_quiz is None
    assert state.mastery_status == "idle"
    assert state.messages[0] == {
        "role": "user",
        "content": "What does this definition mean?",
        "topic": "What Does This Definition Mean",
    }
    assert state.messages[1]["role"] == "assistant"
    assert state.messages[1]["content"] == "Answer: What does this definition mean?"
    assert state.messages[1]["citations"] == ["notes.pdf p.1"]
    assert state.messages[1]["topic"] == "What Does This Definition Mean"
    assert state.messages[1]["query"] == "What does this definition mean?"
    assert state.messages[1]["mode"] == "ask"
    assert isinstance(state.messages[1]["id"], str)


def test_handle_question_routes_mastery_query_through_agent_start(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    quiz = _build_quiz("Checkpoint Quiz: Photosynthesis", "Photosynthesis")
    mastery_session = MasterySession(
        topic="Photosynthesis",
        intro_message="Let's study photosynthesis.",
        citations=["notes.pdf p.1"],
        status="in_progress",
    )

    monkeypatch.setattr(
        agent_module,
        "start_mastery_session",
        lambda question, fallback_topic=None: mastery_session,
    )
    monkeypatch.setattr(agent_module, "generate_quiz", lambda topic: quiz)

    app._handle_question("Help me study photosynthesis")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Photosynthesis"
    assert state.study_topic == "Photosynthesis"
    assert state.mastery_intro == "Let's study photosynthesis."
    assert state.mastery_intro_citations == ["notes.pdf p.1"]
    assert state.current_quiz == quiz
    assert state.quiz_round == 1
    assert state.messages[-1]["role"] == "assistant"
    assert state.messages[-1]["content"] == "Let's study photosynthesis."
    assert state.messages[-1]["topic"] == "Photosynthesis"
    assert state.messages[-1]["query"] == "Help me study photosynthesis"
    assert state.messages[-1]["mode"] == "mastery"
    assert isinstance(state.messages[-1]["id"], str)


def test_handle_question_reuses_previous_topic_for_referential_mastery_prompt(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = "Motor Function"

    quiz = _build_quiz("Checkpoint Quiz: Motor Function", "Motor Function")
    mastery_session = MasterySession(
        topic="Motor Function",
        intro_message="Let's quiz your understanding of motor function.",
        citations=["notes.pdf p.32"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "quiz me on this please"
        assert fallback_topic == "Motor Function"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("quiz me on this please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Motor Function"
    assert state.study_topic == "Motor Function"
    assert state.current_quiz == quiz
    assert state.messages[0] == {
        "role": "user",
        "content": "quiz me on this please",
        "topic": "Motor Function",
    }
    assert state.messages[-1]["role"] == "assistant"
    assert state.messages[-1]["content"] == "Let's quiz your understanding of motor function."
    assert state.messages[-1]["topic"] == "Motor Function"
    assert state.messages[-1]["query"] == "quiz me on this please"
    assert state.messages[-1]["mode"] == "mastery"
    assert isinstance(state.messages[-1]["id"], str)


def test_handle_question_routes_generate_quiz_prompt_into_mastery(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = "Medulla"

    quiz = _build_quiz("Checkpoint Quiz: Medulla", "Medulla")
    mastery_session = MasterySession(
        topic="Medulla",
        intro_message="Let's start a medulla checkpoint.",
        citations=["notes.pdf p.13"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "generate a quiz please"
        assert fallback_topic == "Medulla"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("generate a quiz please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Medulla"
    assert state.study_topic == "Medulla"
    assert state.current_quiz == quiz
    assert state.messages[0] == {
        "role": "user",
        "content": "generate a quiz please",
        "topic": "Medulla",
    }


def test_handle_question_routes_generate_quiz_for_me_prompt_into_mastery(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = "Medulla"

    quiz = _build_quiz("Checkpoint Quiz: Medulla", "Medulla")
    mastery_session = MasterySession(
        topic="Medulla",
        intro_message="Let's start a medulla checkpoint.",
        citations=["notes.pdf p.13"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "generate a quiz for me please"
        assert fallback_topic == "Medulla"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("generate a quiz for me please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Medulla"
    assert state.study_topic == "Medulla"
    assert state.current_quiz == quiz
    assert state.messages[0] == {
        "role": "user",
        "content": "generate a quiz for me please",
        "topic": "Medulla",
    }


def test_handle_question_routes_make_a_quiz_on_this_topic_prompt_into_mastery(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = "Spinal Cord"

    quiz = _build_quiz("Checkpoint Quiz: Spinal Cord", "Spinal Cord")
    mastery_session = MasterySession(
        topic="Spinal Cord",
        intro_message="Let's start a spinal cord checkpoint.",
        citations=["notes.pdf p.125"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "make a quiz on this topic please"
        assert fallback_topic == "Spinal Cord"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("make a quiz on this topic please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Spinal Cord"
    assert state.study_topic == "Spinal Cord"
    assert state.current_quiz == quiz
    assert state.messages[0] == {
        "role": "user",
        "content": "make a quiz on this topic please",
        "topic": "Spinal Cord",
    }


def test_handle_question_routes_make_me_a_quiz_please_into_mastery_with_fallback_topic(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = "Spinal Cord"

    quiz = _build_quiz("Checkpoint Quiz: Spinal Cord", "Spinal Cord")
    mastery_session = MasterySession(
        topic="Spinal Cord",
        intro_message="Let's start a spinal cord checkpoint.",
        citations=["notes.pdf p.125"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "make me a quiz please"
        assert fallback_topic == "Spinal Cord"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("make me a quiz please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Spinal Cord"
    assert state.study_topic == "Spinal Cord"
    assert state.current_quiz == quiz
    assert state.messages[0] == {
        "role": "user",
        "content": "make me a quiz please",
        "topic": "Spinal Cord",
    }


def test_handle_question_uses_recent_message_topic_when_explicit_topic_state_is_missing(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.conversation_topic = None
    state.study_topic = None
    state.messages = [
        {"role": "user", "content": "what is the medulla?", "topic": "Medulla"},
        {"role": "assistant", "content": "The medulla controls vital functions.", "topic": "Medulla"},
    ]

    quiz = _build_quiz("Checkpoint Quiz: Medulla", "Medulla")
    mastery_session = MasterySession(
        topic="Medulla",
        intro_message="Let's start a medulla checkpoint.",
        citations=["notes.pdf p.13"],
        status="in_progress",
    )

    def fake_start_mastery_loop(
        question: str,
        fallback_topic: str | None = None,
    ) -> tuple[MasterySession, MasteryProgress]:
        assert question == "generate a quiz for me please"
        assert fallback_topic == "Medulla"
        return (
            mastery_session,
            MasteryProgress(
                quiz_result=QuizResult(score=0, total=0, weak_concepts=[], feedback=[]),
                remediation_message=None,
                next_quiz=quiz,
                next_quiz_round=1,
                study_plan=None,
                status="in_progress",
            ),
        )

    monkeypatch.setattr(app, "start_mastery_loop", fake_start_mastery_loop)

    app._handle_question("generate a quiz for me please")

    assert state.current_mode == "mastery"
    assert state.conversation_topic == "Medulla"
    assert state.study_topic == "Medulla"
    assert state.current_quiz == quiz


def test_submit_quiz_answers_advances_mastery_loop(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    current_quiz = _build_quiz("Checkpoint Quiz: Photosynthesis", "Photosynthesis")
    next_quiz = _build_quiz("Reinforcement Quiz: Photosynthesis", "Photosynthesis")

    def fake_advance_mastery_progress(
        topic: str,
        quiz_result: QuizResult,
        current_round: int,
    ) -> MasteryProgress:
        assert topic == "Photosynthesis"
        assert current_round == 1
        assert quiz_result.score == 0
        return MasteryProgress(
            quiz_result=quiz_result,
            remediation_message="Review how light reactions power sugar production.",
            next_quiz=next_quiz,
            next_quiz_round=2,
            study_plan=None,
            status="in_progress",
        )

    monkeypatch.setattr(agent_module, "advance_mastery_progress", fake_advance_mastery_progress)
    monkeypatch.setattr(
        app,
        "get_supporting_citations",
        lambda query: ["notes.pdf p.2"] if "Photosynthesis" in query else [],
    )

    app._submit_quiz_answers(current_quiz, 1, ["Wrong concept", "Ignore the topic"])

    assert state.last_quiz_result is not None
    assert state.last_quiz_result.score == 0
    assert state.mastery_status == "in_progress"
    assert state.remediation_message == "Review how light reactions power sugar production."
    assert state.remediation_citations == ["notes.pdf p.2"]
    assert state.current_quiz == next_quiz
    assert state.quiz_round == 2


def test_submit_quiz_answers_and_stop_session_build_study_plans(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    current_quiz = _build_quiz("Checkpoint Quiz: Photosynthesis", "Photosynthesis")
    generated_plan = StudyPlan(
        topic="Photosynthesis",
        reviewed_topics=["Photosynthesis"],
        weak_areas=[],
        recommended_order=["Light reactions", "Calvin cycle"],
        suggested_next_steps=["Review the diagram", "Explain the process out loud"],
    )
    stopped_plan = StudyPlan(
        topic="Photosynthesis",
        reviewed_topics=["Photosynthesis"],
        weak_areas=["light reactions"],
        recommended_order=["Light reactions", "Calvin cycle"],
        suggested_next_steps=["Redo the missed questions"],
    )

    monkeypatch.setattr(
        agent_module,
        "advance_mastery_progress",
        lambda topic, quiz_result, current_round: MasteryProgress(
            quiz_result=quiz_result,
            remediation_message=None,
            next_quiz=None,
            next_quiz_round=None,
            study_plan=None,
            status="completed",
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "build_study_plan",
        lambda topic, weak_concepts, reviewed_concepts=None: generated_plan,
    )
    monkeypatch.setattr(
        app,
        "get_supporting_citations",
        lambda query: ["notes.pdf p.3"] if not query.endswith("light reactions") else ["notes.pdf p.4"],
    )

    app._submit_quiz_answers(current_quiz, 1, ["Correct concept", "Core vocabulary"])

    assert state.last_quiz_result is not None
    assert state.last_quiz_result.score == 2
    assert state.mastery_status == "completed"
    assert state.current_quiz is None
    assert state.study_plan == generated_plan
    assert state.study_plan_citations == ["notes.pdf p.3"]

    state.study_topic = "Photosynthesis"
    state.weak_concepts = ["light reactions"]
    state.current_quiz = current_quiz
    state.quiz_round = 2
    state.remediation_message = "Temporary remediation"
    monkeypatch.setattr(
        agent_module,
        "build_study_plan",
        lambda topic, weak_concepts, reviewed_concepts=None: stopped_plan,
    )

    app.stop_mastery_session()

    assert state.current_quiz is None
    assert state.quiz_round == 0


def test_submit_response_feedback_persists_structured_record(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.active_document_id = "doc-1"
    state.uploaded_sources = ["anatomy_example.pdf"]
    state.current_mode = "ask"

    saved_feedback: list[dict[str, object]] = []

    monkeypatch.setattr(
        app,
        "save_response_feedback",
        lambda feedback: saved_feedback.append(feedback.model_dump()),
    )
    monkeypatch.setattr(app, "persist_document_library", lambda document_library, active_document_id: None)
    monkeypatch.setattr(app.st, "success", lambda *args, **kwargs: None)

    message = {
        "id": "assistant-msg-1",
        "role": "assistant",
        "content": "The medulla helps regulate breathing and heart rate.",
        "citations": ["anatomy_example.pdf, page 13"],
        "topic": "Medulla",
        "query": "what is the medulla?",
        "mode": "ask",
    }

    app._submit_response_feedback(
        message=message,
        rating="Very helpful",
        feedback_text="This was clear and grounded in the notes.",
    )

    assert "assistant-msg-1" in state.message_feedback
    assert state.message_feedback["assistant-msg-1"]["rating"] == "Very helpful"
    assert state.message_feedback["assistant-msg-1"]["query"] == "what is the medulla?"
    assert saved_feedback[0]["response"] == "The medulla helps regulate breathing and heart rate."
    assert saved_feedback[0]["filename"] == "anatomy_example.pdf"
    assert saved_feedback[0]["feedback_text"] == "This was clear and grounded in the notes."


def test_main_restores_last_indexed_document_on_reload(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    restored_workspace = {
        "document_id": "doc-1",
        "session_id": "persisted-session",
        "filename": "anatomy_example.pdf",
        "document_title": "Anatomy Example",
        "document_topic": "Human Anatomy",
        "document_summary": "A survey of the major systems of the human body.",
        "chunks": [
            Chunk(
                id="persisted-session-0",
                text="The brain is part of the central nervous system.",
                filename="anatomy_example.pdf",
                page=12,
                chunk_id=0,
                session_id="persisted-session",
                citation="anatomy_example.pdf, Chapter 1, page 12",
                source_type="pdf",
                topic="Human Anatomy",
                chapter="Chapter 1",
            )
        ],
        "size_mb": 69.6,
        "chunk_count": 1,
        "last_conversation_topic": None,
        "last_opened_at": None,
        "messages": [],
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

    monkeypatch.setattr(app, "restore_document_library", lambda: ([restored_workspace], "doc-1"))
    monkeypatch.setattr(app, "initialize_observability", lambda: False)
    monkeypatch.setattr(app, "render_hero", lambda: None)
    monkeypatch.setattr(app, "render_mode_overview", lambda: None)
    monkeypatch.setattr(app, "render_upload_panel", lambda: None)
    monkeypatch.setattr(app, "render_document_library", lambda: None)
    monkeypatch.setattr(app, "render_document_workspace_header", lambda: None)
    monkeypatch.setattr(app, "_render_active_mode", lambda: None)
    monkeypatch.setattr(app, "render_empty_state", lambda: None)
    monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: None, raising=False)

    app.main()

    assert state.active_document_id == "doc-1"
    assert state.session_id == "persisted-session"
    assert state.uploaded_sources == ["anatomy_example.pdf"]
    assert len(state.document_library) == 1
    assert state.library_status_message == "Restored 1 saved workspace from local cache."


def test_activate_document_workspace_swaps_to_selected_chat(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    first_workspace = {
        "document_id": "doc-1",
        "session_id": "session-1",
        "filename": "biology.pdf",
        "document_title": "Biology",
        "document_topic": "Biology",
        "document_summary": "Biology notes.",
        "chunks": [
            Chunk(
                id="session-1-0",
                text="Biology chunk",
                filename="biology.pdf",
                page=1,
                chunk_id=0,
                session_id="session-1",
                citation="biology.pdf, Chapter 1, page 1",
                source_type="pdf",
                topic="Biology",
                chapter="Chapter 1",
            )
        ],
        "size_mb": 10.0,
        "chunk_count": 12,
        "last_conversation_topic": "Brain",
        "last_opened_at": None,
        "messages": [{"role": "assistant", "content": "Biology chat"}],
        "current_mode": "ask",
        "conversation_topic": "Brain",
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
    second_workspace = {
        "document_id": "doc-2",
        "session_id": "session-2",
        "filename": "chemistry.pdf",
        "document_title": "Chemistry",
        "document_topic": "Chemistry",
        "document_summary": "Chemistry notes.",
        "chunks": [
            Chunk(
                id="session-2-0",
                text="Chemistry chunk",
                filename="chemistry.pdf",
                page=4,
                chunk_id=0,
                session_id="session-2",
                citation="chemistry.pdf, Chapter 4, page 4",
                source_type="pdf",
                topic="Chemistry",
                chapter="Chapter 4",
            )
        ],
        "size_mb": 11.0,
        "chunk_count": 20,
        "last_conversation_topic": "Bonding",
        "last_opened_at": None,
        "messages": [{"role": "assistant", "content": "Chemistry chat"}],
        "current_mode": "mastery",
        "conversation_topic": "Bonding",
        "study_topic": "Bonding",
        "mastery_intro": "Let's study bonding.",
        "mastery_intro_citations": ["chemistry.pdf, page 4"],
        "remediation_message": None,
        "remediation_citations": [],
        "mastery_status": "in_progress",
        "current_quiz": None,
        "quiz_round": 1,
        "last_quiz_result": None,
        "weak_concepts": [],
        "study_plan": None,
        "study_plan_citations": [],
    }

    app_state_module.set_document_library([first_workspace, second_workspace], "doc-1")
    app_state_module.activate_document_workspace("doc-2")

    assert state.active_document_id == "doc-2"
    assert state.session_id == "session-2"
    assert state.uploaded_sources == ["chemistry.pdf"]
    assert state.conversation_topic == "Bonding"
    assert state.messages == [{"role": "assistant", "content": "Chemistry chat"}]
    assert len(state.chunks) == 1


def test_render_upload_panel_indexes_document_and_triggers_rerun(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    indexed_document = app_state_module.IndexedDocument(
        document_id="doc-1",
        session_id="session-1",
        filename="anatomy_example.pdf",
        document_title="Anatomy Example",
        document_topic="Human Anatomy",
        document_summary="A survey of human anatomy.",
        chunks=[],
        size_mb=69.6,
    )

    rerun_called = {"value": False}

    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: object(), raising=False)
    monkeypatch.setattr(app, "index_uploaded_file", lambda uploaded_file: indexed_document)
    monkeypatch.setattr(app, "persist_document_library", lambda document_library, active_document_id: None)
    monkeypatch.setattr(app.st, "rerun", lambda: rerun_called.__setitem__("value", True), raising=False)

    app.render_upload_panel()

    assert state.document_library[0]["filename"] == "anatomy_example.pdf"
    assert state.active_document_id == "doc-1"
    assert state.library_status_message == "Saved anatomy_example.pdf to your study library."
    assert rerun_called["value"] is True


def test_main_recovers_workspace_from_active_session_when_library_is_empty(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    state.uploaded_sources = ["anatomy_example.pdf"]
    state.chunks = [
        Chunk(
            id="session-1-0",
            text="Recovered chunk",
            filename="anatomy_example.pdf",
            page=12,
            chunk_id=0,
            session_id=state.session_id,
            citation="anatomy_example.pdf, Chapter 1, page 12",
            source_type="pdf",
            topic="Human Anatomy",
            chapter="Chapter 1",
        )
    ]
    state.conversation_topic = "Brain"

    monkeypatch.setattr(app, "restore_document_library", lambda: ([], None))
    monkeypatch.setattr(app, "persist_document_library", lambda document_library, active_document_id: None)
    monkeypatch.setattr(app, "initialize_observability", lambda: False)
    monkeypatch.setattr(app, "render_hero", lambda: None)
    monkeypatch.setattr(app, "render_mode_overview", lambda: None)
    monkeypatch.setattr(app, "render_upload_panel", lambda: None)
    monkeypatch.setattr(app, "render_document_library", lambda: None)
    monkeypatch.setattr(app, "render_document_workspace_header", lambda: None)
    monkeypatch.setattr(app, "_render_active_mode", lambda: None)
    monkeypatch.setattr(app, "render_empty_state", lambda: None)
    monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: None, raising=False)

    app.main()

    assert len(state.document_library) == 1
    assert state.document_library[0]["filename"] == "anatomy_example.pdf"
    assert state.active_document_id == f"session-{state.session_id}"
    assert state.library_status_message == "Recovered anatomy_example.pdf from the active session."


def test_main_recovers_workspace_from_pinecone_when_local_sources_are_empty(monkeypatch) -> None:
    state = _initialize_fake_session_state(monkeypatch)
    remote_workspace = {
        "document_id": "doc-remote",
        "session_id": "session-remote",
        "filename": "anatomy_example.pdf",
        "document_title": "Anatomy Example",
        "document_topic": "Human Anatomy",
        "document_summary": "Recovered from Pinecone.",
        "chunks": [],
        "size_mb": 0.0,
        "chunk_count": 2,
        "last_conversation_topic": None,
        "last_opened_at": None,
        "messages": [],
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

    monkeypatch.setattr(app, "restore_document_library", lambda: ([], None))
    monkeypatch.setattr(app, "rebuild_document_library_from_remote", lambda: [remote_workspace])
    monkeypatch.setattr(app, "persist_document_library", lambda document_library, active_document_id: None)
    monkeypatch.setattr(app, "initialize_observability", lambda: False)
    monkeypatch.setattr(app, "render_hero", lambda: None)
    monkeypatch.setattr(app, "render_mode_overview", lambda: None)
    monkeypatch.setattr(app, "render_upload_panel", lambda: None)
    monkeypatch.setattr(app, "render_document_library", lambda: None)
    monkeypatch.setattr(app, "render_document_workspace_header", lambda: None)
    monkeypatch.setattr(app, "_render_active_mode", lambda: None)
    monkeypatch.setattr(app, "render_empty_state", lambda: None)
    monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: None, raising=False)

    app.main()

    assert len(state.document_library) == 1
    assert state.active_document_id == "doc-remote"
    assert state.library_status_message == "Recovered 1 workspace from Pinecone."
