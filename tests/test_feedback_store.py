"""test_feedback_store.py: Tests for test_feedback_store.py."""

from __future__ import annotations

import json
import sqlite3

from src.feedback.feedback_store import load_recent_feedback, save_response_feedback
from src.core.models import ResponseFeedback


def test_save_response_feedback_persists_sqlite_and_jsonl(tmp_path, monkeypatch) -> None:
    """Test save response feedback persists sqlite and jsonl.
    
    Args:
        tmp_path: Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    db_path = tmp_path / "feedback.sqlite3"
    jsonl_path = tmp_path / "response_feedback.jsonl"

    monkeypatch.setattr("src.feedback.feedback_store.feedback_db_path", db_path)
    monkeypatch.setattr("src.feedback.feedback_store.feedback_path", jsonl_path)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_postgres", lambda feedback: False)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_langfuse", lambda feedback: None)

    feedback = ResponseFeedback(
        message_id="msg-1",
        session_id="session-1",
        document_id="doc-1",
        filename="anatomy_example.pdf",
        query="what is the medulla?",
        response="The medulla helps regulate breathing and heart rate.",
        rating="Very helpful",
        feedback_text="Clear and grounded.",
        topic="Medulla",
        mode="ask",
        citations=["anatomy_example.pdf, page 13"],
        created_at="2026-03-18T12:00:00+00:00",
    )

    save_response_feedback(feedback)

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT message_id, query, rating, feedback_text FROM response_feedback WHERE message_id = ?",
            ("msg-1",),
        ).fetchone()
    finally:
        connection.close()

    assert row == ("msg-1", "what is the medulla?", "Very helpful", "Clear and grounded.")

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["message_id"] == "msg-1"
    assert payload["rating"] == "Very helpful"


def test_save_response_feedback_prefers_postgres_when_configured(monkeypatch) -> None:
    """Test save response feedback prefers postgres when configured.
    
    Args:
        monkeypatch: Test fixture or parameter.
    """

    feedback = ResponseFeedback(
        message_id="msg-2",
        session_id="session-2",
        document_id="doc-2",
        filename="biology.pdf",
        query="what is a neuron?",
        response="A neuron is a specialized nerve cell.",
        rating="Somewhat helpful",
        feedback_text=None,
        topic="Neuron",
        mode="ask",
        citations=["biology.pdf, page 4"],
        created_at="2026-03-18T12:05:00+00:00",
    )

    postgres_calls: list[str] = []
    sqlite_calls: list[str] = []
    jsonl_calls: list[str] = []
    langfuse_calls: list[str] = []

    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_postgres", lambda saved_feedback: postgres_calls.append(saved_feedback.message_id) or True)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_sqlite", lambda saved_feedback: sqlite_calls.append(saved_feedback.message_id))
    monkeypatch.setattr("src.feedback.feedback_store._append_feedback_jsonl", lambda saved_feedback: jsonl_calls.append(saved_feedback.message_id))
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_langfuse", lambda saved_feedback: langfuse_calls.append(saved_feedback.message_id))

    save_response_feedback(feedback)

    assert postgres_calls == ["msg-2"]
    assert sqlite_calls == []
    assert jsonl_calls == []
    assert langfuse_calls == ["msg-2"]


def test_load_recent_feedback_reads_sqlite_records(tmp_path, monkeypatch) -> None:
    """Test load recent feedback reads sqlite records.
    
    Args:
        tmp_path: Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    db_path = tmp_path / "feedback.sqlite3"
    jsonl_path = tmp_path / "response_feedback.jsonl"

    monkeypatch.setattr("src.feedback.feedback_store.feedback_db_path", db_path)
    monkeypatch.setattr("src.feedback.feedback_store.feedback_path", jsonl_path)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_postgres", lambda feedback: False)
    monkeypatch.setattr("src.feedback.feedback_store._fetch_feedback_postgres", lambda limit: None)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_langfuse", lambda feedback: None)

    first_feedback = ResponseFeedback(
        message_id="msg-a",
        session_id="session-a",
        document_id="doc-a",
        filename="anatomy_example.pdf",
        query="what is the medulla?",
        response="The medulla regulates vital functions.",
        rating="Very helpful",
        feedback_text=None,
        topic="Medulla",
        mode="ask",
        citations=["anatomy_example.pdf, page 13"],
        created_at="2026-03-18T12:00:00+00:00",
    )
    second_feedback = ResponseFeedback(
        message_id="msg-b",
        session_id="session-b",
        document_id="doc-b",
        filename="biology.pdf",
        query="what is a neuron?",
        response="A neuron is a nerve cell.",
        rating="Not helpful",
        feedback_text="Too short.",
        topic="Neuron",
        mode="ask",
        citations=["biology.pdf, page 4"],
        created_at="2026-03-18T12:05:00+00:00",
    )

    save_response_feedback(first_feedback)
    save_response_feedback(second_feedback)

    records = load_recent_feedback(limit=10)

    assert len(records) == 2
    assert records[0].message_id == "msg-b"
    assert records[1].message_id == "msg-a"


def test_save_response_feedback_migrates_legacy_sqlite_schema(tmp_path, monkeypatch) -> None:
    """Test save response feedback migrates legacy sqlite schema.
    
    Args:
        tmp_path: Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    db_path = tmp_path / "feedback.sqlite3"
    jsonl_path = tmp_path / "response_feedback.jsonl"

    monkeypatch.setattr("src.feedback.feedback_store.feedback_db_path", db_path)
    monkeypatch.setattr("src.feedback.feedback_store.feedback_path", jsonl_path)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_postgres", lambda feedback: False)
    monkeypatch.setattr("src.feedback.feedback_store._persist_feedback_langfuse", lambda feedback: None)

    # Create a legacy table that predates the user_id column.
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE response_feedback (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                document_id TEXT,
                filename TEXT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                rating TEXT NOT NULL,
                feedback_text TEXT,
                topic TEXT,
                mode TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    feedback = ResponseFeedback(
        message_id="msg-legacy",
        session_id="session-legacy",
        user_id="learner-1",
        document_id="doc-legacy",
        filename="legacy.pdf",
        query="what changed?",
        response="Schema migration added user_id.",
        rating="Very helpful",
        feedback_text=None,
        topic="Migration",
        mode="ask",
        citations=["legacy.pdf, page 1"],
        created_at="2026-03-18T12:00:00+00:00",
    )

    save_response_feedback(feedback)

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT user_id FROM response_feedback WHERE message_id = ?",
            ("msg-legacy",),
        ).fetchone()
    finally:
        connection.close()

    assert row == ("learner-1",)
