from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langfuse import get_client

from src.config import settings
from src.models import ResponseFeedback


cache_directory = Path(".studypal_cache")
feedback_path = cache_directory / "response_feedback.jsonl"
feedback_db_path = cache_directory / "feedback.sqlite3"

feedback_table_ddl = """
CREATE TABLE IF NOT EXISTS response_feedback (
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


def _feedback_row(feedback: ResponseFeedback) -> tuple[object, ...]:
    return (
        feedback.message_id,
        feedback.session_id,
        feedback.document_id,
        feedback.filename,
        feedback.query,
        feedback.response,
        feedback.rating,
        feedback.feedback_text,
        feedback.topic,
        feedback.mode,
        json.dumps(feedback.citations),
        json.dumps(
            {
                "document_id": feedback.document_id,
                "filename": feedback.filename,
                "topic": feedback.topic,
                "mode": feedback.mode,
            }
        ),
        feedback.created_at,
    )


def _ensure_feedback_table(connection: sqlite3.Connection) -> None:
    connection.execute(feedback_table_ddl)
    connection.commit()


def _persist_feedback_sqlite(feedback: ResponseFeedback) -> None:
    cache_directory.mkdir(exist_ok=True)
    connection = sqlite3.connect(feedback_db_path)
    try:
        _ensure_feedback_table(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO response_feedback (
                message_id,
                session_id,
                document_id,
                filename,
                query,
                response,
                rating,
                feedback_text,
                topic,
                mode,
                citations_json,
                metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _feedback_row(feedback),
        )
        connection.commit()
    finally:
        connection.close()


def _persist_feedback_postgres(feedback: ResponseFeedback) -> bool:
    if not settings.database_url:
        return False

    try:
        import psycopg
    except ImportError:
        return False

    try:
        with psycopg.connect(settings.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(feedback_table_ddl)
                cursor.execute(
                    """
                    INSERT INTO response_feedback (
                        message_id,
                        session_id,
                        document_id,
                        filename,
                        query,
                        response,
                        rating,
                        feedback_text,
                        topic,
                        mode,
                        citations_json,
                        metadata_json,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        document_id = EXCLUDED.document_id,
                        filename = EXCLUDED.filename,
                        query = EXCLUDED.query,
                        response = EXCLUDED.response,
                        rating = EXCLUDED.rating,
                        feedback_text = EXCLUDED.feedback_text,
                        topic = EXCLUDED.topic,
                        mode = EXCLUDED.mode,
                        citations_json = EXCLUDED.citations_json,
                        metadata_json = EXCLUDED.metadata_json,
                        created_at = EXCLUDED.created_at
                    """,
                    _feedback_row(feedback),
                )
            connection.commit()
        return True
    except Exception:
        return False


def _row_to_feedback(row: tuple[Any, ...]) -> ResponseFeedback:
    (
        message_id,
        session_id,
        document_id,
        filename,
        query,
        response,
        rating,
        feedback_text,
        topic,
        mode,
        citations_json,
        _metadata_json,
        created_at,
    ) = row
    return ResponseFeedback(
        message_id=message_id,
        session_id=session_id,
        document_id=document_id,
        filename=filename,
        query=query,
        response=response,
        rating=rating,
        feedback_text=feedback_text,
        topic=topic,
        mode=mode,
        citations=json.loads(citations_json),
        created_at=created_at,
    )


def _fetch_feedback_postgres(limit: int) -> list[ResponseFeedback] | None:
    if not settings.database_url:
        return None

    try:
        import psycopg
    except ImportError:
        return None

    try:
        with psycopg.connect(settings.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(feedback_table_ddl)
                cursor.execute(
                    """
                    SELECT
                        message_id,
                        session_id,
                        document_id,
                        filename,
                        query,
                        response,
                        rating,
                        feedback_text,
                        topic,
                        mode,
                        citations_json,
                        metadata_json,
                        created_at
                    FROM response_feedback
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [_row_to_feedback(row) for row in rows]
    except Exception:
        return None


def _fetch_feedback_sqlite(limit: int) -> list[ResponseFeedback]:
    if not feedback_db_path.exists():
        return []

    connection = sqlite3.connect(feedback_db_path)
    try:
        _ensure_feedback_table(connection)
        rows = connection.execute(
            """
            SELECT
                message_id,
                session_id,
                document_id,
                filename,
                query,
                response,
                rating,
                feedback_text,
                topic,
                mode,
                citations_json,
                metadata_json,
                created_at
            FROM response_feedback
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_feedback(row) for row in rows]
    finally:
        connection.close()


def _append_feedback_jsonl(feedback: ResponseFeedback) -> None:
    cache_directory.mkdir(exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(feedback.model_dump()) + "\n")


def _persist_feedback_langfuse(feedback: ResponseFeedback) -> None:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return

    try:
        langfuse = get_client()
        trace_id = langfuse.get_current_trace_id()
        observation_id = langfuse.get_current_observation_id()
        langfuse.create_score(
            name="response_helpfulness",
            value=feedback.rating,
            session_id=feedback.session_id,
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="CATEGORICAL",
            comment=feedback.feedback_text,
            metadata={
                "message_id": feedback.message_id,
                "document_id": feedback.document_id,
                "filename": feedback.filename,
                "query": feedback.query,
                "response": feedback.response,
                "topic": feedback.topic,
                "mode": feedback.mode,
                "citations": feedback.citations,
            },
        )
    except Exception:
        return


def save_response_feedback(feedback: ResponseFeedback) -> None:
    if not _persist_feedback_postgres(feedback):
        _persist_feedback_sqlite(feedback)
        _append_feedback_jsonl(feedback)
    _persist_feedback_langfuse(feedback)


def load_recent_feedback(limit: int = 50) -> list[ResponseFeedback]:
    postgres_feedback = _fetch_feedback_postgres(limit)
    if postgres_feedback is not None:
        return postgres_feedback
    return _fetch_feedback_sqlite(limit)
