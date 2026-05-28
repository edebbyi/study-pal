"""publishing_runs.py: Lightweight persistence for Publishing Mode run logs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.api.models import PublishingRunRecord


cache_directory = Path(".studypal_cache")
runs_db_path = cache_directory / "publishing_runs.sqlite3"

runs_table_ddl = """
CREATE TABLE IF NOT EXISTS publishing_runs (
    run_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    output_type TEXT,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    retrieved_chunk_count INTEGER NOT NULL,
    estimated_cost REAL,
    phoenix_trace_id TEXT,
    mlflow_run_id TEXT,
    created_at TEXT NOT NULL,
    user_rating INTEGER,
    user_feedback TEXT
)
"""

runs_index_doc_created_ddl = """
CREATE INDEX IF NOT EXISTS publishing_runs_doc_created_idx
    ON publishing_runs (doc_id, created_at DESC)
"""

runs_index_endpoint_ddl = """
CREATE INDEX IF NOT EXISTS publishing_runs_endpoint_idx
    ON publishing_runs (endpoint)
"""


def _ensure_runs_table(connection: sqlite3.Connection) -> None:
    """Ensure run table/indexes exist before reads/writes."""
    connection.execute(runs_table_ddl)
    # Lightweight migration support for older local DBs.
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(publishing_runs)").fetchall()
    }
    if "phoenix_trace_id" not in columns:
        connection.execute("ALTER TABLE publishing_runs ADD COLUMN phoenix_trace_id TEXT")
    if "mlflow_run_id" not in columns:
        connection.execute("ALTER TABLE publishing_runs ADD COLUMN mlflow_run_id TEXT")
    connection.execute(runs_index_doc_created_ddl)
    connection.execute(runs_index_endpoint_ddl)
    connection.commit()


def _row_to_run(row: tuple[Any, ...]) -> PublishingRunRecord:
    """Convert a DB row to a PublishingRunRecord model."""
    (
        run_id,
        doc_id,
        endpoint,
        output_type,
        model,
        latency_ms,
        retrieved_chunk_count,
        estimated_cost,
        phoenix_trace_id,
        mlflow_run_id,
        created_at,
        user_rating,
        user_feedback,
    ) = row
    return PublishingRunRecord(
        run_id=str(run_id),
        doc_id=str(doc_id),
        endpoint=str(endpoint),
        output_type=str(output_type) if output_type else None,
        model=str(model),
        latency_ms=int(latency_ms),
        retrieved_chunk_count=int(retrieved_chunk_count),
        estimated_cost=float(estimated_cost) if estimated_cost is not None else None,
        phoenix_trace_id=str(phoenix_trace_id) if phoenix_trace_id else None,
        mlflow_run_id=str(mlflow_run_id) if mlflow_run_id else None,
        created_at=str(created_at),
        user_rating=int(user_rating) if user_rating is not None else None,
        user_feedback=str(user_feedback) if user_feedback else None,
    )


def save_publishing_run(record: PublishingRunRecord) -> PublishingRunRecord:
    """Upsert one publishing run record in local SQLite storage."""
    cache_directory.mkdir(exist_ok=True)
    connection = sqlite3.connect(runs_db_path)
    try:
        _ensure_runs_table(connection)
        connection.execute(
            """
            INSERT INTO publishing_runs (
                run_id,
                doc_id,
                endpoint,
                output_type,
                model,
                latency_ms,
                retrieved_chunk_count,
                estimated_cost,
                phoenix_trace_id,
                mlflow_run_id,
                created_at,
                user_rating,
                user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                endpoint = EXCLUDED.endpoint,
                output_type = EXCLUDED.output_type,
                model = EXCLUDED.model,
                latency_ms = EXCLUDED.latency_ms,
                retrieved_chunk_count = EXCLUDED.retrieved_chunk_count,
                estimated_cost = EXCLUDED.estimated_cost,
                phoenix_trace_id = COALESCE(EXCLUDED.phoenix_trace_id, publishing_runs.phoenix_trace_id),
                mlflow_run_id = COALESCE(EXCLUDED.mlflow_run_id, publishing_runs.mlflow_run_id),
                created_at = EXCLUDED.created_at,
                user_rating = COALESCE(EXCLUDED.user_rating, publishing_runs.user_rating),
                user_feedback = COALESCE(EXCLUDED.user_feedback, publishing_runs.user_feedback)
            """,
            (
                record.run_id,
                record.doc_id,
                record.endpoint,
                record.output_type,
                record.model,
                record.latency_ms,
                record.retrieved_chunk_count,
                record.estimated_cost,
                record.phoenix_trace_id,
                record.mlflow_run_id,
                record.created_at,
                record.user_rating,
                record.user_feedback,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return record


def rate_publishing_run(
    *,
    run_id: str,
    user_rating: int | None,
    user_feedback: str | None,
) -> PublishingRunRecord | None:
    """Persist rating/feedback for one run and return updated record."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        return None

    cache_directory.mkdir(exist_ok=True)
    connection = sqlite3.connect(runs_db_path)
    try:
        _ensure_runs_table(connection)
        cursor = connection.execute(
            "SELECT run_id FROM publishing_runs WHERE run_id = ?",
            (normalized_run_id,),
        )
        if cursor.fetchone() is None:
            return None
        connection.execute(
            """
            UPDATE publishing_runs
            SET user_rating = ?, user_feedback = ?
            WHERE run_id = ?
            """,
            (user_rating, user_feedback, normalized_run_id),
        )
        connection.commit()
        refreshed = connection.execute(
            """
            SELECT
                run_id,
                doc_id,
                endpoint,
                output_type,
                model,
                latency_ms,
                retrieved_chunk_count,
                estimated_cost,
                phoenix_trace_id,
                mlflow_run_id,
                created_at,
                user_rating,
                user_feedback
            FROM publishing_runs
            WHERE run_id = ?
            """,
            (normalized_run_id,),
        ).fetchone()
        if refreshed is None:
            return None
        return _row_to_run(tuple(refreshed))
    finally:
        connection.close()


def load_document_runs(*, doc_id: str, limit: int = 25) -> list[PublishingRunRecord]:
    """Load recent publishing runs for one document."""
    normalized_doc_id = doc_id.strip()
    if not normalized_doc_id:
        return []
    if not runs_db_path.exists():
        return []

    connection = sqlite3.connect(runs_db_path)
    try:
        _ensure_runs_table(connection)
        rows = connection.execute(
            """
            SELECT
                run_id,
                doc_id,
                endpoint,
                output_type,
                model,
                latency_ms,
                retrieved_chunk_count,
                estimated_cost,
                phoenix_trace_id,
                mlflow_run_id,
                created_at,
                user_rating,
                user_feedback
            FROM publishing_runs
            WHERE doc_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (normalized_doc_id, max(1, int(limit))),
        ).fetchall()
        return [_row_to_run(tuple(row)) for row in rows]
    finally:
        connection.close()
