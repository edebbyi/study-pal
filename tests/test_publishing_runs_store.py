"""test_publishing_runs_store.py: Tests for Publishing Mode run persistence helpers."""

from __future__ import annotations

from src.api.models import PublishingRunRecord
from src.api.services import publishing_runs


def test_save_and_load_document_runs(tmp_path, monkeypatch) -> None:
    """It stores runs and can fetch them by document id."""
    monkeypatch.setattr(publishing_runs, "cache_directory", tmp_path)
    monkeypatch.setattr(publishing_runs, "runs_db_path", tmp_path / "publishing_runs.sqlite3")

    first = PublishingRunRecord(
        run_id="run-1",
        doc_id="doc-123",
        endpoint="book_brief",
        output_type="book_brief",
        model="openai/gpt-4.1-mini",
        latency_ms=123,
        retrieved_chunk_count=6,
        estimated_cost=None,
        created_at="2026-05-23T19:00:00+00:00",
        user_rating=None,
        user_feedback=None,
    )
    second = PublishingRunRecord(
        run_id="run-2",
        doc_id="doc-123",
        endpoint="marketing_copy",
        output_type="newsletter",
        model="openai/gpt-4.1-mini",
        latency_ms=98,
        retrieved_chunk_count=5,
        estimated_cost=0.0042,
        created_at="2026-05-23T19:01:00+00:00",
        user_rating=None,
        user_feedback=None,
    )
    other_doc = PublishingRunRecord(
        run_id="run-3",
        doc_id="doc-999",
        endpoint="book_brief",
        output_type="book_brief",
        model="retrieval_fallback",
        latency_ms=22,
        retrieved_chunk_count=4,
        estimated_cost=None,
        created_at="2026-05-23T19:02:00+00:00",
        user_rating=None,
        user_feedback=None,
    )

    publishing_runs.save_publishing_run(first)
    publishing_runs.save_publishing_run(second)
    publishing_runs.save_publishing_run(other_doc)

    records = publishing_runs.load_document_runs(doc_id="doc-123", limit=10)
    assert [record.run_id for record in records] == ["run-2", "run-1"]
    assert records[0].output_type == "newsletter"


def test_rate_publishing_run_updates_feedback(tmp_path, monkeypatch) -> None:
    """It updates stored user rating/feedback for a run."""
    monkeypatch.setattr(publishing_runs, "cache_directory", tmp_path)
    monkeypatch.setattr(publishing_runs, "runs_db_path", tmp_path / "publishing_runs.sqlite3")

    record = PublishingRunRecord(
        run_id="run-rate-1",
        doc_id="doc-abc",
        endpoint="marketing_copy",
        output_type="back_cover",
        model="openai/gpt-4.1-mini",
        latency_ms=140,
        retrieved_chunk_count=7,
        estimated_cost=None,
        created_at="2026-05-23T20:00:00+00:00",
        user_rating=None,
        user_feedback=None,
    )
    publishing_runs.save_publishing_run(record)

    updated = publishing_runs.rate_publishing_run(
        run_id="run-rate-1",
        user_rating=5,
        user_feedback="Grounded and useful for campaign planning.",
    )

    assert updated is not None
    assert updated.user_rating == 5
    assert updated.user_feedback == "Grounded and useful for campaign planning."


def test_rate_publishing_run_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    """It returns None for unknown run ids."""
    monkeypatch.setattr(publishing_runs, "cache_directory", tmp_path)
    monkeypatch.setattr(publishing_runs, "runs_db_path", tmp_path / "publishing_runs.sqlite3")

    updated = publishing_runs.rate_publishing_run(
        run_id="does-not-exist",
        user_rating=3,
        user_feedback="ok",
    )
    assert updated is None
