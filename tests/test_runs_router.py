"""test_runs_router.py: Unit tests for runs feedback router handlers."""

from __future__ import annotations

from src.api.models import PublishingRunRecord, RunFeedbackRequest
from src.api.routers import runs as runs_router


def _record() -> PublishingRunRecord:
    return PublishingRunRecord(
        run_id="run-1",
        doc_id="doc-1",
        endpoint="book_brief",
        output_type="book_brief",
        model="openai/gpt-4.1-mini",
        latency_ms=120,
        retrieved_chunk_count=5,
        estimated_cost=None,
        phoenix_trace_id="trace-1",
        mlflow_run_id=None,
        created_at="2026-05-24T00:00:00+00:00",
        user_rating=None,
        user_feedback=None,
    )


def test_save_run_feedback_maps_fields(monkeypatch) -> None:
    """It maps feedback payload into persisted rating + feedback text."""
    calls: dict[str, object] = {}

    def _fake_rate_publishing_run(*, run_id: str, user_rating: int | None, user_feedback: str | None):
        calls["run_id"] = run_id
        calls["user_rating"] = user_rating
        calls["user_feedback"] = user_feedback
        record = _record()
        record.user_rating = user_rating
        record.user_feedback = user_feedback
        return record

    monkeypatch.setattr(runs_router, "rate_publishing_run", _fake_rate_publishing_run)

    response = runs_router.save_run_feedback(
        "run-1",
        RunFeedbackRequest(
            rating=4,
            thumbs="up",
            useful=True,
            grounded=True,
            notes="Helpful output.",
        ),
    )

    assert calls["run_id"] == "run-1"
    assert calls["user_rating"] == 4
    assert "thumbs=up" in str(calls["user_feedback"])
    assert "notes=Helpful output." in str(calls["user_feedback"])
    assert response.user_rating == 4
