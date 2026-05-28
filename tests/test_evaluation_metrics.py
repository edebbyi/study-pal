"""test_evaluation_metrics.py: Coverage tests for live-run metric formulas."""

from __future__ import annotations

from src.api.services.evaluation_metrics import build_live_run_metrics


def test_live_metrics_no_chunks() -> None:
    payload = build_live_run_metrics(retrieved_chunks=[], mode="ask_the_book")
    metrics = payload["run_details_metrics"]
    checks = payload["quality_checks"]

    assert metrics["retrieved_chunk_count"] == 0
    assert metrics["avg_relevance_score"] == 0.0
    assert metrics["top_relevance_score"] == 0.0
    assert metrics["context_coverage_label"] in {"weak", "limited"}
    assert checks["grounded_in_source"] == "unknown"


def test_live_metrics_one_chunk() -> None:
    chunks = [{"score": 0.9, "page": 3}]
    payload = build_live_run_metrics(
        retrieved_chunks=chunks,
        mode="marketing_copy",
        unsupported_claims_detected=False,
    )
    metrics = payload["run_details_metrics"]
    checks = payload["quality_checks"]

    assert metrics["retrieved_chunk_count"] == 1
    assert metrics["avg_relevance_score"] == 0.9
    assert metrics["top_relevance_score"] == 0.9
    assert checks["grounded_in_source"] == "yes"
    assert checks["human_review_recommended"] is True


def test_live_metrics_multiple_chunks_diversity_and_labels() -> None:
    chunks = [
        {"score": 0.9, "page": 1},
        {"score": 0.8, "page": 2},
        {"score": 0.7, "page": 3},
        {"score": 0.6, "page": 3},
    ]
    payload = build_live_run_metrics(
        retrieved_chunks=chunks,
        mode="ask_the_book",
        missing_context_flags=[],
        unsupported_claims_detected=False,
    )
    metrics = payload["run_details_metrics"]
    trace = payload["workflow_trace"]

    assert metrics["retrieved_chunk_count"] == 4
    assert round(float(metrics["avg_relevance_score"]), 3) == 0.75
    assert metrics["source_diversity_score"] is not None
    assert metrics["context_coverage_label"] in {"strong", "moderate", "limited", "weak"}
    assert trace["context_coverage_label"] == metrics["context_coverage_label"]
