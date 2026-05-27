"""test_evaluate_retrieval.py: Tests for retrieval eval metric helpers."""

from __future__ import annotations

from pathlib import Path

from scripts import evaluate_retrieval


def test_load_eval_rows_includes_optional_retrieved_ids(tmp_path: Path) -> None:
    """Loader should parse both relevant and optional retrieved chunk ids."""
    data = (
        '{"doc_id":"doc-a","question":"Q1","relevant_chunk_ids":["1","2"],"retrieved_chunk_ids":["2","3"]}\n'
        '{"doc_id":"doc-b","question":"Q2","relevant_chunk_ids":["7"]}\n'
    )
    path = tmp_path / "eval.jsonl"
    path.write_text(data, encoding="utf-8")

    rows = evaluate_retrieval._load_eval_rows(path)

    assert len(rows) == 2
    assert rows[0].retrieved_chunk_ids == ["2", "3"]
    assert rows[1].retrieved_chunk_ids is None


def test_metric_formulas_basic() -> None:
    """Metric helpers should compute expected values."""
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c", "z"}

    hit1 = 1.0 if retrieved[0] in relevant else 0.0
    precision = evaluate_retrieval._precision_at_k(retrieved, relevant, 3)
    recall = evaluate_retrieval._recall_at_k(retrieved, relevant, 3)
    mrr = evaluate_retrieval._mrr(retrieved, relevant)

    assert hit1 == 0.0
    assert precision == 1 / 3
    assert recall == 1 / 2
    assert mrr == 1 / 3
