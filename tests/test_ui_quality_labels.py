"""test_ui_quality_labels.py: Ensure frontend shows human-readable quality labels."""

from __future__ import annotations

from pathlib import Path


def test_quality_check_labels_are_human_readable() -> None:
    content = Path("app.py").read_text(encoding="utf-8")

    assert "Grounded in source" in content
    assert "Unsupported claims detected" in content
    assert "Missing context present" in content
    assert "Human review recommended" in content
    assert "Context coverage" in content
