"""test_chunk_coercion.py: Regression tests for chunk coercion helpers."""

from __future__ import annotations

import app
from src.core.app_state import _as_chunk_list
from src.core.models import Chunk


class ChunkLike:
    """Minimal chunk-like object that exposes model_dump()."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, object]:
        return dict(self._payload)


def _payload() -> dict[str, object]:
    return {
        "id": "s-1-0",
        "text": "Brain states include wakefulness and sleep stages.",
        "filename": "anatomy_example.pdf",
        "page": 59,
        "chunk_id": 0,
        "session_id": "s-1",
        "citation": "anatomy_example.pdf, page 59",
        "source_type": "pdf",
        "document_id": "doc-1",
        "user_id": "user-1",
    }


def test_coerce_chunks_accepts_chunk_like_object() -> None:
    """App coercion should accept chunk-like objects from hot-reload boundaries."""
    chunk_like = ChunkLike(_payload())
    chunks = app._coerce_chunks([chunk_like])
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].document_id == "doc-1"


def test_as_chunk_list_accepts_chunk_like_object() -> None:
    """Shared app-state chunk parser should accept chunk-like objects too."""
    chunk_like = ChunkLike(_payload())
    chunks = _as_chunk_list([chunk_like])
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].user_id == "user-1"
