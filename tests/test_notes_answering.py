"""test_notes_answering.py: Notes answering flow safety tests."""

from __future__ import annotations

import src.notes.notes_answering as notes_answering
from src.core.models import RetrievedChunk, StructuredAnswer


class SessionState(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


def test_build_structured_answer_retries_with_unscoped_local_retrieval(monkeypatch) -> None:
    """Retry local retrieval without strict scopes when first retrieval returns nothing."""

    state = SessionState(
        chunks=[object()],
        session_id="session-1",
        active_document_id="doc-1",
        user_id="user-1",
        messages=[],
    )
    monkeypatch.setattr(notes_answering.st, "session_state", state, raising=False)

    calls: list[dict[str, object]] = []
    recovered_chunk = RetrievedChunk(
        text="Brain states include wakefulness and sleep stages.",
        filename="anatomy_example.pdf",
        page=59,
        citation="anatomy_example.pdf, page 59",
        score=0.91,
        chunk_id=10,
    )

    def fake_retrieve_chunks(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return []
        return [recovered_chunk]

    monkeypatch.setattr(notes_answering, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(
        notes_answering,
        "generate_structured_answer",
        lambda **kwargs: StructuredAnswer(answer="Brain states are described in your notes."),
    )

    response = notes_answering.build_structured_answer_response("what are brain states?")

    assert response.answer == "Brain states are described in your notes."
    assert calls[0]["session_id"] == "session-1"
    assert calls[0]["document_id"] == "doc-1"
    assert calls[0]["user_id"] == "user-1"
    assert calls[1]["session_id"] == ""
    assert calls[1]["document_id"] is None
    assert calls[1]["user_id"] is None
