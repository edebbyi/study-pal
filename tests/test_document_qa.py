"""test_document_qa.py: Tests for API document Q&A retrieval behavior."""

from __future__ import annotations

from src.api.services import document_qa
from src.core.models import Chunk, RetrievedChunk, TeachingResponse


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Core summary chunk text.",
        filename="book.txt",
        page=1,
        citation="book.txt p.1",
        score=0.9,
        chunk_id=1,
    )


def test_ask_document_question_uses_broader_retrieval_for_theme_questions(monkeypatch) -> None:
    """Broad synthesis questions should request higher top_k and enriched retrieval query."""
    captured: dict[str, object] = {}
    captured_theme_synthesis: dict[str, bool] = {"value": False}

    monkeypatch.setattr(document_qa, "load_workspace", lambda doc_id, user_id: {"document_id": doc_id})

    def _fake_retrieve_workspace_context(**kwargs):
        captured.update(kwargs)
        return [_retrieved_chunk()]

    monkeypatch.setattr(document_qa, "retrieve_workspace_context", _fake_retrieve_workspace_context)
    monkeypatch.setattr(document_qa, "get_openrouter_api_key_for_user", lambda user_id: "sk-or-v1-user")
    monkeypatch.setattr(
        document_qa,
        "answer_from_context",
        lambda question, retrieved_chunks, api_key_override=None, theme_synthesis=False: (
            captured_theme_synthesis.__setitem__("value", bool(theme_synthesis))
            or TeachingResponse(
                answer="Broad answer",
                citations=["book.txt p.1"],
                used_fallback=False,
            )
        ),
    )

    response = document_qa.ask_document_question(
        doc_id="doc-1",
        question="What are the core themes of the entire book?",
        user_id="user-1",
    )

    assert response.answer == "Broad answer"
    assert captured["top_k"] == max(document_qa.settings.top_k, 12)
    assert "table of contents" in str(captured["question"]).lower()
    assert captured_theme_synthesis["value"] is True


def test_ask_document_question_keeps_standard_retrieval_for_specific_question(monkeypatch) -> None:
    """Specific/local questions should keep default retrieval query behavior."""
    captured: dict[str, object] = {}
    captured_theme_synthesis: dict[str, bool] = {"value": True}

    monkeypatch.setattr(document_qa, "load_workspace", lambda doc_id, user_id: {"document_id": doc_id})

    def _fake_retrieve_workspace_context(**kwargs):
        captured.update(kwargs)
        return [_retrieved_chunk()]

    monkeypatch.setattr(document_qa, "retrieve_workspace_context", _fake_retrieve_workspace_context)
    monkeypatch.setattr(document_qa, "get_openrouter_api_key_for_user", lambda user_id: "sk-or-v1-user")
    monkeypatch.setattr(
        document_qa,
        "answer_from_context",
        lambda question, retrieved_chunks, api_key_override=None, theme_synthesis=False: (
            captured_theme_synthesis.__setitem__("value", bool(theme_synthesis))
            or TeachingResponse(
                answer="Specific answer",
                citations=["book.txt p.1"],
                used_fallback=False,
            )
        ),
    )

    response = document_qa.ask_document_question(
        doc_id="doc-1",
        question="What does semantic memory mean?",
        user_id="user-1",
    )

    assert response.answer == "Specific answer"
    assert captured["top_k"] == document_qa.settings.top_k
    assert str(captured["question"]) == "What does semantic memory mean?"
    assert captured_theme_synthesis["value"] is False


def test_ask_document_question_augments_named_list_questions_with_phrase_hits(monkeypatch) -> None:
    """Named-list prompts should include direct phrase hits from workspace chunks."""
    captured_texts: list[str] = []
    workspace = {
        "document_id": "doc-1",
        "chunks": [
            Chunk(
                id="chunk-1",
                text="Random section about neurons and synapses.",
                filename="book.txt",
                page=3,
                chunk_id=1,
                session_id="session-1",
                citation="book.txt p.3",
                source_type="txt",
            ).model_dump(),
            Chunk(
                id="chunk-2",
                text="Core Concepts: 1. The brain is the body's most complex organ. 2. Experience shapes circuits.",
                filename="book.txt",
                page=4,
                chunk_id=2,
                session_id="session-1",
                citation="book.txt p.4",
                source_type="txt",
            ).model_dump(),
            Chunk(
                id="chunk-3",
                text="Core Concepts continue with principles on plasticity and behavior.",
                filename="book.txt",
                page=5,
                chunk_id=3,
                session_id="session-1",
                citation="book.txt p.5",
                source_type="txt",
            ).model_dump(),
        ],
    }

    monkeypatch.setattr(document_qa, "load_workspace", lambda doc_id, user_id: workspace)
    monkeypatch.setattr(
        document_qa,
        "retrieve_workspace_context",
        lambda **kwargs: [
            RetrievedChunk(
                text="Random section about neurons and synapses.",
                filename="book.txt",
                page=3,
                citation="book.txt p.3",
                score=0.91,
                chunk_id=1,
            )
        ],
    )
    monkeypatch.setattr(document_qa, "get_openrouter_api_key_for_user", lambda user_id: "sk-or-v1-user")

    def _fake_answer_from_context(question, retrieved_chunks, api_key_override=None, theme_synthesis=False):
        captured_texts.extend([chunk.text for chunk in retrieved_chunks])
        return TeachingResponse(
            answer="Synthesized core concepts.",
            citations=["book.txt p.4"],
            used_fallback=False,
        )

    monkeypatch.setattr(document_qa, "answer_from_context", _fake_answer_from_context)

    response = document_qa.ask_document_question(
        doc_id="doc-1",
        question="What are the eight core concepts?",
        user_id="user-1",
    )

    assert response.answer == "Synthesized core concepts."
    assert any("core concepts" in text.lower() for text in captured_texts)
