"""test_publishing_marketing_copy.py: Tests for Publishing Mode marketing copy service."""

from __future__ import annotations

import pytest

from src.api.services import publishing_service
from src.core.models import Chunk, RetrievedChunk
from src.publishing.schemas import MarketingCopyRequest


def _workspace() -> dict[str, object]:
    return {
        "document_id": "doc-123",
        "session_id": "session-abc",
        "user_id": "user-1",
        "document_title": "The Orchard at Dusk",
        "chunks": [
            Chunk(
                id="chunk-row-1",
                text="A grieving botanist returns to her hometown and finds coded journals in an abandoned greenhouse.",
                filename="manuscript_excerpt.txt",
                page=1,
                chunk_id=1,
                session_id="session-abc",
                citation="manuscript_excerpt.txt p.1",
                source_type="txt",
                document_id="doc-123",
                document_title="The Orchard at Dusk",
                user_id="user-1",
            ).model_dump()
        ],
    }


def _retrieved_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="A grieving botanist returns to her hometown and finds coded journals in an abandoned greenhouse.",
            filename="manuscript_excerpt.txt",
            page=1,
            citation="manuscript_excerpt.txt p.1",
            score=0.92,
            chunk_id=1,
            topic="memory",
            chapter="opening",
        )
    ]


@pytest.mark.parametrize(
    "output_type",
    ["back_cover", "newsletter", "bookstore_pitch"],
)
def test_generate_marketing_copy_returns_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    output_type: str,
) -> None:
    """It should return structured copy, sources, and metadata for key output types."""
    saved_runs: list[object] = []
    monkeypatch.setattr(publishing_service, "load_workspace", lambda doc_id, user_id: _workspace())
    monkeypatch.setattr(
        publishing_service,
        "retrieve_workspace_context",
        lambda **kwargs: _retrieved_chunks(),
    )
    monkeypatch.setattr(
        publishing_service,
        "generate_marketing_copy_from_context",
        lambda **kwargs: {
            "output_type": kwargs["output_type"],
            "copy": f"{kwargs['output_type']} draft grounded in sources.",
            "rationale": "Emphasized themes and hook language supported by retrieved chunks.",
        },
    )
    monkeypatch.setattr(
        publishing_service,
        "save_publishing_run",
        lambda record: saved_runs.append(record),
    )

    response = publishing_service.generate_marketing_copy_for_document(
        doc_id="doc-123",
        request=MarketingCopyRequest(
            output_type=output_type,
            tone="cinematic",
            audience="adult fiction readers",
            spoiler_level="low",
        ),
        user_id="user-1",
    )

    assert response.output_type == output_type
    assert output_type in response.copy_text
    assert response.rationale is not None
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "1"
    assert response.metadata.retrieved_chunk_count == 1
    assert response.metadata.latency_ms >= 0
    assert response.metadata.run_id
    assert len(saved_runs) == 1


@pytest.mark.parametrize(
    ("output_type", "prefix"),
    [
        ("back_cover", "Back-cover draft"),
        ("newsletter", "Newsletter blurb"),
        ("bookstore_pitch", "Bookseller pitch"),
    ],
)
def test_generate_marketing_copy_falls_back_to_grounded_excerpt(
    monkeypatch: pytest.MonkeyPatch,
    output_type: str,
    prefix: str,
) -> None:
    """If model output is unavailable, it should still return grounded fallback copy."""
    saved_runs: list[object] = []
    monkeypatch.setattr(publishing_service, "load_workspace", lambda doc_id, user_id: _workspace())
    monkeypatch.setattr(
        publishing_service,
        "retrieve_workspace_context",
        lambda **kwargs: _retrieved_chunks(),
    )
    monkeypatch.setattr(
        publishing_service,
        "generate_marketing_copy_from_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        publishing_service,
        "save_publishing_run",
        lambda record: saved_runs.append(record),
    )

    response = publishing_service.generate_marketing_copy_for_document(
        doc_id="doc-123",
        request=MarketingCopyRequest(output_type=output_type),
        user_id="user-1",
    )

    assert response.output_type == output_type
    assert response.copy_text.startswith(prefix)
    assert response.rationale is not None
    assert len(response.sources) == 1
    assert response.metadata.model == "retrieval_fallback"
    assert response.metadata.retrieved_chunk_count == 1
    assert response.metadata.run_id
    assert len(saved_runs) == 1
