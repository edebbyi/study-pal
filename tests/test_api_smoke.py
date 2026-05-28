"""test_api_smoke.py: Lightweight API endpoint smoke tests for publishing flows."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import DocumentAskMetadata, DocumentAskResponse, DocumentAskSource
from src.api.services.document_workspace import DocumentWorkspaceAccessDeniedError
from src.api.services.service_errors import ServiceTimeoutError
from src.api.routers import documents as documents_router
from src.api.routers import publishing as publishing_router
from src.publishing.schemas import (
    BookBriefResponse,
    MarketingCopyResponse,
    QualityChecks,
    ReaderBuyerPersonaSnapshot,
    RunMetadata,
    SourceChunk,
    WorkflowTrace,
)


client = TestClient(app)


def _sample_workflow_trace(selected_output: str) -> WorkflowTrace:
    return WorkflowTrace(
        retrieved_chunk_count=3,
        selected_output=selected_output,
        grounding_check="retrieval_context_attached",
        spoiler_setting="low",
        missing_context_flags_present=False,
        unsupported_claims_detected="no",
        context_coverage_label="strong",
    )


def _sample_quality_checks() -> QualityChecks:
    return QualityChecks(
        grounded_in_source="yes",
        unsupported_claims_detected="no",
        spoiler_level="low",
        missing_context_present=False,
        human_review_recommended=False,
        context_coverage_label="strong",
        context_coverage_score=0.91,
    )


def _sample_metadata(model_name: str) -> RunMetadata:
    return RunMetadata(
        model=model_name,
        latency_ms=120,
        latency_seconds=0.12,
        retrieved_chunk_count=3,
        avg_relevance_score=0.8,
        top_relevance_score=0.9,
        context_coverage_score=0.86,
        run_id="run-test-1",
    )


def test_health_endpoint_smoke() -> None:
    """Health endpoint should return a simple status payload."""
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_documents_ask_endpoint_smoke(monkeypatch) -> None:
    """/api/documents/{doc_id}/ask should return answer + sources + metadata."""

    def _fake_ask_document_question(*, doc_id: str, question: str, user_id: str | None = None) -> DocumentAskResponse:
        return DocumentAskResponse(
            answer=f"Answer for {doc_id}: {question}",
            sources=[
                DocumentAskSource(chunk_id="1", text="Grounded chunk text", score=0.95),
            ],
            metadata=DocumentAskMetadata(
                model="openai/gpt-4.1-mini",
                latency_ms=88,
                retrieved_chunk_count=1,
            ),
            workflow_trace=_sample_workflow_trace("ask_the_book").model_dump(),
            quality_checks=_sample_quality_checks().model_dump(),
            run_details={"latency_ms": 88, "retrieved_chunk_count": 1},
        )

    monkeypatch.setattr(documents_router, "ask_document_question", _fake_ask_document_question)

    response = client.post(
        "/api/documents/doc-123/ask",
        json={"question": "What is this about?"},
        headers={"x-user-id": "user-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert payload["sources"][0]["chunk_id"] == "1"
    assert payload["metadata"]["retrieved_chunk_count"] == 1


def test_documents_ask_rejects_oversized_question() -> None:
    """Oversized ask payloads should fail fast with validation error."""
    response = client.post(
        "/api/documents/doc-123/ask",
        json={"question": "x" * 3001},
    )
    assert response.status_code == 422


def test_documents_ask_rejects_conflicting_user_ids() -> None:
    """Conflicting user identity sources should be rejected."""
    response = client.post(
        "/api/documents/doc-123/ask?user_id=query-user",
        json={"question": "What is this about?", "user_id": "body-user"},
        headers={"x-user-id": "header-user"},
    )
    assert response.status_code == 400


def test_book_brief_endpoint_smoke(monkeypatch) -> None:
    """/api/publishing/{doc_id}/book-brief should return structured JSON."""

    def _fake_generate_book_brief_for_document(*, doc_id: str, request, user_id: str | None = None) -> BookBriefResponse:
        metadata = _sample_metadata("openai/gpt-4.1-mini")
        return BookBriefResponse(
            title="Sample Title",
            genre="Educational Nonfiction / Reference",
            primary_audience="Educators",
            secondary_audience="Students",
            reader_buyer_persona=ReaderBuyerPersonaSnapshot(
                persona_name="Science Educator",
                role_context="High-school biology educator",
                motivation="Needs trustworthy neuroscience references",
                needs="Clear structure and credible sourcing",
                likely_objections="Needs classroom scaffolding",
                discovery_channels="Libraries, educator newsletters",
                messaging_notes="Lead with clarity and authority",
            ),
            core_themes=["Theme A", "Theme B"],
            audience_keywords=["education", "neuroscience"],
            one_sentence_positioning="A grounded positioning sentence.",
            positioning_recommendation="Use as a classroom reference companion.",
            marketing_angles=["Credibility", "Accessibility"],
            sales_use_case="Academic and library channels.",
            risk_flags=["Review market comps manually."],
            workflow_trace=_sample_workflow_trace("positioning_brief"),
            quality_checks=_sample_quality_checks(),
            sources=[SourceChunk(chunk_id="1", text="Chunk text", score=0.9)],
            metadata=metadata,
            run_details=metadata,
        )

    monkeypatch.setattr(
        publishing_router,
        "generate_book_brief_for_document",
        _fake_generate_book_brief_for_document,
    )

    response = client.post(
        "/api/publishing/doc-123/book-brief",
        json={"audience": "educators", "spoiler_level": "low"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Sample Title"
    assert payload["primary_audience"] == "Educators"
    assert payload["metadata"]["model"] == "openai/gpt-4.1-mini"


def test_book_brief_rejects_conflicting_user_ids(monkeypatch) -> None:
    """Publishing endpoints should reject conflicting identity sources."""

    monkeypatch.setattr(
        publishing_router,
        "generate_book_brief_for_document",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    response = client.post(
        "/api/publishing/doc-123/book-brief?user_id=query-user",
        json={"audience": "educators"},
        headers={"x-user-id": "header-user"},
    )
    assert response.status_code == 400


def test_marketing_copy_endpoint_smoke(monkeypatch) -> None:
    """/api/publishing/{doc_id}/marketing-copy should return structured JSON."""

    def _fake_generate_marketing_copy_for_document(*, doc_id: str, request, user_id: str | None = None) -> MarketingCopyResponse:
        metadata = _sample_metadata("openai/gpt-4.1-mini")
        return MarketingCopyResponse(
            output_type="newsletter",
            copy="Newsletter draft text",
            rationale="Source-grounded rationale.",
            workflow_trace=_sample_workflow_trace("newsletter"),
            quality_checks=_sample_quality_checks(),
            sources=[SourceChunk(chunk_id="4", text="Evidence chunk", score=0.83)],
            metadata=metadata,
            run_details=metadata,
        )

    monkeypatch.setattr(
        publishing_router,
        "generate_marketing_copy_for_document",
        _fake_generate_marketing_copy_for_document,
    )

    response = client.post(
        "/api/publishing/doc-123/marketing-copy",
        json={"output_type": "newsletter", "spoiler_level": "low"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output_type"] == "newsletter"
    assert payload["copy"] == "Newsletter draft text"
    assert payload["metadata"]["retrieved_chunk_count"] == 3


def test_documents_ask_access_denied_maps_403(monkeypatch) -> None:
    """Cross-user workspace denial should map to HTTP 403."""
    monkeypatch.setattr(
        documents_router,
        "ask_document_question",
        lambda **kwargs: (_ for _ in ()).throw(
            DocumentWorkspaceAccessDeniedError("You do not have access to this document workspace.")
        ),
    )
    response = client.post(
        "/api/documents/doc-123/ask",
        json={"question": "What is this about?"},
        headers={"x-user-id": "user-1"},
    )
    assert response.status_code == 403


def test_publishing_book_brief_access_denied_maps_403(monkeypatch) -> None:
    """Cross-user workspace denial should map to HTTP 403 for publishing brief."""
    monkeypatch.setattr(
        publishing_router,
        "generate_book_brief_for_document",
        lambda **kwargs: (_ for _ in ()).throw(
            DocumentWorkspaceAccessDeniedError("You do not have access to this document workspace.")
        ),
    )
    response = client.post(
        "/api/publishing/doc-123/book-brief",
        json={"audience": "educators"},
        headers={"x-user-id": "user-1"},
    )
    assert response.status_code == 403


def test_documents_ask_timeout_maps_504(monkeypatch) -> None:
    """Ask timeout should map to HTTP 504 with graceful message."""
    monkeypatch.setattr(
        documents_router,
        "ask_document_question",
        lambda **kwargs: (_ for _ in ()).throw(
            ServiceTimeoutError(
                operation="answer_from_context",
                timeout_seconds=30,
                mode="ask_the_book",
                output_type="ask_the_book",
            )
        ),
    )
    response = client.post(
        "/api/documents/doc-123/ask",
        json={"question": "What is this about?"},
        headers={"x-user-id": "user-1"},
    )
    assert response.status_code == 504


def test_book_brief_timeout_maps_504(monkeypatch) -> None:
    """Book brief timeout should map to HTTP 504 with graceful message."""
    monkeypatch.setattr(
        publishing_router,
        "generate_book_brief_for_document",
        lambda **kwargs: (_ for _ in ()).throw(
            ServiceTimeoutError(
                operation="generate_book_brief_from_context",
                timeout_seconds=45,
                mode="positioning_brief",
                output_type="book_brief",
            )
        ),
    )
    response = client.post(
        "/api/publishing/doc-123/book-brief",
        json={"audience": "educators"},
        headers={"x-user-id": "user-1"},
    )
    assert response.status_code == 504


def test_marketing_copy_timeout_maps_504(monkeypatch) -> None:
    """Marketing copy timeout should map to HTTP 504 with graceful message."""
    monkeypatch.setattr(
        publishing_router,
        "generate_marketing_copy_for_document",
        lambda **kwargs: (_ for _ in ()).throw(
            ServiceTimeoutError(
                operation="generate_marketing_copy_from_context",
                timeout_seconds=40,
                mode="marketing_copy",
                output_type="newsletter",
            )
        ),
    )
    response = client.post(
        "/api/publishing/doc-123/marketing-copy",
        json={"output_type": "newsletter"},
        headers={"x-user-id": "user-1"},
    )
    assert response.status_code == 504
