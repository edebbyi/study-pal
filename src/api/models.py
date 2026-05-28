"""models.py: Shared API response models for the FastAPI backend."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.core.version import APP_VERSION


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str = "ok"
    service: str = "studypal-api"
    version: str = APP_VERSION


class ObservabilityHealthResponse(BaseModel):
    """Runtime status of observability integrations."""

    phoenix_configured: bool
    phoenix_initialized: bool
    otel_span_mode: str | None = None
    mlflow_available: bool
    mlflow_tracking_uri: str | None = None


class PlaceholderResponse(BaseModel):
    """Standard payload for placeholder endpoints."""

    status: str = "not_implemented"
    endpoint: str
    message: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentAskRequest(BaseModel):
    """Request payload for document-grounded Q&A."""

    question: str = Field(min_length=1, max_length=3000)
    user_id: str | None = Field(default=None, max_length=256)


class DocumentAskSource(BaseModel):
    """Source chunk returned with a grounded answer."""

    chunk_id: str
    text: str
    score: float


class DocumentAskMetadata(BaseModel):
    """Runtime metadata for a document Q&A request."""

    model: str
    latency_ms: int
    latency_seconds: float | None = None
    retrieved_chunk_count: int
    avg_relevance_score: float | None = None
    top_relevance_score: float | None = None
    source_diversity_score: float | None = None
    context_coverage_score: float | None = None
    phoenix_trace_id: str | None = None
    mlflow_run_id: str | None = None
    warning: str | None = None


class DocumentAskResponse(BaseModel):
    """Response payload for document-grounded Q&A."""

    answer: str
    sources: list[DocumentAskSource]
    metadata: DocumentAskMetadata
    workflow_trace: dict[str, object] | None = None
    quality_checks: dict[str, object] | None = None
    run_details: dict[str, object] | None = None


class PublishingRunRecord(BaseModel):
    """Stored run metadata for Publishing Mode generations."""

    run_id: str
    doc_id: str
    endpoint: str
    output_type: str | None = None
    model: str
    latency_ms: int
    retrieved_chunk_count: int
    estimated_cost: float | None = None
    phoenix_trace_id: str | None = None
    mlflow_run_id: str | None = None
    created_at: str
    user_rating: int | None = None
    user_feedback: str | None = None


class RunRatingRequest(BaseModel):
    """Input payload for rating a previously logged run."""

    user_rating: int | None = Field(default=None, ge=1, le=5)
    user_feedback: str | None = Field(default=None, max_length=2000)


class RunFeedbackRequest(BaseModel):
    """Optional richer feedback payload for publishing runs."""

    rating: int | None = Field(default=None, ge=1, le=5)
    thumbs: str | None = Field(default=None, max_length=16)
    useful: bool | None = None
    grounded: bool | None = None
    notes: str | None = Field(default=None, max_length=2000)
    corrected_output: str | None = Field(default=None, max_length=4000)
