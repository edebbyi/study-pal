"""schemas.py: Pydantic request/response models for Publishing Mode."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    """Source chunk returned with generated publishing outputs."""

    chunk_id: str
    text: str
    score: float


class RunMetadata(BaseModel):
    """Runtime metadata for a publishing generation request."""

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
    estimated_cost_usd: float | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    # Placeholder fields; only populate when judge/labeled eval flows exist.
    claim_support_coverage: float | None = None
    groundedness_score: float | None = None
    faithfulness_score: float | None = None
    unsupported_claim_count: int | None = None
    answer_relevance: float | None = None
    answer_similarity: float | None = None
    hit_at_1: float | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    ndcg: float | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str | None = None


class WorkflowTrace(BaseModel):
    """Observable workflow steps (no hidden model reasoning)."""

    retrieved_chunk_count: int
    selected_output: str
    grounding_check: str
    spoiler_setting: str
    missing_context_flags_present: bool
    unsupported_claims_detected: Literal["yes", "no", "unknown"]
    context_coverage_label: Literal["strong", "moderate", "limited", "weak"]


class QualityChecks(BaseModel):
    """Observable quality/safety checks for publishing outputs."""

    grounded_in_source: Literal["yes", "no", "unknown"]
    unsupported_claims_detected: Literal["yes", "no", "unknown"]
    spoiler_level: str | None = None
    missing_context_present: bool
    human_review_recommended: bool
    context_coverage_label: Literal["strong", "moderate", "limited", "weak"] | None = None
    context_coverage_score: float | None = None


class BookBriefRequest(BaseModel):
    """Input payload for Book Brief generation."""

    audience: str | None = Field(default=None, max_length=256)
    spoiler_level: str | None = Field(default=None, max_length=32)
    notes: str | None = Field(default=None, max_length=2000)


class ReaderBuyerPersonaSnapshot(BaseModel):
    """Concise internal-facing reader/buyer persona for positioning work."""

    persona_name: str
    role_context: str
    motivation: str
    needs: str
    likely_objections: str
    discovery_channels: str
    messaging_notes: str


class BookBriefResponse(BaseModel):
    """Structured Book Brief output."""

    title: str | None = None
    genre: str | None = None
    primary_audience: str
    secondary_audience: str | None = None
    reader_buyer_persona: ReaderBuyerPersonaSnapshot
    core_themes: list[str]
    audience_keywords: list[str]
    one_sentence_positioning: str
    positioning_recommendation: str
    marketing_angles: list[str]
    sales_use_case: str
    risk_flags: list[str]
    workflow_trace: WorkflowTrace | None = None
    quality_checks: QualityChecks | None = None
    sources: list[SourceChunk]
    metadata: RunMetadata
    run_details: RunMetadata | None = None


class MarketingCopyRequest(BaseModel):
    """Input payload for marketing copy generation."""

    output_type: Literal[
        "back_cover",
        "newsletter",
        "bookstore_pitch",
        "instagram_caption",
        "tiktok_hooks",
        "author_interview_questions",
        "book_club_questions",
    ]
    tone: str | None = Field(default=None, max_length=128)
    audience: str | None = Field(default=None, max_length=256)
    spoiler_level: str | None = Field(default=None, max_length=32)
    length: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2000)


class MarketingCopyResponse(BaseModel):
    """Structured marketing copy output."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    output_type: str
    copy_text: str = Field(alias="copy")
    rationale: str | None = None
    workflow_trace: WorkflowTrace | None = None
    quality_checks: QualityChecks | None = None
    sources: list[SourceChunk]
    metadata: RunMetadata
    run_details: RunMetadata | None = None


class ReaderPersonaRequest(BaseModel):
    """Input payload for reader persona generation."""

    audience: str | None = Field(default=None, max_length=256)
    notes: str | None = Field(default=None, max_length=2000)


class ReaderPersonaResponse(BaseModel):
    """Structured reader persona output."""

    persona_name: str
    reader_profile: str
    motivations: list[str]
    likely_interests: list[str]
    positioning_notes: list[str]
    sources: list[SourceChunk]
    metadata: RunMetadata
