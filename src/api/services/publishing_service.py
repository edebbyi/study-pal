"""publishing_service.py: Service layer for Publishing Mode generation endpoints."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from src.api.models import PublishingRunRecord
from src.api.services.api_logging import log_api_event
from src.api.services.evaluation_metrics import ModeType, build_live_run_metrics
from src.api.services.document_workspace import (
    DocumentRetrievalError,
    load_workspace,
    normalize_user_id,
    retrieve_workspace_context,
    workspace_chunks,
)
from src.api.services.observability_service import (
    end_mlflow_run,
    end_phoenix_trace,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    log_phoenix_generation,
    log_phoenix_retrieval,
    log_phoenix_scores,
    start_mlflow_run,
    start_phoenix_trace,
)
from src.api.services.publishing_runs import save_publishing_run
from src.api.services.service_errors import ServiceTimeoutError
from src.api.services.timeout_utils import run_with_timeout
from src.core.config import settings
from src.core.models import RetrievedChunk
from src.core.openrouter_credentials import get_openrouter_api_key_for_user
from src.llm.llm_client import generate_book_brief_from_context, generate_marketing_copy_from_context
from src.publishing.schemas import (
    BookBriefRequest,
    BookBriefResponse,
    MarketingCopyRequest,
    MarketingCopyResponse,
    QualityChecks,
    ReaderBuyerPersonaSnapshot,
    RunMetadata,
    SourceChunk,
    WorkflowTrace,
)


DEFAULT_OBSERVABILITY_TEMPERATURE = "not_exposed"


def _error_category_metrics(category: str) -> dict[str, float]:
    """Build one-hot style error category metrics for observability backends."""
    return {
        "error_count": 1.0,
        "error_timeout_failure": 1.0 if category == "timeout_failure" else 0.0,
        "error_retrieval_failure": 1.0 if category == "retrieval_failure" else 0.0,
        "error_model_failure": 1.0 if category == "model_failure" else 0.0,
        "error_validation_failure": 1.0 if category == "validation_failure" else 0.0,
        "error_unexpected_failure": 1.0 if category == "unexpected_failure" else 0.0,
    }


def _insufficient() -> str:
    return "Insufficient evidence from provided sources."


def _workspace_title_hint(workspace: dict[str, object]) -> str | None:
    raw_title = workspace.get("document_title")
    if not isinstance(raw_title, str):
        return None
    cleaned = raw_title.strip()
    return cleaned or None


def _infer_genre_from_sources(sources: list[SourceChunk]) -> str | None:
    """Infer a conservative genre/category hint from source text."""
    corpus = " ".join(source.text.lower() for source in sources if source.text).strip()
    if not corpus:
        return None
    if any(token in corpus for token in ["chapter", "glossary", "index", "primer", "companion publication"]):
        return "Educational Nonfiction / Reference"
    if any(token in corpus for token in ["study", "research", "scientist", "neuroscience"]):
        return "Science Nonfiction"
    return None


def _fallback_primary_audience(request: BookBriefRequest) -> str:
    """Resolve a conservative primary audience fallback."""
    if request.audience and request.audience.strip():
        return request.audience.strip()
    return "General readers seeking structured, credible subject-matter guidance (inferred)."


def _fallback_persona_snapshot(primary_audience: str) -> ReaderBuyerPersonaSnapshot:
    """Build a conservative reader/buyer persona snapshot for fallback responses."""
    return ReaderBuyerPersonaSnapshot(
        persona_name="Inferred Core Buyer",
        role_context=f"Inferred from available context: {primary_audience}.",
        motivation="Seeks accurate, accessible material to support learning, reference use, or professional context (inferred).",
        needs="Clear structure, trustworthy sourcing, and practical explanatory depth that can support varied reading goals (inferred).",
        likely_objections="May need supplementary assets or clearer level-specific guidance not visible in retrieved excerpts.",
        discovery_channels="Likely discovered through libraries, bookstores, educator/professional channels, or referrals (inferred).",
        messaging_notes="Emphasize credibility, clarity, and practical value; avoid unsupported market or channel-specific claims.",
    )


def _build_book_brief_retrieval_query(
    *,
    workspace: dict[str, object],
    request: BookBriefRequest,
) -> str:
    """Build a retrieval query that targets publishable brief details."""
    title_hint = _workspace_title_hint(workspace) or ""
    audience = request.audience.strip() if request.audience else ""
    spoiler_level = request.spoiler_level.strip() if request.spoiler_level else ""
    notes = request.notes.strip() if request.notes else ""
    query_parts = [
        "book summary genre themes tone audience hook marketing sales positioning content warnings",
        title_hint,
        audience,
        spoiler_level,
        notes,
    ]
    return " ".join(part for part in query_parts if part).strip()


def _representative_workspace_context(
    *,
    workspace: dict[str, object],
    max_points: int,
) -> list[RetrievedChunk]:
    """Sample representative chunks across the document for broader grounding."""
    raw_chunks = workspace_chunks(workspace)
    if not raw_chunks:
        return []

    total = len(raw_chunks)
    if total <= max_points:
        sampled = raw_chunks
    else:
        step = max(total // max_points, 1)
        sampled = [raw_chunks[index] for index in range(0, total, step)][:max_points]
        if sampled and sampled[-1].chunk_id != raw_chunks[-1].chunk_id:
            sampled[-1] = raw_chunks[-1]

    return [
        RetrievedChunk(
            text=chunk.text,
            filename=chunk.filename,
            page=chunk.page,
            citation=chunk.citation,
            score=0.0,
            chunk_id=chunk.chunk_id,
            chapter=chunk.chapter,
            topic=chunk.topic,
        )
        for chunk in sampled
    ]


def _merge_with_representative_chunks(
    *,
    workspace: dict[str, object],
    retrieved_chunks: list[RetrievedChunk],
    min_total: int,
) -> list[RetrievedChunk]:
    """Merge semantic retrieval with representative chunks to reduce narrow-context outputs."""
    merged: list[RetrievedChunk] = []
    seen_chunk_ids: set[int] = set()

    for chunk in retrieved_chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        merged.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)

    if len(merged) >= min_total:
        return merged

    representatives = _representative_workspace_context(
        workspace=workspace,
        max_points=min_total - len(merged),
    )
    for chunk in representatives:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        merged.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(merged) >= min_total:
            break
    return merged


def _book_brief_sources(retrieved_chunks: list[object]) -> list[SourceChunk]:
    """Map retrieved chunks into API source schema."""
    sources: list[SourceChunk] = []
    for chunk in retrieved_chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        text = getattr(chunk, "text", "")
        score = getattr(chunk, "score", 0.0)
        sources.append(
            SourceChunk(
                chunk_id=str(chunk_id),
                text=str(text),
                score=float(score),
            )
        )
    return sources


def _build_marketing_copy_retrieval_query(
    *,
    workspace: dict[str, object],
    request: MarketingCopyRequest,
) -> str:
    """Build a retrieval query that targets marketing-copy evidence."""
    title_hint = _workspace_title_hint(workspace) or ""
    tone = request.tone.strip() if request.tone else ""
    audience = request.audience.strip() if request.audience else ""
    spoiler_level = request.spoiler_level.strip() if request.spoiler_level else ""
    length = request.length.strip() if request.length else ""
    notes = request.notes.strip() if request.notes else ""
    query_parts = [
        "book summary hooks positioning themes tone audience promotional copy sales angle",
        request.output_type,
        title_hint,
        tone,
        audience,
        spoiler_level,
        length,
        notes,
    ]
    return " ".join(part for part in query_parts if part).strip()


def _normalized_spoiler_level(spoiler_level: str | None) -> str:
    """Normalize spoiler level for trace/check reporting."""
    cleaned = (spoiler_level or "").strip().lower()
    return cleaned or "low"


def _contains_insufficient_signal(value: object) -> bool:
    """Detect explicit missing-context sentinel text in nested structures."""
    if isinstance(value, str):
        return _insufficient().lower() in value.lower()
    if isinstance(value, list):
        return any(_contains_insufficient_signal(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_insufficient_signal(item) for item in value.values())
    return False


def _chunk_preview_payload(sources: list[SourceChunk]) -> list[dict[str, object]]:
    """Return compact chunk payloads for observability artifacts."""
    payload: list[dict[str, object]] = []
    for source in sources:
        preview = " ".join(source.text.split())
        if len(preview) > 260:
            preview = preview[:260].rstrip() + "..."
        payload.append(
            {
                "chunk_id": source.chunk_id,
                "score": source.score,
                "preview": preview,
            }
        )
    return payload


def _workspace_document_type(workspace: dict[str, object]) -> str | None:
    """Infer a simple document type from workspace filename when available."""
    raw_filename = workspace.get("filename")
    if not isinstance(raw_filename, str):
        return None
    filename = raw_filename.strip()
    if "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower() or None


def _build_eval_payload(
    *,
    retrieved_chunks: list[object],
    mode: ModeType,
    selected_output: str,
    spoiler_level: str,
    missing_context_flags: list[str],
    unsupported_claims_detected: bool | None,
) -> tuple[WorkflowTrace, QualityChecks, dict[str, object]]:
    """Build workflow trace + quality checks + numeric metrics."""
    metrics_payload = build_live_run_metrics(
        retrieved_chunks=retrieved_chunks,
        mode=mode,
        output_type=selected_output,
        missing_context_flags=missing_context_flags,
        unsupported_claims_detected=unsupported_claims_detected,
    )
    workflow_trace_dict = dict(metrics_payload.get("workflow_trace", {}))
    workflow_trace_dict["spoiler_setting"] = spoiler_level

    quality_checks_dict = dict(metrics_payload.get("quality_checks", {}))
    quality_checks_dict["spoiler_level"] = spoiler_level

    workflow_trace = WorkflowTrace.model_validate(workflow_trace_dict)
    quality_checks = QualityChecks.model_validate(quality_checks_dict)
    run_details_metrics = dict(metrics_payload.get("run_details_metrics", {}))
    return workflow_trace, quality_checks, run_details_metrics


def _build_run_metadata(
    *,
    model: str,
    latency_ms: int,
    retrieved_chunk_count: int,
    run_id: str,
    run_details_metrics: dict[str, object],
    phoenix_trace_id: str | None,
    mlflow_run_id: str | None,
) -> RunMetadata:
    """Build normalized run metadata for API response + persistence."""
    return RunMetadata(
        model=model,
        latency_ms=latency_ms,
        latency_seconds=round(latency_ms / 1000, 3),
        retrieved_chunk_count=retrieved_chunk_count,
        avg_relevance_score=float(run_details_metrics["avg_relevance_score"]) if isinstance(run_details_metrics.get("avg_relevance_score"), (int, float)) else None,
        top_relevance_score=float(run_details_metrics["top_relevance_score"]) if isinstance(run_details_metrics.get("top_relevance_score"), (int, float)) else None,
        source_diversity_score=float(run_details_metrics["source_diversity_score"]) if isinstance(run_details_metrics.get("source_diversity_score"), (int, float)) else None,
        context_coverage_score=float(run_details_metrics["context_coverage_score"]) if isinstance(run_details_metrics.get("context_coverage_score"), (int, float)) else None,
        phoenix_trace_id=phoenix_trace_id,
        mlflow_run_id=mlflow_run_id,
        run_id=run_id,
    )


def _fallback_book_brief_response(
    *,
    request: BookBriefRequest,
    workspace: dict[str, object],
    sources: list[SourceChunk],
    latency_ms: int,
    run_id: str,
    phoenix_trace_id: str | None,
    mlflow_run_id: str | None,
) -> BookBriefResponse:
    """Build a grounded fallback Book Brief when model generation is unavailable."""
    fallback_primary_audience = _fallback_primary_audience(request)
    evidence_snippet = _insufficient()
    if sources:
        first_text = " ".join(sources[0].text.split())
        if first_text:
            evidence_snippet = first_text[:260].rstrip()
            if len(first_text) > 260:
                evidence_snippet += "..."
    inferred_genre = _infer_genre_from_sources(sources)
    spoiler_level = _normalized_spoiler_level(request.spoiler_level)
    missing_context_flags = [
        "Live model generation was unavailable for this run; this brief is a conservative grounded fallback.",
        "Positioning is inferred from retrieved excerpts and should be reviewed against full-document context.",
    ]
    unsupported_claims_detected = False
    workflow_trace, quality_checks, run_details_metrics = _build_eval_payload(
        retrieved_chunks=sources,
        mode="positioning_brief",
        selected_output="positioning_brief",
        spoiler_level=spoiler_level,
        missing_context_flags=missing_context_flags,
        unsupported_claims_detected=unsupported_claims_detected,
    )
    metadata = _build_run_metadata(
        model="retrieval_fallback",
        latency_ms=latency_ms,
        retrieved_chunk_count=len(sources),
        run_id=run_id,
        run_details_metrics=run_details_metrics,
        phoenix_trace_id=phoenix_trace_id,
        mlflow_run_id=mlflow_run_id,
    )

    return BookBriefResponse(
        title=_workspace_title_hint(workspace),
        genre=inferred_genre,
        primary_audience=fallback_primary_audience,
        secondary_audience=None,
        reader_buyer_persona=_fallback_persona_snapshot(fallback_primary_audience),
        core_themes=[
            "Foundational overview of core subject concepts identified across retrieved excerpts (inferred).",
            "Structured progression through major topics using chapter-based organization cues (inferred).",
            "Educational/reference framing supported by terminology and document metadata cues (inferred).",
        ],
        audience_keywords=[
            "accessible reference",
            "structured chapters",
            "credible sourcing",
            "concept-driven overview",
        ],
        one_sentence_positioning=evidence_snippet,
        positioning_recommendation=(
            "Position as a source-grounded reference overview, emphasizing clarity, breadth, and credibility. "
            "Treat audience/channel specifics as inferred pending additional market inputs."
        ),
        marketing_angles=[
            "Breadth-first overview of major concepts",
            "Credibility-forward educational framing",
            "Useful as reference or supplementary reading",
        ],
        sales_use_case=(
            "Best suited for academic, library, and education-focused channels seeking an accessible reference. "
            "Potential users include educators, students, and general readers interested in a structured overview."
        ),
        risk_flags=[
            *missing_context_flags,
            "Generate channel-specific assets in Marketing Copy after human review.",
        ],
        workflow_trace=workflow_trace,
        quality_checks=quality_checks,
        sources=sources,
        metadata=metadata,
        run_details=metadata,
    )


def _normalize_book_brief_payload(
    payload: dict[str, object],
    *,
    audience_hint: str | None,
) -> dict[str, object]:
    """Normalize model payload into the strategic Positioning Brief schema."""
    primary_audience = payload.get("primary_audience")
    if not isinstance(primary_audience, str) or not primary_audience.strip():
        legacy_target_reader = payload.get("target_reader")
        if isinstance(legacy_target_reader, str) and legacy_target_reader.strip():
            primary_audience = legacy_target_reader.strip()
        elif audience_hint and audience_hint.strip():
            primary_audience = audience_hint.strip()
        else:
            primary_audience = _insufficient()

    one_sentence_positioning = payload.get("one_sentence_positioning")
    if not isinstance(one_sentence_positioning, str) or not one_sentence_positioning.strip():
        legacy_pitch = payload.get("one_sentence_pitch")
        if isinstance(legacy_pitch, str) and legacy_pitch.strip():
            one_sentence_positioning = legacy_pitch.strip()
        else:
            one_sentence_positioning = _insufficient()

    positioning_recommendation = payload.get("positioning_recommendation")
    if not isinstance(positioning_recommendation, str) or not positioning_recommendation.strip():
        legacy_back_cover = payload.get("back_cover_copy")
        if isinstance(legacy_back_cover, str) and legacy_back_cover.strip():
            positioning_recommendation = legacy_back_cover.strip()
        else:
            positioning_recommendation = _insufficient()

    sales_use_case = payload.get("sales_use_case")
    if not isinstance(sales_use_case, str) or not sales_use_case.strip():
        legacy_sales = payload.get("sales_positioning")
        if isinstance(legacy_sales, str) and legacy_sales.strip():
            sales_use_case = legacy_sales.strip()
        else:
            sales_use_case = _insufficient()

    risk_flags = payload.get("risk_flags")
    if not isinstance(risk_flags, list) or not risk_flags:
        risk_flags = []
        if payload.get("genre") in (None, "", _insufficient()):
            risk_flags.append("Genre classification is uncertain from available excerpts.")
        if payload.get("comparable_titles") in (None, [], _insufficient()):
            risk_flags.append("Comparable titles were not clearly grounded in retrieved excerpts.")
        if payload.get("content_warnings") in (None, [], _insufficient()):
            risk_flags.append("No clearly grounded content warnings were identified in retrieved excerpts.")
        if not risk_flags:
            risk_flags.append("Review source evidence before external publication use.")

    persona_payload = payload.get("reader_buyer_persona")
    if not isinstance(persona_payload, dict):
        persona_payload = payload.get("reader_persona")
    if not isinstance(persona_payload, dict):
        persona_payload = {}

    persona_name = persona_payload.get("persona_name")
    if not isinstance(persona_name, str) or not persona_name.strip():
        persona_name = "Inferred Core Buyer"

    role_context = persona_payload.get("role_context")
    if not isinstance(role_context, str) or not role_context.strip():
        role_context = (
            f"Inferred from audience/context: {primary_audience}."
            if isinstance(primary_audience, str) and primary_audience != _insufficient()
            else _insufficient()
        )

    motivation = persona_payload.get("motivation")
    if not isinstance(motivation, str) or not motivation.strip():
        motivation = "Likely motivated by credible, accessible material suited to their role (inferred)."

    needs = persona_payload.get("needs")
    if not isinstance(needs, str) or not needs.strip():
        needs = "Likely needs clear structure, trustworthy sourcing, and practical applicability (inferred)."

    likely_objections = persona_payload.get("likely_objections")
    if not isinstance(likely_objections, str) or not likely_objections.strip():
        likely_objections = "May need supplemental materials or market context not visible in retrieved excerpts."

    discovery_channels = persona_payload.get("discovery_channels")
    if not isinstance(discovery_channels, str) or not discovery_channels.strip():
        discovery_channels = "Likely discovered through libraries, bookstores, professional networks, or referrals (inferred)."

    messaging_notes = persona_payload.get("messaging_notes")
    if not isinstance(messaging_notes, str) or not messaging_notes.strip():
        messaging_notes = "Keep messaging evidence-backed, practical, and role-relevant; avoid unsupported channel claims."

    normalized: dict[str, object] = {
        "title": payload.get("title"),
        "genre": payload.get("genre"),
        "primary_audience": primary_audience,
        "secondary_audience": payload.get("secondary_audience"),
        "reader_buyer_persona": {
            "persona_name": persona_name,
            "role_context": role_context,
            "motivation": motivation,
            "needs": needs,
            "likely_objections": likely_objections,
            "discovery_channels": discovery_channels,
            "messaging_notes": messaging_notes,
        },
        "core_themes": payload.get("core_themes") or [_insufficient()],
        "audience_keywords": payload.get("audience_keywords") or [_insufficient()],
        "one_sentence_positioning": one_sentence_positioning,
        "positioning_recommendation": positioning_recommendation,
        "marketing_angles": payload.get("marketing_angles") or [_insufficient()],
        "sales_use_case": sales_use_case,
        "risk_flags": risk_flags,
    }
    return normalized


def _fallback_marketing_copy_response(
    *,
    request: MarketingCopyRequest,
    sources: list[SourceChunk],
    latency_ms: int,
    run_id: str,
    phoenix_trace_id: str | None,
    mlflow_run_id: str | None,
) -> MarketingCopyResponse:
    """Build a grounded fallback marketing copy response."""
    first_snippet = _insufficient()
    if sources:
        cleaned = " ".join(sources[0].text.split())
        if cleaned:
            first_snippet = cleaned[:260].rstrip()
            if len(cleaned) > 260:
                first_snippet += "..."

    if request.output_type == "bookstore_pitch":
        fallback_copy = (
            "Bookseller pitch (grounded excerpt): "
            f"{first_snippet}"
        )
    elif request.output_type == "newsletter":
        fallback_copy = (
            "Newsletter blurb (grounded excerpt): "
            f"{first_snippet}"
        )
    elif request.output_type == "back_cover":
        fallback_copy = (
            "Back-cover draft (grounded excerpt): "
            f"{first_snippet}"
        )
    else:
        fallback_copy = first_snippet
    spoiler_level = _normalized_spoiler_level(request.spoiler_level)
    missing_context_flags = ["Model output unavailable; generated grounded fallback from retrieved chunks."]
    unsupported_claims_detected = False
    workflow_trace, quality_checks, run_details_metrics = _build_eval_payload(
        retrieved_chunks=sources,
        mode="marketing_copy",
        selected_output=request.output_type,
        spoiler_level=spoiler_level,
        missing_context_flags=missing_context_flags,
        unsupported_claims_detected=unsupported_claims_detected,
    )
    metadata = _build_run_metadata(
        model="retrieval_fallback",
        latency_ms=latency_ms,
        retrieved_chunk_count=len(sources),
        run_id=run_id,
        run_details_metrics=run_details_metrics,
        phoenix_trace_id=phoenix_trace_id,
        mlflow_run_id=mlflow_run_id,
    )

    return MarketingCopyResponse.model_validate(
        {
            "output_type": request.output_type,
            "copy": fallback_copy,
            "rationale": "Generated from grounded excerpts because model output was unavailable.",
            "workflow_trace": workflow_trace.model_dump(),
            "quality_checks": quality_checks.model_dump(),
            "sources": [source.model_dump() for source in sources],
            "metadata": metadata.model_dump(),
            "run_details": metadata.model_dump(),
        }
    )


def generate_book_brief_for_document(
    *,
    doc_id: str,
    request: BookBriefRequest,
    user_id: str | None = None,
) -> BookBriefResponse:
    """Generate a grounded Book Brief for a single document workspace.

    Raises:
        DocumentWorkspaceNotFoundError: If the document workspace cannot be found.
        DocumentRetrievalError: If no grounded context can be retrieved.
    """
    started_at = perf_counter()
    run_id = str(uuid4())
    normalized_user_id = normalize_user_id(user_id)
    log_api_event(
        "service.positioning_brief.start",
        run_id=run_id,
        doc_id=doc_id,
        mode="positioning_brief",
        output_type="book_brief",
    )
    phoenix_trace = start_phoenix_trace(
        span_name="publishing.positioning_brief",
        metadata={
            "run_id": run_id,
            "doc_id": doc_id,
            "mode": "positioning_brief",
            "output_type": "book_brief",
            "prompt_version": settings.prompt_version or "",
            "model": settings.chat_model,
            "top_k": max(settings.top_k, 8),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
        },
    )
    mlflow_run = start_mlflow_run(
        run_name=f"publishing.positioning_brief.{run_id}",
        tags={
            "run_id": run_id,
            "document_id": doc_id,
            "prompt_version": settings.prompt_version or "",
            "mode": "positioning_brief",
            "output_type": "book_brief",
            "app_env": settings.app_env,
        },
    )
    log_mlflow_params(
        mlflow_run,
        {
            "mode": "positioning_brief",
            "output_type": "book_brief",
            "model": settings.chat_model,
            "prompt_version": settings.prompt_version,
            "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
            "top_k": max(settings.top_k, 8),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
            "document_type": "unknown",
            "spoiler_level": request.spoiler_level or "low",
            "audience_provided": "yes" if request.audience else "no",
        },
    )

    try:
        workspace = load_workspace(doc_id, normalized_user_id)
        retrieval_query = _build_book_brief_retrieval_query(workspace=workspace, request=request)
        try:
            retrieved_chunks = retrieve_workspace_context(
                workspace=workspace,
                question=retrieval_query,
                user_id=normalized_user_id,
                top_k=max(settings.top_k, 8),
            )
        except DocumentRetrievalError:
            raw_chunks = workspace_chunks(workspace)
            if not raw_chunks:
                raise
            retrieved_chunks = [
                RetrievedChunk(
                    text=chunk.text,
                    filename=chunk.filename,
                    page=chunk.page,
                    citation=chunk.citation,
                    score=0.0,
                    chunk_id=chunk.chunk_id,
                    chapter=chunk.chapter,
                    topic=chunk.topic,
                )
                for chunk in raw_chunks[: max(settings.top_k, 8)]
            ]
        retrieved_chunks = _merge_with_representative_chunks(
            workspace=workspace,
            retrieved_chunks=retrieved_chunks,
            min_total=max(settings.top_k, 10),
        )
        sources = _book_brief_sources(retrieved_chunks)
        log_api_event(
            "service.positioning_brief.retrieval_complete",
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            retrieved_chunk_count=len(sources),
        )
        spoiler_level = _normalized_spoiler_level(request.spoiler_level)
        log_phoenix_retrieval(
            phoenix_trace,
            {
                "mode": "positioning_brief",
                "output_type": "book_brief",
                "document_id": doc_id,
                "document_title": _workspace_title_hint(workspace),
                "prompt_version": settings.prompt_version or "",
                "model": settings.chat_model,
                "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
                "top_k": max(settings.top_k, 8),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "retrieval_algorithm": settings.retrieval_algorithm,
                "audience": request.audience,
                "spoiler_level": request.spoiler_level,
                "length": None,
                "tone": None,
                "notes": request.notes,
                "retrieved_chunk_count": len(sources),
                "chunks": _chunk_preview_payload(sources),
            },
        )
        log_mlflow_params(
            mlflow_run,
            {
                "document_type": _workspace_document_type(workspace) or "unknown",
            },
        )

        payload = run_with_timeout(
            operation="generate_book_brief_from_context",
            mode="positioning_brief",
            run_output_type="book_brief",
            timeout_seconds=settings.api_timeout_positioning_seconds,
            fn=generate_book_brief_from_context,
            retrieved_chunks=retrieved_chunks,
            audience=request.audience,
            spoiler_level=request.spoiler_level,
            notes=request.notes,
            document_title_hint=_workspace_title_hint(workspace),
            api_key_override=get_openrouter_api_key_for_user(normalized_user_id),
        )
        if payload is None:
            log_api_event(
                "service.positioning_brief.retry",
                level=logging.WARNING,
                run_id=run_id,
                doc_id=doc_id,
                mode="positioning_brief",
                output_type="book_brief",
            )
            payload = run_with_timeout(
                operation="generate_book_brief_from_context.retry",
                mode="positioning_brief",
                run_output_type="book_brief",
                timeout_seconds=settings.api_timeout_positioning_seconds,
                fn=generate_book_brief_from_context,
                retrieved_chunks=retrieved_chunks,
                audience=request.audience,
                spoiler_level=request.spoiler_level,
                notes=request.notes,
                document_title_hint=_workspace_title_hint(workspace),
                api_key_override=get_openrouter_api_key_for_user(normalized_user_id),
            )
        latency_ms = int((perf_counter() - started_at) * 1000)

        if payload is None:
            log_api_event(
                "service.positioning_brief.fallback_used",
                level=logging.WARNING,
                run_id=run_id,
                doc_id=doc_id,
                mode="positioning_brief",
                output_type="book_brief",
            )
            response = _fallback_book_brief_response(
                request=request,
                workspace=workspace,
                sources=sources,
                latency_ms=latency_ms,
                run_id=run_id,
                phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
                mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            )
        else:
            normalized_payload = _normalize_book_brief_payload(payload, audience_hint=request.audience)
            risk_flags = normalized_payload.get("risk_flags")
            risk_list = [str(item) for item in risk_flags] if isinstance(risk_flags, list) else []
            missing_context_flags = risk_list if risk_list else ([_insufficient()] if _contains_insufficient_signal(normalized_payload) else [])
            unsupported_claims_detected = False
            workflow_trace, quality_checks, run_details_metrics = _build_eval_payload(
                retrieved_chunks=retrieved_chunks,
                mode="positioning_brief",
                selected_output="positioning_brief",
                spoiler_level=spoiler_level,
                missing_context_flags=missing_context_flags,
                unsupported_claims_detected=unsupported_claims_detected,
            )
            metadata = _build_run_metadata(
                model=settings.chat_model,
                latency_ms=latency_ms,
                retrieved_chunk_count=len(sources),
                run_id=run_id,
                run_details_metrics=run_details_metrics,
                phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
                mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            )
            response = BookBriefResponse.model_validate(
                {
                    **normalized_payload,
                    "workflow_trace": workflow_trace.model_dump(),
                    "quality_checks": quality_checks.model_dump(),
                    "sources": [source.model_dump() for source in sources],
                    "metadata": metadata.model_dump(),
                    "run_details": metadata.model_dump(),
                }
            )

        run_details_payload = response.run_details.model_dump() if response.run_details else response.metadata.model_dump()
        log_phoenix_generation(
            phoenix_trace,
            {
                "mode": "positioning_brief",
                "output_type": "book_brief",
                "model": response.metadata.model,
                "document_id": doc_id,
                "document_title": _workspace_title_hint(workspace),
                "latency_ms": response.metadata.latency_ms,
                "generated_output_preview": " ".join(str(response.positioning_recommendation).split())[:300],
            },
        )
        if response.quality_checks is not None:
            log_phoenix_scores(phoenix_trace, response.quality_checks.model_dump())
        log_mlflow_metrics(
            mlflow_run,
            {
                "latency_ms": response.metadata.latency_ms,
                "latency_seconds": run_details_payload.get("latency_seconds"),
                "retrieved_chunk_count": response.metadata.retrieved_chunk_count,
                "avg_relevance_score": run_details_payload.get("avg_relevance_score"),
                "top_relevance_score": run_details_payload.get("top_relevance_score"),
                "chunk_count_score": run_details_payload.get("chunk_count_score"),
                "source_diversity_score": run_details_payload.get("source_diversity_score"),
                "context_coverage_score": run_details_payload.get("context_coverage_score"),
                "input_token_count": run_details_payload.get("input_token_count"),
                "output_token_count": run_details_payload.get("output_token_count"),
                "total_token_count": run_details_payload.get("total_token_count"),
                "estimated_cost_usd": run_details_payload.get("estimated_cost_usd"),
                "unsupported_claim_count": run_details_payload.get("unsupported_claim_count"),
            },
        )
        log_mlflow_artifacts(
            mlflow_run,
            {
                "generated_output": response.model_dump(),
                "retrieved_chunks": _chunk_preview_payload(response.sources),
                "quality_checks": response.quality_checks.model_dump() if response.quality_checks else {},
                "workflow_trace": response.workflow_trace.model_dump() if response.workflow_trace else {},
                "run_details": run_details_payload,
                "request_metadata": {
                    "doc_id": doc_id,
                    "mode": "positioning_brief",
                    "output_type": "book_brief",
                    "audience": request.audience,
                    "spoiler_level": request.spoiler_level,
                    "notes": request.notes,
                },
            },
        )
        save_publishing_run(
            PublishingRunRecord(
                run_id=run_id,
                doc_id=doc_id,
                endpoint="book_brief",
                output_type="book_brief",
                model=response.metadata.model,
                latency_ms=response.metadata.latency_ms,
                retrieved_chunk_count=response.metadata.retrieved_chunk_count,
                estimated_cost=response.metadata.estimated_cost_usd,
                phoenix_trace_id=response.metadata.phoenix_trace_id,
                mlflow_run_id=response.metadata.mlflow_run_id,
                created_at=response.metadata.created_at,
                user_rating=None,
                user_feedback=None,
            )
        )
        end_mlflow_run(mlflow_run, status="FINISHED")
        log_api_event(
            "service.positioning_brief.complete",
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            model=response.metadata.model,
            latency_ms=response.metadata.latency_ms,
            retrieved_chunk_count=response.metadata.retrieved_chunk_count,
            context_coverage_label=response.workflow_trace.context_coverage_label if response.workflow_trace else None,
        )
        return response
    except ValidationError:
        latency_ms = int((perf_counter() - started_at) * 1000)
        workspace = load_workspace(doc_id, normalized_user_id)
        sources = _book_brief_sources(workspace_chunks(workspace)[: max(settings.top_k, 8)])
        fallback_response = _fallback_book_brief_response(
            request=request,
            workspace=workspace,
            sources=sources,
            latency_ms=latency_ms,
            run_id=run_id,
            phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
            mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
        )
        end_mlflow_run(mlflow_run, status="FINISHED")
        log_api_event(
            "service.positioning_brief.validation_fallback",
            level=logging.WARNING,
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            latency_ms=latency_ms,
            retrieved_chunk_count=len(sources),
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("validation_failure"))
        return fallback_response
    except ServiceTimeoutError:
        log_api_event(
            "service.positioning_brief.timeout",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            timeout_seconds=settings.api_timeout_positioning_seconds,
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "timeout_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("timeout_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    except DocumentRetrievalError:
        log_api_event(
            "service.positioning_brief.retrieval_failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "retrieval_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("retrieval_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    except Exception as exc:
        log_api_event(
            "service.positioning_brief.failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            error_type=type(exc).__name__,
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("model_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    finally:
        end_phoenix_trace(phoenix_trace)


def generate_marketing_copy_for_document(
    *,
    doc_id: str,
    request: MarketingCopyRequest,
    user_id: str | None = None,
) -> MarketingCopyResponse:
    """Generate grounded marketing copy for one document workspace."""
    started_at = perf_counter()
    run_id = str(uuid4())
    normalized_user_id = normalize_user_id(user_id)
    log_api_event(
        "service.marketing_copy.start",
        run_id=run_id,
        doc_id=doc_id,
        mode="marketing_copy",
        output_type=request.output_type,
    )
    phoenix_trace = start_phoenix_trace(
        span_name="publishing.marketing_copy",
        metadata={
            "run_id": run_id,
            "doc_id": doc_id,
            "mode": "marketing_copy",
            "output_type": request.output_type,
            "prompt_version": settings.prompt_version or "",
            "model": settings.chat_model,
            "top_k": max(settings.top_k, 8),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
        },
    )
    mlflow_run = start_mlflow_run(
        run_name=f"publishing.marketing_copy.{run_id}",
        tags={
            "run_id": run_id,
            "document_id": doc_id,
            "prompt_version": settings.prompt_version or "",
            "mode": "marketing_copy",
            "output_type": request.output_type,
            "app_env": settings.app_env,
        },
    )
    log_mlflow_params(
        mlflow_run,
        {
            "mode": "marketing_copy",
            "output_type": request.output_type,
            "model": settings.chat_model,
            "prompt_version": settings.prompt_version,
            "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
            "top_k": max(settings.top_k, 8),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
            "document_type": "unknown",
            "spoiler_level": request.spoiler_level or "low",
            "tone": request.tone,
            "audience_provided": "yes" if request.audience else "no",
        },
    )

    try:
        workspace = load_workspace(doc_id, normalized_user_id)
        retrieval_query = _build_marketing_copy_retrieval_query(workspace=workspace, request=request)
        try:
            retrieved_chunks = retrieve_workspace_context(
                workspace=workspace,
                question=retrieval_query,
                user_id=normalized_user_id,
                top_k=max(settings.top_k, 8),
            )
        except DocumentRetrievalError:
            raw_chunks = workspace_chunks(workspace)
            if not raw_chunks:
                raise
            retrieved_chunks = [
                RetrievedChunk(
                    text=chunk.text,
                    filename=chunk.filename,
                    page=chunk.page,
                    citation=chunk.citation,
                    score=0.0,
                    chunk_id=chunk.chunk_id,
                    chapter=chunk.chapter,
                    topic=chunk.topic,
                )
                for chunk in raw_chunks[: max(settings.top_k, 8)]
            ]
        sources = _book_brief_sources(retrieved_chunks)
        log_api_event(
            "service.marketing_copy.retrieval_complete",
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=request.output_type,
            retrieved_chunk_count=len(sources),
        )
        spoiler_level = _normalized_spoiler_level(request.spoiler_level)
        log_phoenix_retrieval(
            phoenix_trace,
            {
                "mode": "marketing_copy",
                "output_type": request.output_type,
                "document_id": doc_id,
                "document_title": _workspace_title_hint(workspace),
                "prompt_version": settings.prompt_version or "",
                "model": settings.chat_model,
                "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
                "top_k": max(settings.top_k, 8),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "retrieval_algorithm": settings.retrieval_algorithm,
                "audience": request.audience,
                "tone": request.tone,
                "spoiler_level": request.spoiler_level,
                "length": request.length,
                "notes": request.notes,
                "retrieved_chunk_count": len(sources),
                "chunks": _chunk_preview_payload(sources),
            },
        )
        log_mlflow_params(
            mlflow_run,
            {
                "document_type": _workspace_document_type(workspace) or "unknown",
            },
        )

        payload = run_with_timeout(
            operation="generate_marketing_copy_from_context",
            mode="marketing_copy",
            run_output_type=request.output_type,
            timeout_seconds=settings.api_timeout_marketing_seconds,
            fn=generate_marketing_copy_from_context,
            retrieved_chunks=retrieved_chunks,
            output_type=request.output_type,
            tone=request.tone,
            audience=request.audience,
            spoiler_level=request.spoiler_level,
            length=request.length,
            notes=request.notes,
            document_title_hint=_workspace_title_hint(workspace),
            api_key_override=get_openrouter_api_key_for_user(normalized_user_id),
        )
        if payload is None:
            log_api_event(
                "service.marketing_copy.retry",
                level=logging.WARNING,
                run_id=run_id,
                doc_id=doc_id,
                mode="marketing_copy",
                output_type=request.output_type,
            )
            payload = run_with_timeout(
                operation="generate_marketing_copy_from_context.retry",
                mode="marketing_copy",
                run_output_type=request.output_type,
                timeout_seconds=settings.api_timeout_marketing_seconds,
                fn=generate_marketing_copy_from_context,
                retrieved_chunks=retrieved_chunks,
                output_type=request.output_type,
                tone=request.tone,
                audience=request.audience,
                spoiler_level=request.spoiler_level,
                length=request.length,
                notes=request.notes,
                document_title_hint=_workspace_title_hint(workspace),
                api_key_override=get_openrouter_api_key_for_user(normalized_user_id),
            )
        latency_ms = int((perf_counter() - started_at) * 1000)

        if payload is None:
            log_api_event(
                "service.marketing_copy.fallback_used",
                level=logging.WARNING,
                run_id=run_id,
                doc_id=doc_id,
                mode="marketing_copy",
                output_type=request.output_type,
            )
            response = _fallback_marketing_copy_response(
                request=request,
                sources=sources,
                latency_ms=latency_ms,
                run_id=run_id,
                phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
                mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            )
        else:
            output_copy = payload.get("copy")
            if not isinstance(output_copy, str):
                output_copy = str(output_copy or "")
            output_rationale = payload.get("rationale")
            if not isinstance(output_rationale, str):
                output_rationale = str(output_rationale or "")
            missing_context_flags = []
            if _contains_insufficient_signal(output_copy) or _contains_insufficient_signal(output_rationale):
                missing_context_flags.append(_insufficient())
            unsupported_claims_detected = False
            workflow_trace, quality_checks, run_details_metrics = _build_eval_payload(
                retrieved_chunks=retrieved_chunks,
                mode="marketing_copy",
                selected_output=request.output_type,
                spoiler_level=spoiler_level,
                missing_context_flags=missing_context_flags,
                unsupported_claims_detected=unsupported_claims_detected,
            )
            metadata = _build_run_metadata(
                model=settings.chat_model,
                latency_ms=latency_ms,
                retrieved_chunk_count=len(sources),
                run_id=run_id,
                run_details_metrics=run_details_metrics,
                phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
                mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            )
            response = MarketingCopyResponse.model_validate(
                {
                    **payload,
                    "output_type": str(payload.get("output_type") or request.output_type),
                    "workflow_trace": workflow_trace.model_dump(),
                    "quality_checks": quality_checks.model_dump(),
                    "sources": [source.model_dump() for source in sources],
                    "metadata": metadata.model_dump(),
                    "run_details": metadata.model_dump(),
                }
            )

        run_details_payload = response.run_details.model_dump() if response.run_details else response.metadata.model_dump()
        log_phoenix_generation(
            phoenix_trace,
            {
                "mode": "marketing_copy",
                "output_type": request.output_type,
                "model": response.metadata.model,
                "document_id": doc_id,
                "document_title": _workspace_title_hint(workspace),
                "latency_ms": response.metadata.latency_ms,
                "generated_output_preview": str(response.copy_text)[:300],
            },
        )
        if response.quality_checks is not None:
            log_phoenix_scores(phoenix_trace, response.quality_checks.model_dump())
        log_mlflow_metrics(
            mlflow_run,
            {
                "latency_ms": response.metadata.latency_ms,
                "latency_seconds": run_details_payload.get("latency_seconds"),
                "retrieved_chunk_count": response.metadata.retrieved_chunk_count,
                "avg_relevance_score": run_details_payload.get("avg_relevance_score"),
                "top_relevance_score": run_details_payload.get("top_relevance_score"),
                "chunk_count_score": run_details_payload.get("chunk_count_score"),
                "source_diversity_score": run_details_payload.get("source_diversity_score"),
                "context_coverage_score": run_details_payload.get("context_coverage_score"),
                "input_token_count": run_details_payload.get("input_token_count"),
                "output_token_count": run_details_payload.get("output_token_count"),
                "total_token_count": run_details_payload.get("total_token_count"),
                "estimated_cost_usd": run_details_payload.get("estimated_cost_usd"),
                "unsupported_claim_count": run_details_payload.get("unsupported_claim_count"),
            },
        )
        log_mlflow_artifacts(
            mlflow_run,
            {
                "generated_output": response.model_dump(),
                "retrieved_chunks": _chunk_preview_payload(response.sources),
                "quality_checks": response.quality_checks.model_dump() if response.quality_checks else {},
                "workflow_trace": response.workflow_trace.model_dump() if response.workflow_trace else {},
                "run_details": run_details_payload,
                "request_metadata": {
                    "doc_id": doc_id,
                    "mode": "marketing_copy",
                    "output_type": request.output_type,
                    "tone": request.tone,
                    "audience": request.audience,
                    "spoiler_level": request.spoiler_level,
                    "length": request.length,
                    "notes": request.notes,
                },
            },
        )
        save_publishing_run(
            PublishingRunRecord(
                run_id=run_id,
                doc_id=doc_id,
                endpoint="marketing_copy",
                output_type=request.output_type,
                model=response.metadata.model,
                latency_ms=response.metadata.latency_ms,
                retrieved_chunk_count=response.metadata.retrieved_chunk_count,
                estimated_cost=response.metadata.estimated_cost_usd,
                phoenix_trace_id=response.metadata.phoenix_trace_id,
                mlflow_run_id=response.metadata.mlflow_run_id,
                created_at=response.metadata.created_at,
                user_rating=None,
                user_feedback=None,
            )
        )
        end_mlflow_run(mlflow_run, status="FINISHED")
        log_api_event(
            "service.marketing_copy.complete",
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=response.output_type,
            model=response.metadata.model,
            latency_ms=response.metadata.latency_ms,
            retrieved_chunk_count=response.metadata.retrieved_chunk_count,
            context_coverage_label=response.workflow_trace.context_coverage_label if response.workflow_trace else None,
        )
        return response
    except ValidationError:
        latency_ms = int((perf_counter() - started_at) * 1000)
        fallback_response = _fallback_marketing_copy_response(
            request=request,
            sources=[],
            latency_ms=latency_ms,
            run_id=run_id,
            phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
            mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
        )
        end_mlflow_run(mlflow_run, status="FINISHED")
        log_api_event(
            "service.marketing_copy.validation_fallback",
            level=logging.WARNING,
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=request.output_type,
            latency_ms=latency_ms,
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("validation_failure"))
        return fallback_response
    except ServiceTimeoutError:
        log_api_event(
            "service.marketing_copy.timeout",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=request.output_type,
            timeout_seconds=settings.api_timeout_marketing_seconds,
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "timeout_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("timeout_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    except DocumentRetrievalError:
        log_api_event(
            "service.marketing_copy.retrieval_failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=request.output_type,
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "retrieval_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("retrieval_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    except Exception as exc:
        log_api_event(
            "service.marketing_copy.failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=request.output_type,
            error_type=type(exc).__name__,
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("model_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        raise
    finally:
        end_phoenix_trace(phoenix_trace)
