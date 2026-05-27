"""runs.py: Routes for Publishing Mode run feedback."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models import PublishingRunRecord, RunFeedbackRequest, RunRatingRequest
from src.api.services.observability_service import PhoenixTraceHandle, log_phoenix_scores
from src.api.services.publishing_runs import rate_publishing_run
from src.api.services.rate_limit import enforce_feedback_rate_limit


router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/{run_id}/rate", response_model=PublishingRunRecord)
def rate_run(
    run_id: str,
    payload: RunRatingRequest,
    _: None = Depends(enforce_feedback_rate_limit),
) -> PublishingRunRecord:
    """Save user rating/feedback for a previously logged run."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run id cannot be empty.",
        )

    try:
        record = rate_publishing_run(
            run_id=normalized_run_id,
            user_rating=payload.user_rating,
            user_feedback=payload.user_feedback.strip() if payload.user_feedback else None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{normalized_run_id}' was not found.",
        )
    return record


@router.post("/{run_id}/feedback", response_model=PublishingRunRecord)
def save_run_feedback(
    run_id: str,
    payload: RunFeedbackRequest,
    _: None = Depends(enforce_feedback_rate_limit),
) -> PublishingRunRecord:
    """Save richer user feedback while keeping backward-compatible run storage."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run id cannot be empty.",
        )

    normalized_thumbs = (payload.thumbs or "").strip().lower()
    notes = (payload.notes or "").strip()
    corrected_output = (payload.corrected_output or "").strip()
    fragments: list[str] = []
    if normalized_thumbs in {"up", "down"}:
        fragments.append(f"thumbs={normalized_thumbs}")
    if payload.useful is not None:
        fragments.append(f"useful={'yes' if payload.useful else 'no'}")
    if payload.grounded is not None:
        fragments.append(f"grounded={'yes' if payload.grounded else 'no'}")
    if notes:
        fragments.append(f"notes={notes}")
    if corrected_output:
        fragments.append(f"corrected_output={corrected_output}")
    merged_feedback = " | ".join(fragments) if fragments else None

    try:
        record = rate_publishing_run(
            run_id=normalized_run_id,
            user_rating=payload.rating,
            user_feedback=merged_feedback,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{normalized_run_id}' was not found.",
        )

    # Best-effort observability logging: never break API feedback writes.
    try:
        if record.phoenix_trace_id:
            log_phoenix_scores(
                PhoenixTraceHandle(trace_id=record.phoenix_trace_id, span_name="publishing.feedback"),
                {
                    "rating": payload.rating,
                    "thumbs": payload.thumbs,
                    "useful": payload.useful,
                    "grounded": payload.grounded,
                    "notes_present": bool((payload.notes or "").strip()),
                },
            )
    except Exception:
        pass

    try:
        if record.mlflow_run_id:
            import mlflow

            with mlflow.start_run(run_id=record.mlflow_run_id):
                if payload.rating is not None:
                    mlflow.log_metric("user_rating", float(payload.rating))
                if payload.useful is not None:
                    mlflow.log_metric("user_useful", 1.0 if payload.useful else 0.0)
                if payload.grounded is not None:
                    mlflow.log_metric("user_grounded", 1.0 if payload.grounded else 0.0)
                if payload.thumbs:
                    mlflow.set_tag("user_thumbs", payload.thumbs.strip().lower())
                if payload.notes:
                    mlflow.set_tag("user_notes_present", "yes")
    except Exception:
        pass

    return record
