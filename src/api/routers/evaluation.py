"""evaluation.py: Placeholder routes for run logging/evaluation APIs."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.models import PlaceholderResponse


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRunRequest(BaseModel):
    """Input payload for evaluation logging endpoints."""

    endpoint: str
    rating: str | None = None
    notes: str | None = None


@router.post("/runs", response_model=PlaceholderResponse)
def log_run(payload: EvaluationRunRequest) -> PlaceholderResponse:
    """Log a run/evaluation record (placeholder)."""
    _ = payload
    return PlaceholderResponse(
        endpoint="/api/evaluation/runs",
        message="Run logging is not implemented yet.",
    )
