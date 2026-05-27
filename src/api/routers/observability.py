"""observability.py: Health endpoints for observability integrations."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.models import ObservabilityHealthResponse
from src.api.services.observability_service import (
    get_phoenix_span_mode,
    init_mlflow_experiment,
    init_phoenix,
    is_mlflow_enabled,
    is_phoenix_enabled,
    is_phoenix_initialized,
)
from src.core.config import settings


router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/health", response_model=ObservabilityHealthResponse)
def observability_health() -> ObservabilityHealthResponse:
    """Return runtime readiness for Phoenix and MLflow integrations."""
    phoenix_configured = is_phoenix_enabled()
    if phoenix_configured and not is_phoenix_initialized():
        init_phoenix()

    mlflow_available = is_mlflow_enabled()
    if mlflow_available:
        init_mlflow_experiment()

    return ObservabilityHealthResponse(
        phoenix_configured=phoenix_configured,
        phoenix_initialized=is_phoenix_initialized(),
        otel_span_mode=get_phoenix_span_mode(),
        mlflow_available=mlflow_available,
        mlflow_tracking_uri=settings.mlflow_tracking_uri.strip() or None,
    )
