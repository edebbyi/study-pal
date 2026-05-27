"""test_observability_health_modes.py: Coverage for observability health fallback modes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import observability as observability_router


client = TestClient(app)


def test_observability_health_when_disabled(monkeypatch) -> None:
    """Endpoint should stay healthy when Phoenix/MLflow are unavailable."""
    monkeypatch.setattr(observability_router, "is_phoenix_enabled", lambda: False)
    monkeypatch.setattr(observability_router, "is_phoenix_initialized", lambda: False)
    monkeypatch.setattr(observability_router, "is_mlflow_enabled", lambda: False)
    monkeypatch.setattr(observability_router, "get_phoenix_span_mode", lambda: "disabled")

    response = client.get("/api/observability/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["phoenix_configured"] is False
    assert payload["phoenix_initialized"] is False
    assert payload["otel_span_mode"] == "disabled"
    assert payload["mlflow_available"] is False
