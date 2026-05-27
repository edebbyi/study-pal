"""test_observability_router.py: Tests for observability health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import observability as observability_router


client = TestClient(app)


def test_observability_health_smoke(monkeypatch) -> None:
    """It should return runtime readiness fields without failing."""
    monkeypatch.setattr(observability_router, "is_phoenix_enabled", lambda: True)
    monkeypatch.setattr(observability_router, "is_phoenix_initialized", lambda: True)
    monkeypatch.setattr(observability_router, "is_mlflow_enabled", lambda: True)
    monkeypatch.setattr(observability_router, "init_phoenix", lambda: True)
    monkeypatch.setattr(observability_router, "init_mlflow_experiment", lambda: True)
    monkeypatch.setattr(observability_router, "get_phoenix_span_mode", lambda: "enabled")

    response = client.get("/api/observability/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["phoenix_configured"] is True
    assert payload["phoenix_initialized"] is True
    assert payload["otel_span_mode"] == "enabled"
    assert payload["mlflow_available"] is True
