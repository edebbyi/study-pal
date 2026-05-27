"""observability_service.py: Defensive Phoenix + MLflow observability helpers.

These helpers are intentionally no-fail:
- Missing config returns no-op behavior.
- Import/runtime failures are logged as warnings.
- User-facing API handlers should continue even when observability fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.config import settings


logger = logging.getLogger(__name__)
_phoenix_initialized = False
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "bearer",
    "private_key",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{10,}", re.IGNORECASE),
)


@dataclass(frozen=True)
class PhoenixTraceHandle:
    """Handle for one Phoenix trace/root span context."""

    trace_id: str
    span_name: str
    tracer: Any | None = None
    root_span: Any | None = None


@dataclass(frozen=True)
class MlflowRunHandle:
    """Minimal handle for one MLflow run context."""

    run_id: str


def _looks_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(fragment in normalized for fragment in _SENSITIVE_KEYWORDS)


def _redact_string(value: str) -> str:
    cleaned = value
    for pattern in _SECRET_VALUE_PATTERNS:
        cleaned = pattern.sub(_REDACTED, cleaned)
    return cleaned


def sanitize_observability_payload(value: object) -> object:
    """Redact secrets/tokens from arbitrary payloads recursively."""
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, raw in value.items():
            key_text = str(key)
            if _looks_sensitive_key(key_text):
                sanitized[key_text] = _REDACTED
                continue
            sanitized[key_text] = sanitize_observability_payload(raw)
        return sanitized
    if isinstance(value, list):
        return [sanitize_observability_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_observability_payload(item) for item in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value


def is_phoenix_enabled() -> bool:
    """Return True when Phoenix collector config exists."""
    return bool(settings.phoenix_collector_endpoint.strip())


def is_phoenix_initialized() -> bool:
    """Return True when Phoenix instrumentation init succeeded in this process."""
    return _phoenix_initialized


def get_phoenix_span_mode() -> str:
    """Return current Phoenix span mode: enabled, fallback, or disabled."""
    if not is_phoenix_enabled():
        return "disabled"
    if not is_phoenix_initialized():
        return "fallback"
    try:
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer("studypal.observability.health")
        span = tracer.start_span("phoenix.healthcheck")
        span.end()
        return "enabled"
    except Exception:
        return "fallback"


def is_mlflow_enabled() -> bool:
    """Return True when MLflow package is importable.

    Tracking URI may still be blank; in that case local file-based tracking is used.
    """
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:
        return False


def init_phoenix() -> bool:
    """Initialize Phoenix/OpenInference instrumentation when available."""
    global _phoenix_initialized
    if _phoenix_initialized:
        return True

    if not is_phoenix_enabled():
        logger.warning("Phoenix disabled: PHOENIX_COLLECTOR_ENDPOINT is not configured.")
        return False

    try:
        # Keep config in env for libs that read at import-time.
        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", settings.phoenix_collector_endpoint.strip())
        os.environ.setdefault("PHOENIX_PROJECT_NAME", settings.phoenix_project_name.strip())
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", settings.phoenix_collector_endpoint.strip())
        if settings.phoenix_api_key.strip():
            os.environ.setdefault("PHOENIX_API_KEY", settings.phoenix_api_key.strip())
            os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"api_key={settings.phoenix_api_key.strip()}")

        # Optional OpenAI instrumentation (best-effort).
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
        except Exception as instrument_error:
            logger.warning("Phoenix OpenInference instrumentation unavailable: %s", instrument_error)
        _phoenix_initialized = True
        return True
    except Exception as exc:
        logger.warning("Phoenix initialization skipped due to error: %s", exc)
        return False


def init_mlflow_experiment() -> bool:
    """Initialize MLflow tracking URI + experiment with graceful fallback."""
    if not is_mlflow_enabled():
        logger.warning("MLflow is not installed; skipping experiment initialization.")
        return False

    try:
        import mlflow

        tracking_uri = settings.mlflow_tracking_uri.strip()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Local safe default for developer environments.
            local_store = Path(".studypal_cache") / "mlruns"
            local_store.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(local_store.resolve().as_uri())

        mlflow.set_experiment(settings.mlflow_experiment_name.strip() or "StudyPal Publishing Mode")
        return True
    except Exception as exc:
        logger.warning("MLflow initialization skipped due to error: %s", exc)
        return False


def start_phoenix_trace(*, span_name: str, metadata: dict[str, object] | None = None) -> PhoenixTraceHandle | None:
    """Start a Phoenix trace handle (observable-only metadata, no secrets)."""
    if not is_phoenix_enabled():
        return None
    if not init_phoenix():
        return None

    try:
        trace_id = str(uuid4())
        tracer: Any | None = None
        root_span: Any | None = None
        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("studypal.publishing")
            root_span = tracer.start_span(span_name)
            root_span.set_attribute("studypal.trace_id", trace_id)
            if metadata:
                for key, value in _phoenix_attributes(metadata).items():
                    root_span.set_attribute(f"studypal.meta.{key}", value)
        except Exception as span_error:
            logger.warning("Phoenix root span unavailable; falling back to lightweight handle: %s", span_error)
        if metadata:
            logger.info("Phoenix trace started: %s (%s)", span_name, trace_id)
        return PhoenixTraceHandle(trace_id=trace_id, span_name=span_name, tracer=tracer, root_span=root_span)
    except Exception as exc:
        logger.warning("Phoenix trace start failed: %s", exc)
        return None


def _phoenix_attributes(payload: dict[str, object]) -> dict[str, str | int | float | bool]:
    """Normalize payload into span-safe attributes."""
    safe_payload = sanitize_observability_payload(payload)
    if not isinstance(safe_payload, dict):
        return {}
    attributes: dict[str, str | int | float | bool] = {}
    for key, value in safe_payload.items():
        if isinstance(value, bool):
            attributes[key] = value
        elif isinstance(value, int):
            attributes[key] = value
        elif isinstance(value, float):
            attributes[key] = value
        elif isinstance(value, str):
            attributes[key] = value[:1000]
        elif value is None:
            continue
        else:
            try:
                compact = json.dumps(value, ensure_ascii=True, default=str)
                attributes[key] = compact[:2000]
            except Exception:
                attributes[key] = str(value)[:1000]
    return attributes


def _phoenix_child_span(trace: PhoenixTraceHandle | None, name: str, payload: dict[str, object]) -> None:
    """Create a short-lived child span under the root trace when possible."""
    if trace is None:
        return
    try:
        safe_payload = sanitize_observability_payload(payload)
        if not isinstance(safe_payload, dict):
            safe_payload = {"payload": str(safe_payload)}
        if trace.tracer is None or trace.root_span is None:
            logger.debug("Phoenix %s trace=%s keys=%s", name, trace.trace_id, sorted(safe_payload.keys()))
            return
        from opentelemetry import trace as otel_trace

        parent_context = otel_trace.set_span_in_context(trace.root_span)
        child = trace.tracer.start_span(f"{trace.span_name}.{name}", context=parent_context)
        child.set_attribute("studypal.trace_id", trace.trace_id)
        for key, value in _phoenix_attributes(safe_payload).items():
            child.set_attribute(f"studypal.{name}.{key}", value)
        child.end()
    except Exception as exc:
        logger.warning("Phoenix %s span logging failed: %s", name, exc)


def log_phoenix_retrieval(trace: PhoenixTraceHandle | None, payload: dict[str, object]) -> None:
    """Log retrieval payload for Phoenix trace (best-effort)."""
    _phoenix_child_span(trace, "retrieval", payload)


def log_phoenix_generation(trace: PhoenixTraceHandle | None, payload: dict[str, object]) -> None:
    """Log generation payload for Phoenix trace (best-effort)."""
    _phoenix_child_span(trace, "generation", payload)


def log_phoenix_scores(trace: PhoenixTraceHandle | None, payload: dict[str, object]) -> None:
    """Log score/eval payload for Phoenix trace (best-effort)."""
    _phoenix_child_span(trace, "scores", payload)


def end_phoenix_trace(trace: PhoenixTraceHandle | None) -> None:
    """End Phoenix trace (best-effort no-op placeholder)."""
    if trace is None:
        return
    try:
        if trace.root_span is not None:
            trace.root_span.end()
        logger.debug("Phoenix trace ended: %s", trace.trace_id)
    except Exception as exc:
        logger.warning("Phoenix trace end failed: %s", exc)


def start_mlflow_run(*, run_name: str, tags: dict[str, str] | None = None) -> MlflowRunHandle | None:
    """Start MLflow run safely.

    Returns None when MLflow unavailable/unconfigured.
    """
    if not init_mlflow_experiment():
        return None

    try:
        import mlflow

        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        return MlflowRunHandle(run_id=str(run.info.run_id))
    except Exception as exc:
        logger.warning("MLflow run start failed: %s", exc)
        return None


def log_mlflow_params(run: MlflowRunHandle | None, params: dict[str, object]) -> None:
    """Log MLflow params safely."""
    if run is None:
        return
    try:
        import mlflow

        sanitized = sanitize_observability_payload(params)
        if not isinstance(sanitized, dict):
            sanitized = {}
        safe_params = {k: str(v)[:500] for k, v in sanitized.items() if v is not None}
        mlflow.log_params(safe_params)
    except Exception as exc:
        logger.warning("MLflow param logging failed: %s", exc)


def log_mlflow_metrics(run: MlflowRunHandle | None, metrics: dict[str, float | int | None]) -> None:
    """Log MLflow metrics safely."""
    if run is None:
        return
    try:
        import mlflow

        safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if safe_metrics:
            mlflow.log_metrics(safe_metrics)
    except Exception as exc:
        logger.warning("MLflow metric logging failed: %s", exc)


def log_mlflow_artifacts(run: MlflowRunHandle | None, artifacts: dict[str, Any]) -> None:
    """Log small JSON artifacts to MLflow safely.

    Artifacts are serialized to temporary files and uploaded individually.
    """
    if run is None:
        return
    try:
        import mlflow

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, payload in artifacts.items():
                filename = f"{name}.json" if not str(name).endswith(".json") else str(name)
                path = Path(tmp_dir) / filename
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        sanitize_observability_payload(payload),
                        handle,
                        ensure_ascii=True,
                        indent=2,
                        default=str,
                    )
                mlflow.log_artifact(str(path))
    except Exception as exc:
        logger.warning("MLflow artifact logging failed: %s", exc)


def end_mlflow_run(run: MlflowRunHandle | None, status: str = "FINISHED") -> None:
    """End MLflow run safely."""
    if run is None:
        return
    try:
        import mlflow

        mlflow.end_run(status=status)
    except Exception as exc:
        logger.warning("MLflow run end failed: %s", exc)
