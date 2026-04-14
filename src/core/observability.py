"""observability.py: Langfuse and OpenLit tracing helpers."""

from __future__ import annotations

import os

from langfuse import get_client

from src.core.config import settings


def langfuse_enabled() -> bool:
    """Check whether Langfuse is configured.

    Returns:
        bool: True when Langfuse keys are present.
    """
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def configure_langfuse_environment() -> bool:
    """Ensure Langfuse environment variables are set.

    Returns:
        bool: True when Langfuse is configured.
    """
    if not langfuse_enabled():
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url
    return True


def build_langfuse_metadata(feature: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build lightweight metadata to tag Langfuse traces.

    Args:
        feature (str): Feature name for the trace.
        extra (dict[str, str] | None): Additional metadata to include.

    Returns:
        dict[str, str]: Metadata payload for Langfuse.
    """
    metadata = {
        "app": "Study Pal",
        "feature": feature,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "openrouter_base_url": settings.openrouter_base_url,
    }
    if extra:
        metadata.update(extra)
    return metadata


def log_langfuse_event(
    name: str,
    *,
    session_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Record a lightweight Langfuse event when tracing is enabled.

    Args:
        name (str): Event name.
        session_id (str | None): Session identifier.
        metadata (dict[str, object] | None): Additional event metadata.
    """
    if not configure_langfuse_environment():
        return

    try:
        langfuse = get_client()
        payload: dict[str, object] = {"name": name}
        if session_id:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata

        if hasattr(langfuse, "create_event"):
            langfuse.create_event(**payload)
            return
        if hasattr(langfuse, "event"):
            langfuse.event(**payload)
            return
        if hasattr(langfuse, "trace"):
            langfuse.trace(**payload)
    except Exception:
        return


def log_langfuse_score(
    *,
    name: str,
    value: int,
    trace_id: str | None = None,
    observation_id: str | None = None,
    session_id: str | None = None,
    comment: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Record a Langfuse score (e.g., user feedback).

    Args:
        name (str): Score name.
        value (int): Score value.
        trace_id (str | None): Langfuse trace identifier.
        observation_id (str | None): Langfuse observation identifier.
        session_id (str | None): Session identifier.
        comment (str | None): Optional score comment.
        metadata (dict[str, object] | None): Additional metadata.
    """
    if not configure_langfuse_environment():
        return

    try:
        langfuse = get_client()
        payload: dict[str, object] = {
            "name": name,
            "value": value,
        }
        if trace_id:
            payload["trace_id"] = trace_id
        if observation_id:
            payload["observation_id"] = observation_id
        if session_id:
            payload["session_id"] = session_id
        if comment:
            payload["comment"] = comment
        if metadata:
            payload["metadata"] = metadata

        if hasattr(langfuse, "score"):
            langfuse.score(**payload)
            return
        if hasattr(langfuse, "create_score"):
            langfuse.create_score(**payload)
            return
    except Exception:
        return


def initialize_observability() -> bool:
    """Initialize Langfuse and OpenLit when configured.

    Returns:
        bool: True when observability is enabled.
    """
    if not configure_langfuse_environment():
        return False

    try:
        langfuse = get_client()
        if not langfuse.auth_check():
            return False
        try:
            import openlit
        except ModuleNotFoundError:
            return False
        openlit.init(disable_batch=True)  # Avoid batching to keep traces near-real-time.
    except Exception:
        return False
    return True
