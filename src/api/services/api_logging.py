"""api_logging.py: Lightweight structured logging for API/service boundaries."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("studypal.api")


def _normalize_value(value: Any) -> str:
    """Normalize log field values into compact safe strings."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if not text:
        return "empty"
    if len(text) > 140:
        return text[:140].rstrip() + "..."
    return text.replace("\n", " ")


def log_api_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured API event with stable correlation fields."""
    normalized = {key: _normalize_value(value) for key, value in fields.items()}
    structured_fields = " ".join(f"{key}={normalized[key]}" for key in sorted(normalized))
    logger.log(
        level,
        "%s | %s",
        event,
        structured_fields,
        extra={"event": event, "correlation": normalized},
    )
