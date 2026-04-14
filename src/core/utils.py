"""utils.py: Small helpers for IDs and text normalization."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable


def generate_session_id() -> str:
    """Generate a short session identifier.

    Returns:
        str: Short session ID.
    """
    return uuid.uuid4().hex[:12]


def generate_document_id() -> str:
    """Generate a short document identifier.

    Returns:
        str: Short document ID.
    """
    return uuid.uuid4().hex[:10]


def generate_message_id() -> str:
    """Generate a short message identifier.

    Returns:
        str: Short message ID.
    """
    return uuid.uuid4().hex[:14]


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order.

    Args:
        items (Iterable[str]): Items to dedupe.

    Returns:
        list[str]: Deduped list in original order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def clean_text(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with single spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def humanize_label(text: str) -> str:
    """Make labels more readable for display.

    Args:
        text (str): Raw label text.

    Returns:
        str: Human-friendly label.
    """
    normalized_text = clean_text(re.sub(r"[_-]+", " ", text))
    if not normalized_text:
        return ""
    return normalized_text[0].upper() + normalized_text[1:]
