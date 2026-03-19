from __future__ import annotations

import re
import uuid
from collections.abc import Iterable


def generate_session_id() -> str:
    return uuid.uuid4().hex[:12]


def generate_document_id() -> str:
    return uuid.uuid4().hex[:10]


def generate_message_id() -> str:
    return uuid.uuid4().hex[:14]


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def humanize_label(text: str) -> str:
    normalized_text = clean_text(re.sub(r"[_-]+", " ", text))
    if not normalized_text:
        return ""
    return normalized_text[0].upper() + normalized_text[1:]
