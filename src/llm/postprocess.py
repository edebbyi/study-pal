"""postprocess.py: Post-processing helpers for LLM answer payloads."""

from __future__ import annotations

import re

from src.core.models import InfoLane, QuizLane, StructuredAnswer


def truncate_sentences(text: str, max_sentences: int) -> str:
    """Trim text to at most ``max_sentences`` sentence boundaries."""
    if max_sentences <= 0:
        return ""
    normalized = " ".join(text.split())
    if not normalized:
        return normalized
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    if len(sentences) <= max_sentences:
        return normalized
    return " ".join(sentences[:max_sentences]).strip()


def _ensure_info_lane_emoji(label: str, fallback_emoji: str = "🧠") -> str:
    """Ensure info-lane labels begin with an emoji for quick UI affordance."""
    stripped = label.strip()
    if not stripped:
        return f"{fallback_emoji} Learn more"
    first_char = stripped[0]
    if first_char.isascii():
        return f"{fallback_emoji} {stripped}"
    return stripped


def _derive_subject(question: str, structured_topic_subject: str | None) -> str:
    """Derive a compact subject label for default action lanes."""
    if structured_topic_subject and structured_topic_subject.strip():
        return structured_topic_subject.strip()

    raw = question.strip().rstrip("?.! ")
    raw = re.sub(
        r"^(what\s+is|what\s+are|who\s+is|define|explain|tell\s+me\s+about|how\s+does|how\s+do)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return "this topic"
    return raw[:60]


def ensure_action_lanes(structured: StructuredAnswer, question: str) -> None:
    """Ensure every structured answer has usable info and quiz lanes."""
    subject = _derive_subject(question, structured.topic_subject)

    if structured.info_lane is None:
        structured.info_lane = InfoLane(
            button_label=f"🧠 Why is {subject} important?",
            query=f"Explain {subject} in more detail from the notes.",
        )
    else:
        if not structured.info_lane.button_label.strip():
            structured.info_lane.button_label = f"🧠 Why is {subject} important?"
        if not structured.info_lane.query.strip():
            structured.info_lane.query = f"Explain {subject} in more detail from the notes."
    structured.info_lane.button_label = _ensure_info_lane_emoji(structured.info_lane.button_label)

    if structured.quiz_lane is None:
        structured.quiz_lane = QuizLane(button_label=f"Test your {subject} knowledge")
    elif not structured.quiz_lane.button_label.strip():
        structured.quiz_lane.button_label = f"Test your {subject} knowledge"


def strip_inline_citations(text: str) -> str:
    """Remove inline citation markers from answer text for UI display."""
    if not text:
        return text
    cleaned = re.sub(
        r"\[[^\]]*(?:\.pdf|page|p\.|chapter|notes|doc)[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip()
