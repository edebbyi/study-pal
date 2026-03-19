from __future__ import annotations

from src.models import AppMode


mastery_mode_triggers = (
    "help me study",
    "quiz me",
    "generate a quiz",
    "make a quiz",
    "make me a quiz",
    "give me a quiz",
    "test me",
    "master",
    "practice until",
    "study plan",
    "prepare me for",
    "exam prep",
)

topic_prefixes = (
    "help me study",
    "quiz me on",
    "generate a quiz on",
    "generate a quiz",
    "make a quiz on",
    "make a quiz",
    "make me a quiz on",
    "make me a quiz",
    "give me a quiz on",
    "give me a quiz",
    "test me on",
    "prepare me for",
)

ask_topic_prefixes = (
    "what is",
    "what are",
    "how does",
    "how do",
    "tell me about",
    "explain",
    "define",
)

referential_topic_phrases = {
    "this",
    "that",
    "it",
    "please",
    "for me",
    "for me please",
    "this topic",
    "that topic",
    "this concept",
    "that concept",
    "this subject",
    "that subject",
    "this chapter",
    "that chapter",
    "this please",
    "that please",
    "it please",
}

non_topic_tokens = {
    "this",
    "that",
    "it",
    "topic",
    "concept",
    "subject",
    "chapter",
    "please",
    "for",
    "me",
}

trailing_polite_phrases = (
    " please",
    " for me",
    " thanks",
    " thank you",
)

generic_mastery_topic_phrases = {
    "help me study",
    "quiz me",
    "quiz me please",
    "generate a quiz",
    "generate a quiz please",
    "generate a quiz for me",
    "generate a quiz for me please",
    "make a quiz",
    "make a quiz please",
    "make a quiz for me",
    "make a quiz for me please",
    "make a quiz on this topic",
    "make a quiz on this topic please",
    "make me a quiz",
    "make me a quiz please",
    "give me a quiz",
    "give me a quiz please",
    "test me",
    "test me please",
    "please",
    "for me",
    "for me please",
}


def detect_app_mode(user_input: str) -> AppMode:
    normalized_input = user_input.strip().lower()

    # Keep mode routing deterministic for now so future agent behavior starts from a simple contract.
    if any(trigger in normalized_input for trigger in mastery_mode_triggers):
        return "mastery"
    return "ask"


def _clean_topic_candidate(topic: str) -> str:
    cleaned_topic = topic.strip(" :.?!,")

    if cleaned_topic.lower().startswith("on "):
        cleaned_topic = cleaned_topic[3:].strip(" :.?!,")

    while True:
        updated_topic = cleaned_topic
        for phrase in trailing_polite_phrases:
            if updated_topic.lower().endswith(phrase):
                updated_topic = updated_topic[: -len(phrase)].strip(" :.?!,")
        if updated_topic == cleaned_topic:
            break
        cleaned_topic = updated_topic

    return cleaned_topic


def _is_referential_topic_candidate(topic: str) -> bool:
    lowered_topic = topic.lower()
    if lowered_topic in referential_topic_phrases:
        return True

    topic_tokens = lowered_topic.split()
    return bool(topic_tokens) and all(token in non_topic_tokens for token in topic_tokens)


def is_generic_mastery_topic(topic: str | None) -> bool:
    if topic is None:
        return False

    cleaned_topic = _clean_topic_candidate(topic)
    if not cleaned_topic:
        return True

    lowered_topic = cleaned_topic.lower()
    return lowered_topic in generic_mastery_topic_phrases or _is_referential_topic_candidate(cleaned_topic)


def _resolve_topic_candidate(topic: str, fallback_topic: str | None) -> str:
    cleaned_topic = _clean_topic_candidate(topic)
    if not cleaned_topic and fallback_topic:
        return fallback_topic
    if _is_referential_topic_candidate(cleaned_topic) and fallback_topic:
        return fallback_topic
    return cleaned_topic or fallback_topic or topic.strip(" :.?!")


def extract_study_topic(user_input: str, fallback_topic: str | None = None) -> str:
    normalized_input = user_input.strip()
    lowered_input = normalized_input.lower()

    for prefix in topic_prefixes:
        if lowered_input.startswith(prefix):
            topic = normalized_input[len(prefix) :]
            return _resolve_topic_candidate(topic, fallback_topic)

    if "study plan for" in lowered_input:
        _, _, remainder = normalized_input.partition("study plan for")
        return _resolve_topic_candidate(remainder, fallback_topic)

    return _resolve_topic_candidate(normalized_input, fallback_topic)


def extract_conversation_topic(
    user_input: str,
    fallback_topic: str | None = None,
) -> str:
    if detect_app_mode(user_input) == "mastery":
        return extract_study_topic(user_input, fallback_topic)

    normalized_input = user_input.strip()
    lowered_input = normalized_input.lower()

    for prefix in ask_topic_prefixes:
        if lowered_input.startswith(prefix):
            topic = _clean_topic_candidate(normalized_input[len(prefix) :])
            if topic.lower().startswith("the "):
                topic = topic[4:]
            return topic.title() or normalized_input.strip(" :.?!")

    return normalized_input.strip(" :.?!").title()
