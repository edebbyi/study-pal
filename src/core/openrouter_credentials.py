"""openrouter_credentials.py: Resolve the active OpenRouter key for runtime calls."""

from __future__ import annotations

from src.core.config import settings


def _session_user_id() -> str | None:
    """Read the authenticated user id from Streamlit session state when available."""
    try:
        import streamlit as st

        user_id = st.session_state.get("user_id")
    except Exception:
        return None
    if not isinstance(user_id, str):
        return None
    cleaned = user_id.strip()
    return cleaned or None


def _session_user_openrouter_key() -> str | None:
    """Read the signed-in user's resolved OpenRouter key from session state."""
    try:
        import streamlit as st

        api_key = st.session_state.get("user_openrouter_api_key")
    except Exception:
        return None
    if not isinstance(api_key, str):
        return None
    cleaned = api_key.strip()
    return cleaned or None


def get_effective_openrouter_api_key() -> str:
    """Return the runtime OpenRouter API key.

    Returns:
        str: Effective API key string, or empty string when unavailable.
    """
    user_id = _session_user_id()
    user_key = _session_user_openrouter_key()
    if user_key:
        return user_key

    if user_id and not settings.openrouter_allow_global_fallback:
        return ""

    return settings.openrouter_api_key.strip()
