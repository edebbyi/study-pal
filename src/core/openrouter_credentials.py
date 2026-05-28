"""openrouter_credentials.py: Resolve the active OpenRouter key for runtime calls."""

from __future__ import annotations

import sys

from src.auth.user_openrouter_keys import load_user_openrouter_key
from src.core.config import settings


def _session_state_get(session_state: object, key: str) -> object | None:
    """Read a key from dict-like or object-like session state containers."""
    getter = getattr(session_state, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return getattr(session_state, key, None)


def _streamlit_session_value(key: str) -> object | None:
    """Safely read a Streamlit session-state value.

    Prefer real Streamlit run context when available. Fall back to lightweight
    test module replacements used in unit tests.
    """
    streamlit_module = sys.modules.get("streamlit")
    if streamlit_module is None:
        return None

    # Tests may inject a lightweight replacement object as `streamlit`.
    if getattr(streamlit_module, "__name__", "") != "streamlit":
        session_state = getattr(streamlit_module, "session_state", None)
        if session_state is None:
            return None
        return _session_state_get(session_state, key)

    # Real Streamlit module is available; only read session state when the app
    # is running inside a script context.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return None
        session_state = getattr(streamlit_module, "session_state", None)
        if session_state is None:
            return None
        return _session_state_get(session_state, key)
    except Exception:
        return None


def _session_user_id() -> str | None:
    """Read the authenticated user id from Streamlit session state when available."""
    user_id = _streamlit_session_value("user_id")
    if not isinstance(user_id, str):
        return None
    cleaned = user_id.strip()
    return cleaned or None


def _session_user_openrouter_key() -> str | None:
    """Read the signed-in user's resolved OpenRouter key from session state."""
    api_key = _streamlit_session_value("user_openrouter_api_key")
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


def get_openrouter_api_key_for_user(user_id: str | None) -> str:
    """Resolve an OpenRouter key for API/server contexts.

    Priority:
    1) Saved per-user key from persistent storage (when user_id is provided)
    2) Existing runtime resolver fallback (`get_effective_openrouter_api_key`)
    """
    cleaned_user_id = (user_id or "").strip()
    if cleaned_user_id:
        record, _ = load_user_openrouter_key(cleaned_user_id)
        if record and record.api_key.strip():
            return record.api_key.strip()
        if not settings.openrouter_allow_global_fallback:
            return ""
    return get_effective_openrouter_api_key()
