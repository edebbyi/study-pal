"""test_openrouter_credentials.py: Tests for effective OpenRouter key resolution."""

from __future__ import annotations

from types import SimpleNamespace
import sys

import src.core.openrouter_credentials as credentials


def _set_streamlit_session(monkeypatch, session_state: dict[str, object]) -> None:
    fake_streamlit = SimpleNamespace(session_state=session_state)
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)


def test_effective_key_prefers_signed_in_user_key(monkeypatch) -> None:
    """Use the signed-in user's key when available."""
    _set_streamlit_session(
        monkeypatch,
        {"user_id": "learner-1", "user_openrouter_api_key": "sk-or-v1-user-key"},
    )
    monkeypatch.setattr(
        credentials,
        "settings",
        SimpleNamespace(openrouter_api_key="sk-or-v1-global", openrouter_allow_global_fallback=False),
    )

    assert credentials.get_effective_openrouter_api_key() == "sk-or-v1-user-key"


def test_effective_key_blocks_global_fallback_for_signed_in_user(monkeypatch) -> None:
    """Return empty when user is signed in but has no saved key and fallback is disabled."""
    _set_streamlit_session(
        monkeypatch,
        {"user_id": "learner-1", "user_openrouter_api_key": None},
    )
    monkeypatch.setattr(
        credentials,
        "settings",
        SimpleNamespace(openrouter_api_key="sk-or-v1-global", openrouter_allow_global_fallback=False),
    )

    assert credentials.get_effective_openrouter_api_key() == ""


def test_effective_key_allows_global_fallback_when_enabled(monkeypatch) -> None:
    """Use global key for signed-in users when fallback is explicitly enabled."""
    _set_streamlit_session(
        monkeypatch,
        {"user_id": "learner-1", "user_openrouter_api_key": None},
    )
    monkeypatch.setattr(
        credentials,
        "settings",
        SimpleNamespace(openrouter_api_key="sk-or-v1-global", openrouter_allow_global_fallback=True),
    )

    assert credentials.get_effective_openrouter_api_key() == "sk-or-v1-global"


def test_effective_key_uses_global_key_without_authenticated_user(monkeypatch) -> None:
    """Use global key when there is no signed-in user context."""
    _set_streamlit_session(monkeypatch, {"user_id": None, "user_openrouter_api_key": None})
    monkeypatch.setattr(
        credentials,
        "settings",
        SimpleNamespace(openrouter_api_key="sk-or-v1-global", openrouter_allow_global_fallback=False),
    )

    assert credentials.get_effective_openrouter_api_key() == "sk-or-v1-global"
