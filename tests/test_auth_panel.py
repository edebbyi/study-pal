"""test_auth_panel.py: Focused tests for auth gate rendering behavior."""

from __future__ import annotations

from typing import Literal

import app
import src.core.app_state as app_state_module


class _SessionState(dict):
    """Tiny session-state helper that supports attribute access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


class _DummyContext:
    """Simple context manager used to mock Streamlit layout containers."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        return False


def _prepare_streamlit_mocks(monkeypatch) -> _SessionState:
    """Create a predictable mocked Streamlit surface for auth-panel tests."""
    state = _SessionState()
    query_params: dict[str, str] = {}
    
    def _mock_columns(spec, *args, **kwargs):
        """Return a tuple of dummy contexts sized to the requested layout spec."""
        column_count = spec if isinstance(spec, int) else len(spec)
        return tuple(_DummyContext() for _ in range(column_count))

    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(app_state_module.st, "session_state", state, raising=False)
    app_state_module.initialize_session_state()

    monkeypatch.setattr(app.st, "query_params", query_params, raising=False)
    monkeypatch.setattr(app.st, "sidebar", _DummyContext(), raising=False)
    monkeypatch.setattr(app.st, "columns", _mock_columns, raising=False)
    monkeypatch.setattr(app.st, "text_input", lambda *args, **kwargs: kwargs.get("value", ""), raising=False)
    monkeypatch.setattr(app.st, "button", lambda *args, **kwargs: False, raising=False)
    monkeypatch.setattr(app.st, "markdown", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app.st, "success", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app.st, "error", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app.st, "info", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app.st, "rerun", lambda: None, raising=False)
    monkeypatch.setattr(app, "log_langfuse_event", lambda *args, **kwargs: None)
    return state


def test_auth_panel_bypasses_when_supabase_disabled(monkeypatch) -> None:
    """Auth panel should allow app boot when Supabase auth is not configured."""
    _prepare_streamlit_mocks(monkeypatch)
    monkeypatch.setattr(app, "supabase_enabled", lambda: False)

    assert app._render_auth_panel() is True


def test_auth_panel_returns_true_when_user_already_authenticated(monkeypatch) -> None:
    """Auth panel should pass through when a user is already signed in."""
    state = _prepare_streamlit_mocks(monkeypatch)
    state.user_id = "user-1"
    state.user_email = "learner@example.com"
    monkeypatch.setattr(app, "supabase_enabled", lambda: True)
    monkeypatch.setattr(app, "complete_sign_in_from_callback", lambda params: (None, None, False))

    assert app._render_auth_panel() is True


def test_auth_panel_returns_false_and_shows_gate_when_logged_out(monkeypatch) -> None:
    """Auth panel should render the gate and block app content when logged out."""
    state = _prepare_streamlit_mocks(monkeypatch)
    state.user_id = None
    state.user_email = None
    monkeypatch.setattr(app, "supabase_enabled", lambda: True)
    monkeypatch.setattr(app, "complete_sign_in_from_callback", lambda params: (None, None, False))

    assert app._render_auth_panel() is False


def test_auth_panel_shows_inline_error_for_invalid_email(monkeypatch) -> None:
    """Auth panel should show a validation error when the email format is invalid."""
    state = _prepare_streamlit_mocks(monkeypatch)
    state.user_id = None
    state.user_email = None
    errors: list[str] = []
    button_calls: list[dict[str, object]] = []

    monkeypatch.setattr(app, "supabase_enabled", lambda: True)
    monkeypatch.setattr(app, "complete_sign_in_from_callback", lambda params: (None, None, False))
    monkeypatch.setattr(app.st, "text_input", lambda *args, **kwargs: "invalid-email", raising=False)
    monkeypatch.setattr(app.st, "error", lambda message, *args, **kwargs: errors.append(str(message)), raising=False)

    def _button_capture(*args, **kwargs):
        button_calls.append(dict(kwargs))
        return False

    monkeypatch.setattr(app.st, "button", _button_capture, raising=False)

    assert app._render_auth_panel() is False
    assert "Enter a valid email address, like name@example.com." in errors
    send_button = next(call for call in button_calls if call.get("key") == "auth_send_link")
    assert send_button.get("disabled") is True
