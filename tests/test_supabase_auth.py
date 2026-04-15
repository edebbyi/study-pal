"""test_supabase_auth.py: Focused tests for Supabase magic-link helpers."""

from __future__ import annotations

from types import SimpleNamespace

import src.auth.supabase_auth as supabase_auth


class _FakeUser:
    """Simple fake user object with model_dump support."""

    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, str]:
        """Return the serialized fake user payload."""
        return dict(self._payload)


class _FakeResponse:
    """Simple fake auth response object."""

    def __init__(self, user_payload: dict[str, str]) -> None:
        self.user = _FakeUser(user_payload)


class _FakeAuth:
    """Fake Supabase auth API used for deterministic tests."""

    def __init__(self) -> None:
        self.sign_in_payload: dict[str, object] | None = None
        self.verify_payload: dict[str, object] | None = None
        self.exchange_payload: dict[str, object] | None = None

    def sign_in_with_otp(self, payload: dict[str, object]) -> None:
        """Record the sign-in payload."""
        self.sign_in_payload = payload

    def verify_otp(self, payload: dict[str, object]) -> _FakeResponse:
        """Record verify payload and return a fake user."""
        self.verify_payload = payload
        return _FakeResponse({"id": "user-1", "email": "learner@example.com"})

    def exchange_code_for_session(self, payload: dict[str, object]) -> _FakeResponse:
        """Record exchange payload and return a fake user."""
        self.exchange_payload = payload
        return _FakeResponse({"id": "user-2", "email": "coder@example.com"})


class _FakeClient:
    """Wrapper that mimics the Supabase client shape."""

    def __init__(self) -> None:
        self.auth = _FakeAuth()


def test_send_magic_link_includes_redirect_url_when_configured(monkeypatch) -> None:
    """Ensure magic-link send passes email redirect options when configured."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        supabase_auth,
        "settings",
        SimpleNamespace(supabase_redirect_url="http://localhost:8501"),
    )

    ok, error = supabase_auth.send_magic_link("learner@example.com")

    assert ok is True
    assert error is None
    assert fake_client.auth.sign_in_payload == {
        "email": "learner@example.com",
        "options": {"email_redirect_to": "http://localhost:8501"},
    }


def test_send_magic_link_without_redirect_uses_email_only(monkeypatch) -> None:
    """Ensure magic-link send stays minimal when no redirect URL is configured."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)
    monkeypatch.setattr(supabase_auth, "settings", SimpleNamespace(supabase_redirect_url=""))

    ok, error = supabase_auth.send_magic_link("learner@example.com")

    assert ok is True
    assert error is None
    assert fake_client.auth.sign_in_payload == {"email": "learner@example.com"}


def test_complete_sign_in_handles_token_hash_list_query_params(monkeypatch) -> None:
    """Ensure callback parser accepts list-valued query params from Streamlit."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)

    user, error, handled = supabase_auth.complete_sign_in_from_callback(
        {"token_hash": ["abc123"], "type": ["magiclink"], "email": ["learner@example.com"]}
    )

    assert handled is True
    assert error is None
    assert user == {"id": "user-1", "email": "learner@example.com"}
    assert fake_client.auth.verify_payload == {"token_hash": "abc123", "type": "magiclink"}


def test_complete_sign_in_handles_code_exchange_with_verifier(monkeypatch) -> None:
    """Ensure PKCE callback exchange forwards both code and verifier when present."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)

    user, error, handled = supabase_auth.complete_sign_in_from_callback(
        {"code": "auth-code-1", "code_verifier": "verifier-1"}
    )

    assert handled is True
    assert error is None
    assert user == {"id": "user-2", "email": "coder@example.com"}
    assert fake_client.auth.exchange_payload == {
        "auth_code": "auth-code-1",
        "code_verifier": "verifier-1",
    }


def test_complete_sign_in_surfaces_callback_error_without_api_calls(monkeypatch) -> None:
    """Ensure callback errors are surfaced immediately and safely."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)

    user, error, handled = supabase_auth.complete_sign_in_from_callback(
        {"error": "access_denied", "error_description": "User denied access"}
    )

    assert handled is True
    assert user is None
    assert error == "User denied access"
    assert fake_client.auth.verify_payload is None
    assert fake_client.auth.exchange_payload is None


def test_complete_sign_in_returns_unhandled_when_no_callback_params(monkeypatch) -> None:
    """Ensure non-callback URLs do not trigger auth behavior."""
    fake_client = _FakeClient()
    monkeypatch.setattr(supabase_auth, "_get_client", lambda: fake_client)

    user, error, handled = supabase_auth.complete_sign_in_from_callback({})

    assert handled is False
    assert user is None
    assert error is None


def test_is_valid_email_address_accepts_basic_valid_shape() -> None:
    """Ensure the local validator accepts common valid email formats."""
    assert supabase_auth.is_valid_email_address("learner@example.com") is True
    assert supabase_auth.is_valid_email_address("name+tag@school.edu") is True


def test_is_valid_email_address_rejects_invalid_values() -> None:
    """Ensure the local validator rejects malformed email input."""
    assert supabase_auth.is_valid_email_address("invalid") is False
    assert supabase_auth.is_valid_email_address("missing-at.example.com") is False
    assert supabase_auth.is_valid_email_address("missing-domain@") is False
