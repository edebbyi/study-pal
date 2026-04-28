"""test_user_openrouter_keys.py: Tests for per-user OpenRouter key persistence helpers."""

from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

import src.auth.user_openrouter_keys as key_store


class _DummyResponse:
    """Minimal urlopen response stub."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


def test_validate_openrouter_api_key_rejects_wrong_prefix() -> None:
    ok, error = key_store.validate_openrouter_api_key("not-an-openrouter-key")

    assert ok is False
    assert error == "OpenRouter keys should start with 'sk-or-v1-'."


def test_validate_openrouter_api_key_rejects_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_unauthorized(*args: object, **kwargs: object) -> None:
        headers: Message[str, str] = Message()
        raise HTTPError(
            url="https://openrouter.ai/api/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=headers,
            fp=BytesIO(b"unauthorized"),
        )

    monkeypatch.setattr(key_store, "urlopen", _raise_unauthorized)

    ok, error = key_store.validate_openrouter_api_key("sk-or-v1-valid-looking-but-bad")

    assert ok is False
    assert error == "OpenRouter rejected this key. Please double-check and try again."


def test_validate_openrouter_api_key_handles_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_network_error(*args: object, **kwargs: object) -> None:
        raise URLError("network down")

    monkeypatch.setattr(key_store, "urlopen", _raise_network_error)

    ok, error = key_store.validate_openrouter_api_key("sk-or-v1-temporary")

    assert ok is False
    assert error == "Could not reach OpenRouter to validate the key. Try again in a moment."


def test_save_user_openrouter_key_stops_when_validation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(key_store, "_storage_enabled", lambda: (True, None))
    monkeypatch.setattr(
        key_store,
        "validate_openrouter_api_key",
        lambda api_key: (False, "OpenRouter rejected this key."),
    )
    monkeypatch.setattr(
        key_store,
        "_connect",
        lambda: (_ for _ in ()).throw(AssertionError("DB should not be reached when key is invalid")),
    )

    ok, error = key_store.save_user_openrouter_key("learner-1", "sk-or-v1-not-accepted")

    assert ok is False
    assert error == "OpenRouter rejected this key."


def test_validate_openrouter_api_key_accepts_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(key_store, "urlopen", lambda *args, **kwargs: _DummyResponse(status=200))

    ok, error = key_store.validate_openrouter_api_key("sk-or-v1-real-looking-key")

    assert ok is True
    assert error is None
