"""test_rate_limit.py: Tests for API rate limiting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.services import rate_limit


def _request(*, client_host: str = "127.0.0.1", forwarded_for: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode("utf-8")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents/doc-1/ask",
        "raw_path": b"/api/documents/doc-1/ask",
        "query_string": b"",
        "headers": headers,
        "client": (client_host, 44321),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_enforce_ask_rate_limit_blocks_after_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rate_limit,
        "settings",
        SimpleNamespace(
            api_rate_limit_window_seconds=60,
            api_rate_limit_ask_requests=1,
            api_rate_limit_publishing_requests=1,
            api_rate_limit_feedback_requests=1,
        ),
    )
    monkeypatch.setattr(rate_limit, "_limiter", rate_limit.InMemoryRateLimiter())

    request = _request(client_host="10.0.0.1")
    rate_limit.enforce_ask_rate_limit(request)

    with pytest.raises(HTTPException) as exc:
        rate_limit.enforce_ask_rate_limit(request)
    assert exc.value.status_code == 429


def test_rate_limit_uses_forwarded_for_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rate_limit,
        "settings",
        SimpleNamespace(
            api_rate_limit_window_seconds=60,
            api_rate_limit_ask_requests=1,
            api_rate_limit_publishing_requests=1,
            api_rate_limit_feedback_requests=1,
        ),
    )
    monkeypatch.setattr(rate_limit, "_limiter", rate_limit.InMemoryRateLimiter())

    request_a = _request(client_host="127.0.0.1", forwarded_for="1.2.3.4")
    request_b = _request(client_host="127.0.0.1", forwarded_for="5.6.7.8")

    rate_limit.enforce_ask_rate_limit(request_a)
    rate_limit.enforce_ask_rate_limit(request_b)
