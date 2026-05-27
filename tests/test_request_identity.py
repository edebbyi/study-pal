"""test_request_identity.py: Identity resolution and conflict checks."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.services.request_identity import resolve_request_user_id


def _request(*, headers: dict[str, str] | None = None) -> Request:
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("utf-8"), value.encode("utf-8")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/test",
        "raw_path": b"/api/test",
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _bearer_token(sub: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8")).decode("utf-8").rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode("utf-8")).decode("utf-8").rstrip("=")
    signature = "sig"
    return f"{header}.{payload}.{signature}"


def test_resolve_identity_prefers_bearer_sub() -> None:
    req = _request(headers={"Authorization": f"Bearer {_bearer_token('user-123')}"})
    resolved = resolve_request_user_id(req)
    assert resolved == "user-123"


def test_resolve_identity_rejects_bearer_conflict() -> None:
    req = _request(headers={"Authorization": f"Bearer {_bearer_token('user-123')}"})
    with pytest.raises(HTTPException) as exc:
        resolve_request_user_id(req, query_user_id="user-other")
    assert exc.value.status_code == 403


def test_resolve_identity_rejects_query_body_without_header_or_token() -> None:
    req = _request()
    with pytest.raises(HTTPException) as exc:
        resolve_request_user_id(req, query_user_id="user-123")
    assert exc.value.status_code == 401
