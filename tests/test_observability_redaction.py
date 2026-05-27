"""test_observability_redaction.py: Secret redaction coverage for observability payloads."""

from __future__ import annotations

from src.api.services.observability_service import sanitize_observability_payload


def test_sanitize_observability_payload_redacts_sensitive_keys() -> None:
    payload = {
        "api_key": "sk-or-v1-abcdefghijklmnop",
        "nested": {
            "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            "safe": "hello",
        },
    }

    sanitized = sanitize_observability_payload(payload)

    assert isinstance(sanitized, dict)
    assert sanitized["api_key"] == "[REDACTED]"
    assert isinstance(sanitized["nested"], dict)
    assert sanitized["nested"]["Authorization"] == "[REDACTED]"
    assert sanitized["nested"]["safe"] == "hello"


def test_sanitize_observability_payload_redacts_secret_like_values() -> None:
    payload = {
        "note": "use key sk-or-v1-abcde12345secretvalue for testing",
        "tokenized": "Bearer supersecrettokenvalue12345",
        "list": ["sk-abcdef1234567890", "normal"],
    }

    sanitized = sanitize_observability_payload(payload)

    assert isinstance(sanitized, dict)
    assert "[REDACTED]" in str(sanitized["note"])
    assert "[REDACTED]" in str(sanitized["tokenized"])
    assert isinstance(sanitized["list"], list)
    assert sanitized["list"][0] == "[REDACTED]"
    assert sanitized["list"][1] == "normal"
