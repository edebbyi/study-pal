"""request_identity.py: Request-scoped user identity resolution helpers.

Transitional auth boundary:
- Primary identity source: Authorization Bearer token `sub` claim.
- Secondary identity source: X-User-Id header.
- query/body user_id are treated as untrusted compatibility hints and
  only used for mismatch detection (not identity derivation).
"""

from __future__ import annotations

import base64
import json

from fastapi import HTTPException, Request, status


MAX_USER_ID_LENGTH = 256


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_USER_ID_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is too long.",
        )
    return cleaned


def _extract_bearer_subject(authorization_header: str | None) -> str | None:
    """Extract user id (`sub`) from bearer JWT payload (no signature verification)."""
    raw = (authorization_header or "").strip()
    if not raw:
        return None
    if not raw.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer token format.",
        )
    token = raw[7:].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token format is invalid.",
        )
    try:
        payload_part = parts[1]
        payload_part += "=" * (-len(payload_part) % 4)
        decoded = base64.urlsafe_b64decode(payload_part.encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token payload could not be decoded.",
        ) from exc
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token did not include a valid subject claim.",
        )
    return _normalize(subject)


def resolve_request_user_id(
    request: Request,
    *,
    query_user_id: str | None = None,
    body_user_id: str | None = None,
) -> str | None:
    """Resolve a single request user id and reject identity-source mismatches."""
    token_user_id = _extract_bearer_subject(request.headers.get("authorization"))
    header_user_id = _normalize(request.headers.get("x-user-id"))
    query_value = _normalize(query_user_id)
    body_value = _normalize(body_user_id)

    if token_user_id is not None:
        for candidate in [header_user_id, query_value, body_value]:
            if candidate is not None and candidate != token_user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Provided user_id does not match bearer token identity.",
                )
        return token_user_id

    if header_user_id is not None:
        values = [value for value in [header_user_id, query_value, body_value] if value is not None]
        if len(set(values)) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conflicting user identity values were provided. Use a single consistent user_id source.",
            )
        return header_user_id

    # Query/body user_id are no longer accepted as primary identity sources.
    if query_value is not None or body_value is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provide identity via Authorization bearer token or X-User-Id header.",
        )
    return None
