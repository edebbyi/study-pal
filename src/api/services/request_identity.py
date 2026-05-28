"""Resolve a request user id from trusted request fields.

Identity priority:
1. Bearer token subject (`sub`) from `Authorization`.
2. `X-User-Id` header when no bearer token is present.

Query/body `user_id` values are not used as the main identity source.
They are only checked for conflicts with trusted identity values.
"""

from __future__ import annotations

import base64
import json

from fastapi import HTTPException, Request, status


MAX_USER_ID_LENGTH = 256


def _normalize(value: str | None) -> str | None:
    """Trim and validate a user id value.

    Returns None for empty values and raises 400 when the value is too long.
    """
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
    """Read user id (`sub`) from a bearer token payload.

    This checks token shape and decodes the payload, but does not verify
    the JWT signature in this helper.
    """
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
    """Return one consistent user id for the request.

    Rules:
    - If a bearer token is present, its `sub` value is the source of truth.
    - If no token is present, `X-User-Id` can be used.
    - Query/body `user_id` values cannot act as primary identity.
    - Conflicting identity values return an HTTP error.
    """
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

    # Query/body values are allowed only as conflict checks, not identity inputs.
    if query_value is not None or body_value is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provide identity via Authorization bearer token or X-User-Id header.",
        )
    return None
