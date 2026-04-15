"""supabase_auth.py: Supabase magic-link/OTP helpers for Streamlit auth."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Mapping

from supabase import Client, create_client

from src.core.config import settings

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@lru_cache(maxsize=1)
def _get_client() -> Client | None:
    """Create a Supabase client when configuration is present.

    Returns:
        Client | None: Supabase client or None when not configured.
    """
    if not settings.supabase_url or not settings.supabase_public_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_public_key)


def supabase_enabled() -> bool:
    """Check whether Supabase auth is configured.

    Returns:
        bool: True when Supabase credentials are present.
    """
    return bool(settings.supabase_url and settings.supabase_public_key)


def is_valid_email_address(email: str) -> bool:
    """Validate whether an email has a minimal address shape.

    Args:
        email (str): User-supplied email input.

    Returns:
        bool: True when the value resembles a valid email address.
    """
    return bool(_EMAIL_PATTERN.match(email.strip()))


def send_magic_link(email: str) -> tuple[bool, str | None]:
    """Send a magic-link/OTP email to the user.

    Args:
        email (str): User email address.

    Returns:
        tuple[bool, str | None]: Success flag and optional error.
    """
    client = _get_client()
    if client is None:
        return False, "Supabase auth is not configured."

    try:
        payload: dict[str, Any] = {"email": email}
        if settings.supabase_redirect_url:
            payload["options"] = {"email_redirect_to": settings.supabase_redirect_url}
        client.auth.sign_in_with_otp(payload)
        return True, None
    except Exception as exc:  # pragma: no cover - depends on external API
        return False, str(exc)


def complete_sign_in_from_callback(
    query_params: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Complete a sign-in flow from URL callback parameters.

    Args:
        query_params (Mapping[str, Any]): Current URL query parameters.

    Returns:
        tuple[dict[str, Any] | None, str | None, bool]:
            User payload, optional error, and whether callback params were handled.
    """
    client = _get_client()
    if client is None:
        return None, "Supabase auth is not configured.", False

    def _first_param(name: str) -> str:
        """Normalize Streamlit query params values to a single string."""
        raw_value = query_params.get(name, "")
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else ""
        return str(raw_value or "").strip()

    code = _first_param("code")
    code_verifier = _first_param("code_verifier")
    token_hash = _first_param("token_hash")
    token = _first_param("token")
    token_type = _first_param("type")
    email = _first_param("email")
    callback_error = _first_param("error")
    callback_error_description = _first_param("error_description")
    if callback_error or callback_error_description:
        message = callback_error_description or callback_error or "Supabase sign-in callback returned an error."
        return None, message, True
    if not code and not token_hash:
        # Some providers may use `token` instead of `token_hash`.
        if not token:
            return None, None, False

    try:
        response: Any
        if code:
            exchange_payload: dict[str, str] = {"auth_code": code}
            if code_verifier:
                exchange_payload["code_verifier"] = code_verifier
            response = client.auth.exchange_code_for_session(exchange_payload)
        elif token_hash:
            response = client.auth.verify_otp({"token_hash": token_hash, "type": token_type or "magiclink"})
        elif email and token:
            response = client.auth.verify_otp({"email": email, "token": token, "type": token_type or "email"})
        else:
            return None, "Auth callback is missing required token fields.", True

        if hasattr(response, "user") and response.user is not None:
            user_obj = response.user
            if hasattr(user_obj, "model_dump"):
                return user_obj.model_dump(), None, True
            return dict(user_obj), None, True

        payload = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(payload, dict):
            user_payload = payload.get("user")
            if isinstance(user_payload, dict):
                return user_payload, None, True
        return None, "Sign-in callback was handled but no user was returned.", True
    except Exception as exc:  # pragma: no cover - depends on external API
        return None, str(exc), True
