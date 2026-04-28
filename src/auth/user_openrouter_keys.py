"""user_openrouter_keys.py: Persist encrypted per-user OpenRouter API keys."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings


user_openrouter_keys_table_ddl = """
CREATE TABLE IF NOT EXISTS user_openrouter_keys (
    user_id TEXT PRIMARY KEY,
    encrypted_api_key TEXT NOT NULL,
    key_hint TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class UserOpenRouterKeyRecord:
    """Represent a decrypted per-user OpenRouter key record."""

    api_key: str
    key_hint: str
    updated_at: str


OPENROUTER_KEY_PREFIX = "sk-or-v1-"


def _storage_enabled() -> tuple[bool, str | None]:
    """Validate whether persistent key storage is configured.

    Returns:
        tuple[bool, str | None]: Enabled flag and optional error message.
    """
    if not settings.database_url:
        return False, "Set DATABASE_URL before saving per-user OpenRouter keys."
    if not settings.openrouter_key_encryption_secret.strip():
        return False, "Set OPENROUTER_KEY_ENCRYPTION_SECRET before saving per-user keys."
    return True, None


def _key_hint(api_key: str) -> str:
    """Create a masked display hint for an OpenRouter key."""
    cleaned = api_key.strip()
    if len(cleaned) <= 10:
        return "*" * len(cleaned)
    return f"{cleaned[:7]}...{cleaned[-4:]}"


def _normalize_user_id(user_id: str) -> str:
    """Normalize the user id used for persistence lookups."""
    return user_id.strip().lower()


def _fernet() -> Fernet:
    """Build a Fernet client from the configured encryption secret."""
    secret = settings.openrouter_key_encryption_secret.strip().encode("utf-8")
    derived_key = hashlib.sha256(secret).digest()
    fernet_key = base64.urlsafe_b64encode(derived_key)
    return Fernet(fernet_key)


def _encrypt_api_key(api_key: str) -> str:
    """Encrypt an OpenRouter API key for at-rest storage."""
    return _fernet().encrypt(api_key.strip().encode("utf-8")).decode("utf-8")


def _decrypt_api_key(ciphertext: str) -> str | None:
    """Decrypt an encrypted OpenRouter API key."""
    try:
        plaintext = _fernet().decrypt(ciphertext.encode("utf-8"))
    except (InvalidToken, ValueError):
        return None
    return plaintext.decode("utf-8").strip()


def _connect() -> Any:
    """Open a psycopg connection for key persistence."""
    import psycopg

    return psycopg.connect(settings.database_url)


def validate_openrouter_api_key(api_key: str) -> tuple[bool, str | None]:
    """Validate an OpenRouter key before persisting it.

    Args:
        api_key (str): User-provided OpenRouter key.

    Returns:
        tuple[bool, str | None]: Success flag and optional validation error.
    """
    cleaned_key = api_key.strip()
    if not cleaned_key:
        return False, "Enter an OpenRouter key before saving."
    if not cleaned_key.startswith(OPENROUTER_KEY_PREFIX):
        return False, "OpenRouter keys should start with 'sk-or-v1-'."

    endpoint = settings.openrouter_base_url.rstrip("/") + "/models"
    request = Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {cleaned_key}",
            "HTTP-Referer": "https://study-pal.local",
            "X-Title": "Study Pal",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            status = int(getattr(response, "status", 200))
            if status >= 400:
                return False, f"OpenRouter validation failed with status {status}."
        return True, None
    except HTTPError as exc:
        if exc.code in {401, 403}:
            return False, "OpenRouter rejected this key. Please double-check and try again."
        return False, f"OpenRouter validation failed with status {exc.code}."
    except URLError:
        return False, "Could not reach OpenRouter to validate the key. Try again in a moment."
    except Exception:
        return False, "Could not validate the OpenRouter key right now. Try again."


def save_user_openrouter_key(user_id: str, api_key: str) -> tuple[bool, str | None]:
    """Encrypt and persist an OpenRouter key for a user.

    Args:
        user_id (str): Authenticated user id.
        api_key (str): User-provided OpenRouter API key.

    Returns:
        tuple[bool, str | None]: Success flag and optional error.
    """
    enabled, error = _storage_enabled()
    if not enabled:
        return False, error

    normalized_user_id = _normalize_user_id(user_id)
    cleaned_key = api_key.strip()
    if not normalized_user_id:
        return False, "Cannot save a key without a signed-in user."
    if not cleaned_key:
        return False, "Enter an OpenRouter key before saving."
    key_valid, validation_error = validate_openrouter_api_key(cleaned_key)
    if not key_valid:
        return False, validation_error

    key_hint = _key_hint(cleaned_key)
    key_fingerprint = hashlib.sha256(cleaned_key.encode("utf-8")).hexdigest()
    encrypted_api_key = _encrypt_api_key(cleaned_key)
    now = datetime.now(timezone.utc).isoformat()

    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(user_openrouter_keys_table_ddl)
                cursor.execute(
                    """
                    INSERT INTO user_openrouter_keys (
                        user_id,
                        encrypted_api_key,
                        key_hint,
                        key_fingerprint,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        encrypted_api_key = EXCLUDED.encrypted_api_key,
                        key_hint = EXCLUDED.key_hint,
                        key_fingerprint = EXCLUDED.key_fingerprint,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        normalized_user_id,
                        encrypted_api_key,
                        key_hint,
                        key_fingerprint,
                        now,
                        now,
                    ),
                )
            connection.commit()
        return True, None
    except Exception as exc:
        return False, str(exc)


def load_user_openrouter_key(user_id: str) -> tuple[UserOpenRouterKeyRecord | None, str | None]:
    """Load and decrypt a user's saved OpenRouter key.

    Args:
        user_id (str): Authenticated user id.

    Returns:
        tuple[UserOpenRouterKeyRecord | None, str | None]:
            Decrypted key record and optional error.
    """
    enabled, error = _storage_enabled()
    if not enabled:
        return None, error

    normalized_user_id = _normalize_user_id(user_id)
    if not normalized_user_id:
        return None, "Cannot load a key without a signed-in user."

    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(user_openrouter_keys_table_ddl)
                cursor.execute(
                    """
                    SELECT encrypted_api_key, key_hint, updated_at
                    FROM user_openrouter_keys
                    WHERE user_id = %s
                    """,
                    (normalized_user_id,),
                )
                row = cursor.fetchone()
        if not row:
            return None, None

        encrypted_api_key, key_hint, updated_at = row
        api_key = _decrypt_api_key(str(encrypted_api_key))
        if not api_key:
            return None, "Saved OpenRouter key could not be decrypted. Check encryption secret rotation."

        return (
            UserOpenRouterKeyRecord(
                api_key=api_key,
                key_hint=str(key_hint),
                updated_at=str(updated_at),
            ),
            None,
        )
    except Exception as exc:
        return None, str(exc)


def delete_user_openrouter_key(user_id: str) -> tuple[bool, str | None]:
    """Delete a persisted OpenRouter key for a user.

    Args:
        user_id (str): Authenticated user id.

    Returns:
        tuple[bool, str | None]: Success flag and optional error.
    """
    enabled, error = _storage_enabled()
    if not enabled:
        return False, error

    normalized_user_id = _normalize_user_id(user_id)
    if not normalized_user_id:
        return False, "Cannot delete a key without a signed-in user."

    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(user_openrouter_keys_table_ddl)
                cursor.execute(
                    "DELETE FROM user_openrouter_keys WHERE user_id = %s",
                    (normalized_user_id,),
                )
            connection.commit()
        return True, None
    except Exception as exc:
        return False, str(exc)


def openrouter_key_storage_ready() -> tuple[bool, str | None]:
    """Expose whether key storage prerequisites are configured."""
    return _storage_enabled()
