"""rate_limit.py: Lightweight in-memory API rate limiting for protection.

Notes:
- This is process-local and best-effort for local/single-process deployments.
- For multi-instance production, replace with shared-store rate limiting (e.g., Redis).
"""

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status

from src.core.config import settings


class InMemoryRateLimiter:
    """Simple fixed-window limiter keyed by bucket + client identity."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, *, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        if limit <= 0 or window_seconds <= 0:
            return True, 0

        now = monotonic()
        cutoff = now - window_seconds

        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0


_limiter = InMemoryRateLimiter()


def _client_identity(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown-client"


def _enforce(request: Request, *, bucket: str, limit: int) -> None:
    allowed, retry_after = _limiter.allow(
        key=f"{bucket}:{_client_identity(request)}",
        limit=limit,
        window_seconds=settings.api_rate_limit_window_seconds,
    )
    if allowed:
        return

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded. Please retry shortly.",
        headers={"Retry-After": str(retry_after)},
    )


def enforce_ask_rate_limit(request: Request) -> None:
    """Apply Ask-the-Book request rate limit."""
    _enforce(request, bucket="ask", limit=settings.api_rate_limit_ask_requests)


def enforce_publishing_rate_limit(request: Request) -> None:
    """Apply Publishing endpoint rate limit."""
    _enforce(request, bucket="publishing", limit=settings.api_rate_limit_publishing_requests)


def enforce_feedback_rate_limit(request: Request) -> None:
    """Apply run feedback endpoint rate limit."""
    _enforce(request, bucket="feedback", limit=settings.api_rate_limit_feedback_requests)
