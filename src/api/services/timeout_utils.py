"""timeout_utils.py: Helpers for enforcing bounded service execution time."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar

from src.api.services.service_errors import ServiceTimeoutError


_T = TypeVar("_T")


def run_with_timeout(
    *,
    operation: str,
    mode: str,
    run_output_type: str,
    timeout_seconds: int | float,
    fn: Callable[..., _T],
    **kwargs: object,
) -> _T:
    """Run a function and stop waiting when the timeout is reached."""
    if timeout_seconds <= 0:
        return fn(**kwargs)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, **kwargs)
    try:
        return future.result(timeout=float(timeout_seconds))
    except FuturesTimeoutError as exc:
        future.cancel()
        raise ServiceTimeoutError(
            operation=operation,
            timeout_seconds=timeout_seconds,
            mode=mode,
            output_type=run_output_type,
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
