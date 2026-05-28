"""service_errors.py: Shared service-layer exception types."""

from __future__ import annotations


class ServiceTimeoutError(RuntimeError):
    """Raised when an API/service operation exceeds its configured timeout budget."""

    def __init__(
        self,
        *,
        operation: str,
        timeout_seconds: int | float,
        mode: str,
        output_type: str,
    ) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.mode = mode
        self.output_type = output_type
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s for mode '{mode}' ({output_type})."
        )
