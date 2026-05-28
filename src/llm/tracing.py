"""tracing.py: LLM generation tracing helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, cast

from langfuse import get_client

from src.core.config import settings
from src.core.observability import build_langfuse_metadata, configure_langfuse_environment
from src.llm.prompts import PromptBundle


def langfuse_generation(
    *,
    enabled: bool,
    feature: str,
    prompt_bundle: PromptBundle,
    metadata: dict[str, str],
) -> ContextManager[Any]:
    """Create a Langfuse generation context manager when tracing is enabled."""
    if not enabled:
        return nullcontext()

    if not configure_langfuse_environment():
        return nullcontext()

    try:
        langfuse = cast(Any, get_client())
        return langfuse.start_as_current_observation(
            name=feature,
            as_type="generation",
            prompt=prompt_bundle.prompt,  # preserves prompt linking/versioning in Langfuse
            input=prompt_bundle.text,
            metadata=build_langfuse_metadata(feature, metadata),
            model=settings.chat_model,
        )
    except Exception:
        return nullcontext()


def update_generation(generation: object, output_text: str, response: object) -> None:
    """Update a Langfuse generation with output and token usage details."""
    if not hasattr(generation, "update"):
        return

    usage = getattr(response, "usage", None)
    usage_details = None
    if usage is not None:
        usage_details = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    try:
        generation.update(output=output_text, usage_details=usage_details)
    except Exception:
        return


def extract_generation_ids(generation: object) -> tuple[str | None, str | None]:
    """Return `(trace_id, observation_id)` from a Langfuse generation object."""
    if generation is None:
        return None, None
    trace_id = getattr(generation, "trace_id", None) or getattr(generation, "traceId", None)
    observation_id = (
        getattr(generation, "id", None)
        or getattr(generation, "observation_id", None)
        or getattr(generation, "observationId", None)
    )
    return trace_id, observation_id
