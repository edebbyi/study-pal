"""embeddings.py: Embedding helpers and OpenRouter/OpenAI client wrappers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TypeAlias, TypeGuard

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.core.config import settings
from src.core.observability import configure_langfuse_environment, langfuse_enabled, log_langfuse_event


EmbeddingVector: TypeAlias = dict[str, list[float]]
TokenCounts: TypeAlias = Counter[str]
EmbeddingResult: TypeAlias = EmbeddingVector | TokenCounts


@dataclass(frozen=True)
class EmbeddingClient:
    client: OpenAI
    enable_tracing: bool



def _get_embedding_client() -> EmbeddingClient | None:
    """Create an embeddings client when API credentials are available.
    
    Returns:
        EmbeddingClient | None: Result value.
    """

    if not settings.openrouter_api_key:
        return None
    if langfuse_enabled():
        configure_langfuse_environment()
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI

            return EmbeddingClient(
                client=LangfuseOpenAI(
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                ),
                enable_tracing=True,
            )
        except Exception:
            pass
    return EmbeddingClient(
        client=OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        ),
        enable_tracing=False,
    )



def _to_token_counts(text: str) -> TokenCounts:
    """Convert text into a simple token-count vector.
    
    Args:
        text (str): Input text to process.
    
    Returns:
        TokenCounts: Result value.
    """

    return Counter(token.lower() for token in text.split())



def is_embedding_vector(value: EmbeddingResult) -> TypeGuard[EmbeddingVector]:
    """Check if the embedding result is a real vector payload.
    
    Args:
        value (EmbeddingResult): Input parameter.
    
    Returns:
        TypeGuard[EmbeddingVector]: Result value.
    """

    return isinstance(value, dict) and "values" in value



def embed_text(text: str) -> EmbeddingResult:
    """Embed a single text or fall back to token counts.
    
    Args:
        text (str): Input text to process.
    
    Returns:
        EmbeddingResult: Result value.
    """

    embed_client = _get_embedding_client()
    if embed_client is not None:
        if embed_client.enable_tracing:
            log_langfuse_event(
                "embed_text",
                metadata={"text_chars": len(text)},
            )
        try:
            response = embed_client.client.embeddings.create(
                model=settings.embedding_model,
                input=[text],
                dimensions=settings.embedding_dimensions,
            )
            data = getattr(response, "data", None)
            if not data:
                raise ValueError("Embedding response missing data")
            return {"values": data[0].embedding}
        except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, Exception):
            pass
    return _to_token_counts(text)  # fallback to lexical counts when embeddings are unavailable



def embed_texts(texts: list[str]) -> list[EmbeddingResult]:
    """Embed a list of texts or fall back to token counts.
    
    Args:
        texts (list[str]): Input texts to process.
    
    Returns:
        list[EmbeddingResult]: List of results.
    """

    embed_client = _get_embedding_client()
    if embed_client is not None:
        if embed_client.enable_tracing:
            log_langfuse_event(
                "embed_texts",
                metadata={"num_texts": len(texts)},
            )
        try:
            response = embed_client.client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
            )
            data = getattr(response, "data", None)
            if not data:
                raise ValueError("Embedding response missing data")
            return [{"values": item.embedding} for item in data]
        except (APIConnectionError, APIStatusError, APITimeoutError, ValueError, Exception):
            pass
    return [embed_text(text) for text in texts]  # reuse single-text fallback for each item
