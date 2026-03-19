from __future__ import annotations

from collections import Counter
from typing import TypeAlias, TypeGuard

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import settings


EmbeddingVector: TypeAlias = dict[str, list[float]]
TokenCounts: TypeAlias = Counter[str]
EmbeddingResult: TypeAlias = EmbeddingVector | TokenCounts


def _get_embedding_client() -> OpenAI | None:
    if not settings.openrouter_api_key:
        return None
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _to_token_counts(text: str) -> TokenCounts:
    return Counter(token.lower() for token in text.split())


def is_embedding_vector(value: EmbeddingResult) -> TypeGuard[EmbeddingVector]:
    return isinstance(value, dict) and "values" in value


def embed_text(text: str) -> EmbeddingResult:
    client = _get_embedding_client()
    if client is not None:
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=[text],
                dimensions=settings.embedding_dimensions,
            )
            return {"values": response.data[0].embedding}
        except (APIConnectionError, APIStatusError, APITimeoutError):
            pass
    return _to_token_counts(text)


def embed_texts(texts: list[str]) -> list[EmbeddingResult]:
    client = _get_embedding_client()
    if client is not None:
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
            )
            return [{"values": item.embedding} for item in response.data]
        except (APIConnectionError, APIStatusError, APITimeoutError):
            pass
    return [embed_text(text) for text in texts]
