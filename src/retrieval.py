from __future__ import annotations

import math
from collections import Counter

from src.config import settings
from src.embeddings import embed_text, is_embedding_vector
from src.models import Chunk, RetrievedChunk
from src.vector_store import normalize_retrieved_chunks, query_remote_chunks


def _tokenize(text: str) -> Counter[str]:
    return Counter(token.lower().strip(".,!?;:()[]{}\"'") for token in text.split() if token.strip())


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve_chunks(
    question: str,
    chunks: list[Chunk],
    session_id: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    query_embedding = embed_text(question)

    # Prefer the remote vector index when we have a real embedding available.
    if is_embedding_vector(query_embedding):
        remote_results = query_remote_chunks(query_embedding, session_id, top_k)
        if remote_results:
            return remote_results

    query_tokens = _tokenize(question)
    session_chunks = [chunk for chunk in chunks if chunk.session_id == session_id]
    scored = [
        (chunk, _cosine_similarity(query_tokens, _tokenize(chunk.text)))
        for chunk in session_chunks
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    scored = [item for item in scored if item[1] > 0][:top_k]
    return normalize_retrieved_chunks(scored)
