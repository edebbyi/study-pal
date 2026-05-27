"""retrieval.py: Retrieve, merge, and rerank note chunks for RAG."""

from __future__ import annotations

import math
from collections import Counter
import json
from urllib import request, error

from src.core.config import settings
from src.core.openrouter_credentials import get_effective_openrouter_api_key
from src.core.observability import log_langfuse_event
from src.data.embeddings import embed_text, is_embedding_vector
from src.core.models import Chunk, RetrievedChunk
from src.data.vector_store import normalize_retrieved_chunks, query_remote_chunks



def _normalize_token(token: str) -> str:
    """Normalize token.
    
    Args:
        token (str): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    cleaned = token.lower().strip(".,!?;:()[]{}\"'")
    if cleaned.endswith("ies") and len(cleaned) > 4:
        return cleaned[:-3] + "y"
    if cleaned.endswith("es") and len(cleaned) > 4:
        return cleaned[:-2]
    if cleaned.endswith("s") and len(cleaned) > 3:
        return cleaned[:-1]
    return cleaned



def _tokenize(text: str) -> Counter[str]:
    """Tokenize text into a simple frequency counter.
    
    Args:
        text (str): Input text to process.
    
    Returns:
        Counter[str]: Result value.
    """

    tokens = []
    for raw_token in text.replace("-", " ").split():
        normalized = _normalize_token(raw_token)
        if normalized:
            tokens.append(normalized)
    return Counter(tokens)



def _loggable_query(question: str, limit: int = 160) -> str:
    """Loggable query.
    
    Args:
        question (str): The user question to answer.
        limit (int): Maximum number of items to return.
    
    Returns:
        str: Formatted text result.
    """

    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],

    top_n: int,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks with an external reranker if configured.
    
    Args:
        question (str): The user question to answer.
        chunks (list[RetrievedChunk]): Candidate chunks to evaluate or return.
        top_n (int): Input parameter.
    
    Returns:
        list[RetrievedChunk]: List of results.
    """

    api_key = get_effective_openrouter_api_key()
    if not settings.rerank_model or not api_key:
        return chunks
    if len(chunks) <= 1:
        return chunks

    payload = {
        "model": settings.rerank_model,
        "query": question,
        "documents": [chunk.text for chunk in chunks],
        "top_n": min(top_n, len(chunks)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoint = settings.openrouter_base_url.rstrip("/") + "/rerank"
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
    except (error.URLError, json.JSONDecodeError, TimeoutError, Exception):
        return chunks

    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        return chunks

    ranked: list[RetrievedChunk] = []
    seen_indices: set[int] = set()
    for item in results:
        index = item.get("index")
        if not isinstance(index, int):
            continue
        if index < 0 or index >= len(chunks):
            continue
        if index in seen_indices:
            continue
        seen_indices.add(index)
        ranked.append(chunks[index])
    for index, chunk in enumerate(chunks):
        if index in seen_indices:
            continue
        ranked.append(chunk)
    return ranked[:top_n]



def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Compute cosine similarity between two token counters.
    
    Args:
        left (Counter[str]): Input parameter.
        right (Counter[str]): Input parameter.
    
    Returns:
        float: Computed numeric result.
    """

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
    *,
    document_id: str | None = None,
    user_id: str | None = None,

    top_k: int | None = None,
    use_rerank: bool = True,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.
    
    Args:
        question (str): The user question to answer.
        chunks (list[Chunk]): Candidate chunks to evaluate or return.
        session_id (str): Session identifier for the current chat.
        document_id (str | None): Document identifier for the current workspace.
        user_id (str | None): Input parameter.
        top_k (int | None): Input parameter.
    
    Returns:
        list[RetrievedChunk]: List of results.
    """

    top_k = top_k or settings.top_k
    dense_candidate_k = top_k
    rerank_enabled = use_rerank and bool(settings.rerank_model)
    if rerank_enabled:
        dense_candidate_k = max(top_k, settings.rerank_candidates)  # pull extra candidates so reranker can reshuffle
    lexical_candidate_k = max(dense_candidate_k, top_k * 4)
    merged_candidate_k = max(dense_candidate_k, lexical_candidate_k)
    query_embedding = embed_text(question)
    query_label = _loggable_query(question)
    remote_results: list[RetrievedChunk] = []
    if is_embedding_vector(query_embedding):
        remote_results = query_remote_chunks(
            query_embedding,
            session_id,
            dense_candidate_k,
            document_id=document_id,
            user_id=user_id,
        )
    local_results = _local_retrieval(question, chunks, session_id, document_id, lexical_candidate_k)

    if remote_results and local_results:
        merged_candidates = _merge_retrieval_results(remote_results, local_results, merged_candidate_k)
        retrieval_mode = "hybrid"
    elif remote_results:
        merged_candidates = remote_results[:merged_candidate_k]
        retrieval_mode = "remote_only"
    else:
        merged_candidates = local_results[:merged_candidate_k]
        retrieval_mode = "local_only"

    if not merged_candidates:
        log_langfuse_event(
            "retrieval",
            session_id=session_id,
            metadata={
                "mode": "empty",
                "num_results": 0,
                "top_k": top_k,
                "query": query_label,
                "document_id": document_id or "",
                "dense_candidates": len(remote_results),
                "lexical_candidates": len(local_results),
                "merged_candidates": 0,
            },
        )
        return []

    if rerank_enabled:
        final_results = _rerank_chunks(question, merged_candidates, top_k)
        log_langfuse_event(
            "rerank",
            session_id=session_id,
            metadata={
                "model": settings.rerank_model,
                "candidates": len(merged_candidates),
                "top_n": top_k,
                "query": query_label,
                "document_id": document_id or "",
            },
        )
    else:
        final_results = merged_candidates[:top_k]
    log_langfuse_event(
        "retrieval",
        session_id=session_id,
        metadata={
            "mode": retrieval_mode,
            "num_results": len(final_results),
            "top_k": top_k,
            "query": query_label,
            "document_id": document_id or "",
            "dense_candidates": len(remote_results),
            "lexical_candidates": len(local_results),
            "merged_candidates": len(merged_candidates),
            "rerank_enabled": rerank_enabled,
        },
    )
    return final_results


def _local_retrieval(
    question: str,
    chunks: list[Chunk],
    session_id: str,
    document_id: str | None,

    top_k: int,
) -> list[RetrievedChunk]:
    """Local retrieval.
    
    Args:
        question (str): The user question to answer.
        chunks (list[Chunk]): Candidate chunks to evaluate or return.
        session_id (str): Session identifier for the current chat.
        document_id (str | None): Document identifier for the current workspace.
        top_k (int): Input parameter.
    
    Returns:
        list[RetrievedChunk]: List of results.
    """

    query_tokens = _tokenize(question)
    session_chunks = [chunk for chunk in chunks if chunk.session_id == session_id]
    if document_id:
        document_chunks = [chunk for chunk in chunks if chunk.document_id == document_id]
        if document_chunks:
            session_chunks = document_chunks  # prefer active document to avoid cross-doc bleed
    if not session_chunks and chunks:
        # Legacy safety: if session/document metadata drift, fall back to the currently loaded
        # in-memory workspace chunk list instead of returning an empty retrieval set.
        session_chunks = chunks
    scored = [
        (chunk, _cosine_similarity(query_tokens, _tokenize(chunk.text)))
        for chunk in session_chunks
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    scored = [item for item in scored if item[1] > 0][:top_k]
    return normalize_retrieved_chunks(scored)


def _merge_retrieval_results(
    remote_results: list[RetrievedChunk],
    local_results: list[RetrievedChunk],

    top_k: int,
) -> list[RetrievedChunk]:
    """Merge retrieval results.
    
    Args:
        remote_results (list[RetrievedChunk]): Input parameter.
        local_results (list[RetrievedChunk]): Input parameter.
        top_k (int): Input parameter.
    
    Returns:
        list[RetrievedChunk]: List of results.
    """

    merged: list[RetrievedChunk] = []
    seen_ids: set[int] = set()
    for result in remote_results + local_results:
        if result.chunk_id in seen_ids:
            continue
        seen_ids.add(result.chunk_id)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged
