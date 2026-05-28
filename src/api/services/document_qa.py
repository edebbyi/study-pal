"""document_qa.py: Reusable document-grounded Q&A service for API routes."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from src.api.models import DocumentAskMetadata, DocumentAskResponse, DocumentAskSource
from src.api.services.api_logging import log_api_event
from src.api.services.evaluation_metrics import build_live_run_metrics
from src.api.services.document_workspace import (
    DocumentRetrievalError,
    load_workspace,
    normalize_user_id,
    retrieve_workspace_context,
    workspace_chunks,
)
from src.api.services.observability_service import (
    end_mlflow_run,
    end_phoenix_trace,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    log_phoenix_generation,
    log_phoenix_retrieval,
    log_phoenix_scores,
    start_mlflow_run,
    start_phoenix_trace,
)
from src.api.services.service_errors import ServiceTimeoutError
from src.api.services.timeout_utils import run_with_timeout
from src.core.config import settings
from src.core.models import Chunk, RetrievedChunk
from src.core.openrouter_credentials import get_openrouter_api_key_for_user
from src.llm.llm_client import answer_from_context


DEFAULT_OBSERVABILITY_TEMPERATURE = "not_exposed"


def _error_category_metrics(category: str) -> dict[str, float]:
    """Build one-hot style error category metrics for observability backends."""
    return {
        "error_count": 1.0,
        "error_timeout_failure": 1.0 if category == "timeout_failure" else 0.0,
        "error_retrieval_failure": 1.0 if category == "retrieval_failure" else 0.0,
        "error_model_failure": 1.0 if category == "model_failure" else 0.0,
        "error_validation_failure": 1.0 if category == "validation_failure" else 0.0,
        "error_unexpected_failure": 1.0 if category == "unexpected_failure" else 0.0,
    }


_BROAD_THEME_HINTS: tuple[str, ...] = (
    "core theme",
    "main theme",
    "major theme",
    "key theme",
    "big picture",
    "overall",
    "summary",
    "overview",
    "across the book",
    "entire book",
    "whole book",
)

_NAMED_LIST_HINTS: tuple[str, ...] = (
    "core concept",
    "eight core concept",
    "8 core concept",
    "list the eight",
    "what are the eight",
)


def _is_broad_document_question(question: str) -> bool:
    """Return True when the question asks for book-level synthesis."""
    normalized = " ".join(question.lower().split())
    if not normalized:
        return False
    return any(hint in normalized for hint in _BROAD_THEME_HINTS)


def _is_named_list_question(question: str) -> bool:
    """Return True when the question likely targets a named enumerated list."""
    normalized = " ".join(question.lower().split())
    if not normalized:
        return False
    return any(hint in normalized for hint in _NAMED_LIST_HINTS)


def _retrieval_query(question: str, *, broad_scope: bool) -> str:
    """Build retrieval query text, adding broad-scope anchors when needed."""
    if not broad_scope:
        return question
    return (
        f"{question}\n\n"
        "Focus on broad coverage: overall summary, introduction, preface, table of contents, "
        "chapter overview, key ideas, major themes, and recurring concepts."
    )


def _chunk_to_retrieved(chunk: Chunk, *, score: float) -> RetrievedChunk:
    """Convert a workspace chunk into a retrieved chunk record."""
    return RetrievedChunk(
        text=chunk.text,
        filename=chunk.filename,
        page=chunk.page,
        citation=chunk.citation,
        score=score,
        chunk_id=chunk.chunk_id,
        chapter=chunk.chapter,
        topic=chunk.topic,
        chapter_index=chunk.chapter_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_index_in_page=chunk.chunk_index_in_page,
    )


def _augment_named_list_context(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    all_chunks: list[Chunk],
    target_k: int,
) -> list[RetrievedChunk]:
    """Add direct phrase matches + nearby chunks for named-list style questions."""
    if not all_chunks:
        return retrieved_chunks

    normalized_question = question.lower()
    phrase_hits: list[Chunk] = []
    for chunk in all_chunks:
        text = chunk.text.lower()
        if any(hint in text for hint in _NAMED_LIST_HINTS):
            phrase_hits.append(chunk)
            continue
        # Light heuristic for numbered-list style chunks.
        if ("core concepts" in normalized_question) and any(token in text for token in ("1.", "2.", "3.", "4.")):
            phrase_hits.append(chunk)

    if not phrase_hits:
        return retrieved_chunks

    chunk_by_id = {int(chunk.chunk_id): chunk for chunk in all_chunks}
    neighbor_candidates: list[Chunk] = []
    for hit in phrase_hits:
        for offset in (-2, -1, 0, 1, 2):
            neighbor = chunk_by_id.get(int(hit.chunk_id) + offset)
            if neighbor is not None:
                neighbor_candidates.append(neighbor)

    combined: list[RetrievedChunk] = []
    seen_ids: set[int] = set()
    # Keep vector-search ranking first.
    for item in retrieved_chunks:
        if int(item.chunk_id) in seen_ids:
            continue
        seen_ids.add(int(item.chunk_id))
        combined.append(item)
    # Add lexical exact/nearby context with moderate scores.
    for neighbor in neighbor_candidates:
        neighbor_id = int(neighbor.chunk_id)
        if neighbor_id in seen_ids:
            continue
        seen_ids.add(neighbor_id)
        score = 0.35 if any(hint in neighbor.text.lower() for hint in _NAMED_LIST_HINTS) else 0.22
        combined.append(_chunk_to_retrieved(neighbor, score=score))

    return combined[:target_k]


def ask_document_question(
    *,
    doc_id: str,
    question: str,
    user_id: str | None = None,
) -> DocumentAskResponse:
    """Run document-grounded retrieval + answer generation for one question.

    Raises:
        DocumentWorkspaceNotFoundError: If no workspace exists for the provided doc_id.
        DocumentRetrievalError: If retrieval cannot find grounded context for the question.
    """
    run_id = str(uuid4())
    normalized_user_id = normalize_user_id(user_id)
    cleaned_question = " ".join(question.strip().split())
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")
    broad_scope = _is_broad_document_question(cleaned_question)
    named_list_scope = _is_named_list_question(cleaned_question)
    if named_list_scope:
        retrieval_top_k = max(settings.top_k, 18)
    elif broad_scope:
        retrieval_top_k = max(settings.top_k, 12)
    else:
        retrieval_top_k = settings.top_k

    started_at = perf_counter()
    log_api_event(
        "service.ask_the_book.start",
        run_id=run_id,
        doc_id=doc_id,
        mode="ask_the_book",
        output_type="ask_the_book",
        broad_scope=broad_scope,
        named_list_scope=named_list_scope,
        top_k=retrieval_top_k,
    )
    phoenix_trace = start_phoenix_trace(
        span_name="publishing.ask_the_book",
        metadata={
            "run_id": run_id,
            "doc_id": doc_id,
            "mode": "ask_the_book",
            "output_type": "ask_the_book",
            "prompt_version": settings.prompt_version or "",
            "model": settings.chat_model,
            "top_k": retrieval_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
        },
    )
    mlflow_run = start_mlflow_run(
        run_name=f"publishing.ask_the_book.{run_id}",
        tags={
            "run_id": run_id,
            "document_id": doc_id,
            "prompt_version": settings.prompt_version or "",
            "mode": "ask_the_book",
            "output_type": "ask_the_book",
            "app_env": settings.app_env,
        },
    )
    log_mlflow_params(
        mlflow_run,
        {
            "mode": "ask_the_book",
            "output_type": "ask_the_book",
            "model": settings.chat_model,
            "prompt_version": settings.prompt_version,
            "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
            "top_k": retrieval_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
            "document_type": "unknown",
            "spoiler_level": "low",
            "tone": None,
            "audience_provided": "no",
            "question_scope": "broad" if broad_scope else "targeted",
        },
    )
    try:
        workspace = load_workspace(doc_id, normalized_user_id)
        retrieved_chunks = retrieve_workspace_context(
            workspace=workspace,
            question=_retrieval_query(cleaned_question, broad_scope=broad_scope),
            user_id=normalized_user_id,
            top_k=retrieval_top_k,
        )
        if named_list_scope:
            retrieved_chunks = _augment_named_list_context(
                question=cleaned_question,
                retrieved_chunks=retrieved_chunks,
                all_chunks=workspace_chunks(workspace),
                target_k=retrieval_top_k,
            )
        log_api_event(
            "service.ask_the_book.retrieval_complete",
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
            retrieved_chunk_count=len(retrieved_chunks),
        )
        log_phoenix_retrieval(
            phoenix_trace,
            {
                "mode": "ask_the_book",
                "output_type": "ask_the_book",
                "document_id": doc_id,
                "document_title": str(workspace.get("document_title") or "").strip() or None,
                "prompt_version": settings.prompt_version or "",
                "model": settings.chat_model,
                "temperature": DEFAULT_OBSERVABILITY_TEMPERATURE,
                "top_k": retrieval_top_k,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "retrieval_algorithm": settings.retrieval_algorithm,
                "question": cleaned_question,
                "retrieved_chunk_count": len(retrieved_chunks),
                "chunks": [
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "score": float(chunk.score),
                        "preview": " ".join(chunk.text.split())[:260],
                    }
                    for chunk in retrieved_chunks
                ],
            },
        )
        filename_value = str(workspace.get("filename") or "").strip()
        document_type = filename_value.rsplit(".", 1)[-1].lower() if "." in filename_value else "unknown"
        log_mlflow_params(
            mlflow_run,
            {
                "document_type": document_type,
            },
        )

        answer_payload = run_with_timeout(
            operation="answer_from_context",
            mode="ask_the_book",
            run_output_type="ask_the_book",
            timeout_seconds=settings.api_timeout_ask_seconds,
            fn=answer_from_context,
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
            api_key_override=get_openrouter_api_key_for_user(normalized_user_id),
            theme_synthesis=broad_scope,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
    except ServiceTimeoutError:
        log_api_event(
            "service.ask_the_book.timeout",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
            timeout_seconds=settings.api_timeout_ask_seconds,
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "timeout_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("timeout_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        end_phoenix_trace(phoenix_trace)
        raise
    except DocumentRetrievalError:
        log_api_event(
            "service.ask_the_book.retrieval_failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        log_phoenix_scores(
            phoenix_trace,
            {
                "error_category": "retrieval_failure",
                "human_review_recommended": True,
            },
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("retrieval_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        end_phoenix_trace(phoenix_trace)
        raise
    except ValueError:
        log_api_event(
            "service.ask_the_book.validation_failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("validation_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        end_phoenix_trace(phoenix_trace)
        raise
    except Exception:
        log_api_event(
            "service.ask_the_book.failed",
            level=logging.ERROR,
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        log_mlflow_metrics(mlflow_run, _error_category_metrics("model_failure"))
        end_mlflow_run(mlflow_run, status="FAILED")
        end_phoenix_trace(phoenix_trace)
        raise
    warning = None
    model_name = settings.chat_model
    if answer_payload.used_fallback:
        model_name = "retrieval_fallback"
        warning = (
            "Answer was generated from retrieved chunks without a live model completion. "
            "Configure an OpenRouter key for full model responses."
        )
        log_api_event(
            "service.ask_the_book.fallback_used",
            level=logging.WARNING,
            run_id=run_id,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )

    metrics_payload = build_live_run_metrics(
        retrieved_chunks=retrieved_chunks,
        mode="ask_the_book",
        output_type="ask_the_book",
        missing_context_flags=[],
        unsupported_claims_detected=False,
    )
    workflow_trace = dict(metrics_payload.get("workflow_trace", {}))
    workflow_trace["spoiler_setting"] = "low"
    quality_checks = dict(metrics_payload.get("quality_checks", {}))
    quality_checks["spoiler_level"] = "low"
    run_details = dict(metrics_payload.get("run_details_metrics", {}))
    run_details.update(
        {
            "run_id": run_id,
            "model": model_name,
            "latency_ms": latency_ms,
            "latency_seconds": round(latency_ms / 1000, 3),
            "phoenix_trace_id": phoenix_trace.trace_id if phoenix_trace else None,
            "mlflow_run_id": mlflow_run.run_id if mlflow_run else None,
        }
    )

    response = DocumentAskResponse(
        answer=answer_payload.answer,
        sources=[
            DocumentAskSource(
                chunk_id=str(chunk.chunk_id),
                text=chunk.text,
                score=float(chunk.score),
            )
            for chunk in retrieved_chunks
        ],
        metadata=DocumentAskMetadata(
            model=model_name,
            latency_ms=latency_ms,
            latency_seconds=round(latency_ms / 1000, 3),
            retrieved_chunk_count=len(retrieved_chunks),
            avg_relevance_score=run_details.get("avg_relevance_score") if isinstance(run_details.get("avg_relevance_score"), (int, float)) else None,
            top_relevance_score=run_details.get("top_relevance_score") if isinstance(run_details.get("top_relevance_score"), (int, float)) else None,
            source_diversity_score=run_details.get("source_diversity_score") if isinstance(run_details.get("source_diversity_score"), (int, float)) else None,
            context_coverage_score=run_details.get("context_coverage_score") if isinstance(run_details.get("context_coverage_score"), (int, float)) else None,
            phoenix_trace_id=phoenix_trace.trace_id if phoenix_trace else None,
            mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            warning=warning,
        ),
        workflow_trace=workflow_trace,
        quality_checks=quality_checks,
        run_details=run_details,
    )
    log_phoenix_generation(
        phoenix_trace,
        {
            "mode": "ask_the_book",
            "output_type": "ask_the_book",
            "model": model_name,
            "document_id": doc_id,
            "document_title": str(workspace.get("document_title") or "").strip() or None,
            "latency_ms": latency_ms,
            "answer_preview": " ".join(answer_payload.answer.split())[:300],
        },
    )
    log_phoenix_scores(phoenix_trace, quality_checks)
    log_mlflow_metrics(
        mlflow_run,
        {
            "latency_ms": latency_ms,
            "latency_seconds": run_details.get("latency_seconds"),
            "retrieved_chunk_count": len(retrieved_chunks),
            "avg_relevance_score": run_details.get("avg_relevance_score"),
            "top_relevance_score": run_details.get("top_relevance_score"),
            "chunk_count_score": run_details.get("chunk_count_score"),
            "source_diversity_score": run_details.get("source_diversity_score"),
            "context_coverage_score": run_details.get("context_coverage_score"),
            "input_token_count": run_details.get("input_token_count"),
            "output_token_count": run_details.get("output_token_count"),
            "total_token_count": run_details.get("total_token_count"),
            "estimated_cost_usd": run_details.get("estimated_cost_usd"),
            "unsupported_claim_count": run_details.get("unsupported_claim_count"),
        },
    )
    log_mlflow_artifacts(
        mlflow_run,
        {
            "generated_output": response.model_dump(),
            "retrieved_chunks": [item.model_dump() for item in response.sources],
            "quality_checks": quality_checks,
            "workflow_trace": workflow_trace,
            "run_details": run_details,
            "request_metadata": {
                "doc_id": doc_id,
                "question": cleaned_question,
                "broad_scope": broad_scope,
                "named_list_scope": named_list_scope,
            },
        },
    )
    end_mlflow_run(mlflow_run, status="FINISHED")
    end_phoenix_trace(phoenix_trace)
    log_api_event(
        "service.ask_the_book.complete",
        run_id=run_id,
        doc_id=doc_id,
        mode="ask_the_book",
        output_type="ask_the_book",
        model=model_name,
        latency_ms=latency_ms,
        retrieved_chunk_count=len(retrieved_chunks),
        context_coverage_label=workflow_trace.get("context_coverage_label"),
    )
    return response
