"""documents.py: Placeholder routes for document APIs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.models import DocumentAskRequest, DocumentAskResponse, PlaceholderResponse, PublishingRunRecord
from src.api.services.api_logging import log_api_event
from src.api.services.document_qa import ask_document_question
from src.api.services.service_errors import ServiceTimeoutError
from src.api.services.publishing_runs import load_document_runs
from src.api.services.document_workspace import (
    DocumentRetrievalError,
    DocumentWorkspaceAccessDeniedError,
    DocumentWorkspaceNotFoundError,
)
from src.api.services.rate_limit import enforce_ask_rate_limit
from src.api.services.request_identity import resolve_request_user_id


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=PlaceholderResponse)
def list_documents() -> PlaceholderResponse:
    """List uploaded documents (placeholder)."""
    return PlaceholderResponse(
        endpoint="/api/documents",
        message="Document listing is not implemented yet.",
    )


@router.post("/upload", response_model=PlaceholderResponse)
def upload_document() -> PlaceholderResponse:
    """Upload and index a document (placeholder)."""
    return PlaceholderResponse(
        endpoint="/api/documents/upload",
        message="Document upload is not implemented yet.",
    )


@router.post("/{doc_id}/ask", response_model=DocumentAskResponse)
def ask_document(
    doc_id: str,
    payload: DocumentAskRequest,
    request: Request,
    user_id: str | None = None,
    _: None = Depends(enforce_ask_rate_limit),
) -> DocumentAskResponse:
    """Answer a question grounded in one indexed document."""
    log_api_event(
        "api.documents.ask.request",
        doc_id=doc_id,
        mode="ask_the_book",
        output_type="ask_the_book",
        user_hint_provided=bool(user_id or payload.user_id),
    )
    try:
        resolved_user_id = resolve_request_user_id(
            request,
            query_user_id=user_id,
            body_user_id=payload.user_id,
        )
        response = ask_document_question(
            doc_id=doc_id,
            question=payload.question,
            user_id=resolved_user_id,
        )
        log_api_event(
            "api.documents.ask.success",
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
            run_id=response.run_details.get("run_id") if isinstance(response.run_details, dict) else None,
            model=response.metadata.model,
            latency_ms=response.metadata.latency_ms,
            retrieved_chunk_count=response.metadata.retrieved_chunk_count,
        )
        return response
    except DocumentWorkspaceNotFoundError as exc:
        log_api_event(
            "api.documents.ask.not_found",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentWorkspaceAccessDeniedError as exc:
        log_api_event(
            "api.documents.ask.forbidden",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except DocumentRetrievalError as exc:
        log_api_event(
            "api.documents.ask.retrieval_error",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        log_api_event(
            "api.documents.ask.bad_request",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ServiceTimeoutError:
        log_api_event(
            "api.documents.ask.timeout",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Ask the Book timed out while generating a response. "
                "Please retry, reduce prompt scope, or try again shortly."
            ),
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        log_api_event(
            "api.documents.ask.error",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="ask_the_book",
            output_type="ask_the_book",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while answering the document question.",
        ) from exc


@router.get("/{doc_id}/runs", response_model=list[PublishingRunRecord])
def list_document_runs(doc_id: str, limit: int = 25) -> list[PublishingRunRecord]:
    """List recent Publishing Mode runs for one document."""
    normalized_doc_id = doc_id.strip()
    if not normalized_doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document id cannot be empty.",
        )
    safe_limit = min(max(int(limit), 1), 200)
    return load_document_runs(doc_id=normalized_doc_id, limit=safe_limit)
