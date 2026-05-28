"""publishing.py: Placeholder routes for publishing generation APIs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.models import PlaceholderResponse
from src.api.services.api_logging import log_api_event
from src.api.services.document_workspace import (
    DocumentRetrievalError,
    DocumentWorkspaceAccessDeniedError,
    DocumentWorkspaceNotFoundError,
)
from src.api.services.rate_limit import enforce_publishing_rate_limit
from src.api.services.request_identity import resolve_request_user_id
from src.api.services.service_errors import ServiceTimeoutError
from src.api.services.publishing_service import (
    generate_book_brief_for_document,
    generate_marketing_copy_for_document,
)
from src.publishing.schemas import (
    BookBriefRequest,
    BookBriefResponse,
    MarketingCopyRequest,
    MarketingCopyResponse,
)


router = APIRouter(prefix="/publishing", tags=["publishing"])


@router.post("/{doc_id}/book-brief", response_model=BookBriefResponse)
def generate_book_brief(
    doc_id: str,
    payload: BookBriefRequest,
    request: Request,
    user_id: str | None = None,
    _: None = Depends(enforce_publishing_rate_limit),
) -> BookBriefResponse:
    """Generate a grounded, structured Book Brief for one document."""
    log_api_event(
        "api.publishing.book_brief.request",
        doc_id=doc_id,
        mode="positioning_brief",
        output_type="book_brief",
        user_hint_provided=bool(user_id),
    )
    try:
        resolved_user_id = resolve_request_user_id(
            request,
            query_user_id=user_id,
        )
        response = generate_book_brief_for_document(
            doc_id=doc_id,
            request=payload,
            user_id=resolved_user_id,
        )
        log_api_event(
            "api.publishing.book_brief.success",
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            run_id=response.metadata.run_id,
            model=response.metadata.model,
            latency_ms=response.metadata.latency_ms,
            retrieved_chunk_count=response.metadata.retrieved_chunk_count,
        )
        return response
    except DocumentWorkspaceNotFoundError as exc:
        log_api_event(
            "api.publishing.book_brief.not_found",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentWorkspaceAccessDeniedError as exc:
        log_api_event(
            "api.publishing.book_brief.forbidden",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except DocumentRetrievalError as exc:
        log_api_event(
            "api.publishing.book_brief.retrieval_error",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        log_api_event(
            "api.publishing.book_brief.bad_request",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ServiceTimeoutError:
        log_api_event(
            "api.publishing.book_brief.timeout",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Positioning Brief generation timed out. "
                "Please retry or reduce prompt complexity."
            ),
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        log_api_event(
            "api.publishing.book_brief.error",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="positioning_brief",
            output_type="book_brief",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while generating the book brief.",
        ) from exc


@router.post("/{doc_id}/marketing-copy", response_model=MarketingCopyResponse)
def generate_marketing_copy(
    doc_id: str,
    payload: MarketingCopyRequest,
    request: Request,
    user_id: str | None = None,
    _: None = Depends(enforce_publishing_rate_limit),
) -> MarketingCopyResponse:
    """Generate grounded marketing copy for one document."""
    log_api_event(
        "api.publishing.marketing_copy.request",
        doc_id=doc_id,
        mode="marketing_copy",
        output_type=payload.output_type,
        user_hint_provided=bool(user_id),
    )
    try:
        resolved_user_id = resolve_request_user_id(
            request,
            query_user_id=user_id,
        )
        response = generate_marketing_copy_for_document(
            doc_id=doc_id,
            request=payload,
            user_id=resolved_user_id,
        )
        log_api_event(
            "api.publishing.marketing_copy.success",
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=response.output_type,
            run_id=response.metadata.run_id,
            model=response.metadata.model,
            latency_ms=response.metadata.latency_ms,
            retrieved_chunk_count=response.metadata.retrieved_chunk_count,
        )
        return response
    except DocumentWorkspaceNotFoundError as exc:
        log_api_event(
            "api.publishing.marketing_copy.not_found",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentWorkspaceAccessDeniedError as exc:
        log_api_event(
            "api.publishing.marketing_copy.forbidden",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except DocumentRetrievalError as exc:
        log_api_event(
            "api.publishing.marketing_copy.retrieval_error",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        log_api_event(
            "api.publishing.marketing_copy.bad_request",
            level=logging.WARNING,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ServiceTimeoutError:
        log_api_event(
            "api.publishing.marketing_copy.timeout",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Marketing Copy generation timed out. "
                "Please retry or use a shorter output request."
            ),
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        log_api_event(
            "api.publishing.marketing_copy.error",
            level=logging.ERROR,
            doc_id=doc_id,
            mode="marketing_copy",
            output_type=payload.output_type,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while generating marketing copy.",
        ) from exc


@router.post("/reader-persona", response_model=PlaceholderResponse)
def generate_reader_persona() -> PlaceholderResponse:
    """Generate a reader persona (placeholder)."""
    return PlaceholderResponse(
        endpoint="/api/publishing/reader-persona",
        message="Reader Persona generation is not implemented yet.",
    )
