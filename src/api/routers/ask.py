"""ask.py: Placeholder routes for Ask the Book APIs."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.models import PlaceholderResponse


router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    """Input payload for Ask endpoints."""

    question: str
    document_id: str | None = None


@router.post("/book", response_model=PlaceholderResponse)
def ask_the_book(payload: AskRequest) -> PlaceholderResponse:
    """Run RAG-style Ask the Book (placeholder)."""
    _ = payload
    return PlaceholderResponse(
        endpoint="/api/ask/book",
        message="Ask the Book is not implemented yet.",
    )
