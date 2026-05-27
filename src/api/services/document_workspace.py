"""document_workspace.py: Shared document workspace lookup and retrieval helpers."""

from __future__ import annotations

from src.core.models import Chunk, RetrievedChunk
from src.data.index_cache import restore_document_library
from src.data.retrieval import retrieve_chunks
from src.data.vector_store import rebuild_document_library_from_remote


class DocumentWorkspaceNotFoundError(ValueError):
    """Raised when a requested document workspace cannot be found."""


class DocumentRetrievalError(RuntimeError):
    """Raised when grounded retrieval for a document cannot produce context."""


class DocumentWorkspaceAccessDeniedError(PermissionError):
    """Raised when the requester does not own the requested document workspace."""


def normalize_user_id(user_id: str | None) -> str | None:
    """Normalize optional user id values."""
    if not user_id:
        return None
    cleaned = user_id.strip()
    return cleaned or None


def workspace_document_id(workspace: dict[str, object]) -> str:
    """Return the normalized document id from a workspace record."""
    return str(workspace.get("document_id") or "").strip()


def workspace_user_id(workspace: dict[str, object]) -> str | None:
    """Return the normalized user id from a workspace record."""
    raw = workspace.get("user_id")
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def workspace_session_id(workspace: dict[str, object]) -> str:
    """Return the normalized session id from a workspace record."""
    return str(workspace.get("session_id") or "").strip()


def workspace_chunks(workspace: dict[str, object]) -> list[Chunk]:
    """Coerce workspace chunk payloads into Chunk models."""
    raw_chunks = workspace.get("chunks")
    if not isinstance(raw_chunks, list):
        return []

    normalized: list[Chunk] = []
    for raw_chunk in raw_chunks:
        if isinstance(raw_chunk, Chunk):
            normalized.append(raw_chunk)
            continue
        if isinstance(raw_chunk, dict):
            try:
                normalized.append(Chunk.model_validate(raw_chunk))
            except Exception:
                continue
    return normalized


def find_workspace(doc_id: str, user_id: str | None) -> dict[str, object] | None:
    """Resolve a document workspace from local cache or remote index."""
    normalized_doc_id = doc_id.strip()
    if not normalized_doc_id:
        return None

    checked_workspaces, _ = restore_document_library(user_id=user_id)
    for workspace in checked_workspaces:
        if workspace_document_id(workspace) == normalized_doc_id:
            return workspace

    remote_workspaces = rebuild_document_library_from_remote(user_id=user_id)
    for workspace in remote_workspaces:
        if workspace_document_id(workspace) == normalized_doc_id:
            return workspace
    return None


def load_workspace(doc_id: str, user_id: str | None) -> dict[str, object]:
    """Load a workspace or raise a not-found error."""
    normalized_user_id = normalize_user_id(user_id)
    workspace = find_workspace(doc_id, user_id)
    if workspace is None:
        raise DocumentWorkspaceNotFoundError(f"Document '{doc_id}' was not found.")
    owner_user_id = workspace_user_id(workspace)
    if normalized_user_id and owner_user_id and normalized_user_id != owner_user_id:
        raise DocumentWorkspaceAccessDeniedError("You do not have access to this document workspace.")
    return workspace


def retrieve_workspace_context(
    *,
    workspace: dict[str, object],
    question: str,
    user_id: str | None = None,
    top_k: int | None = None,
    use_rerank: bool = True,
) -> list[RetrievedChunk]:
    """Retrieve grounded chunks for a question within one workspace."""
    chunks = workspace_chunks(workspace)
    if not chunks:
        raise DocumentRetrievalError("Document exists but has no indexed chunks.")

    resolved_user_id = workspace_user_id(workspace) or normalize_user_id(user_id)
    retrieved_chunks = retrieve_chunks(
        question=question,
        chunks=chunks,
        session_id=workspace_session_id(workspace),
        document_id=workspace_document_id(workspace),
        user_id=resolved_user_id,
        top_k=top_k,
        use_rerank=use_rerank,
    )
    if not retrieved_chunks:
        raise DocumentRetrievalError(
            "No relevant grounded content was retrieved for this document and question."
        )
    return retrieved_chunks
