"""index_cache.py: Persist and restore the local study library cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.core.models import Chunk, QuizResult, StudyPlan, StudyQuiz


cache_directory = Path(".studypal_cache")
document_library_path = cache_directory / "document_library.json"
legacy_indexed_document_path = cache_directory / "last_indexed_document.json"

workspace_model_keys: dict[str, type[BaseModel]] = {
    "chunks": Chunk,
    "current_quiz": StudyQuiz,
    "last_quiz_result": QuizResult,
    "study_plan": StudyPlan,
}



def _serialize_value(value: Any) -> Any:
    """Convert models and nested structures into JSON-friendly values.
    
    Args:
        value (Any): Input parameter.
    
    Returns:
        Any: Result value.
    """

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value



def _deserialize_workspace(workspace: dict[str, Any]) -> dict[str, Any]:
    """Restore cached workspace data back into model objects.
    
    Args:
        workspace (dict[str, Any]): Input parameter.
    
    Returns:
        dict[str, Any]: Mapping of computed results.
    """

    restored_workspace = dict(workspace)
    for key, model in workspace_model_keys.items():
        value = restored_workspace.get(key)
        if value is None:
            continue
        if key == "chunks":
            restored_workspace[key] = [model.model_validate(item) for item in value]
        else:
            restored_workspace[key] = model.model_validate(value)
    return restored_workspace


def persist_document_library(
    document_library: list[dict[str, object]],

    active_document_id: str | None,
) -> None:
    """Save the document library and active workspace to disk.
    
    Args:
        document_library (list[dict[str, object]]): Input parameter.
        active_document_id (str | None): Input parameter.
    """

    cache_directory.mkdir(exist_ok=True)
    payload = {
        "active_document_id": active_document_id,
        "document_library": [_serialize_value(workspace) for workspace in document_library],
    }
    document_library_path.write_text(json.dumps(payload), encoding="utf-8")



def restore_document_library() -> tuple[list[dict[str, object]], str | None]:
    """Load the document library and active workspace from disk.
    
    Returns:
        tuple[list[dict[str, object]], str | None]: Result value.
    """

    if not document_library_path.exists():
        return _restore_legacy_document_library()

    try:
        payload = json.loads(document_library_path.read_text(encoding="utf-8"))
        document_library = [
            _deserialize_workspace(workspace)
            for workspace in payload.get("document_library", [])
        ]
        for workspace in document_library:
            summary = workspace.get("document_summary")
            if isinstance(summary, str) and summary.strip().lower().startswith("recovered workspace for"):
                workspace["document_summary"] = None  # drop the Pinecone recovery banner from card UI
        active_document_id = payload.get("active_document_id")
        if active_document_id is not None:
            active_document_id = str(active_document_id)
        return document_library, active_document_id
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _restore_legacy_document_library()



def _restore_legacy_document_library() -> tuple[list[dict[str, object]], str | None]:
    """Fallback loader for the legacy cache format.
    
    Returns:
        tuple[list[dict[str, object]], str | None]: Result value.
    """

    if not legacy_indexed_document_path.exists():
        return [], None

    try:
        payload = json.loads(legacy_indexed_document_path.read_text(encoding="utf-8"))
        document_id = str(payload.get("document_id", "legacy-doc"))
        chunks = [Chunk.model_validate(chunk) for chunk in payload["chunks"]]
        workspace = {
            "document_id": document_id,
            "session_id": str(payload["session_id"]),
            "filename": str(payload["filename"]),
            "document_title": str(payload.get("filename", "")).rsplit(".", 1)[0].replace("_", " ").title(),
            "document_topic": str(payload.get("filename", "")).rsplit(".", 1)[0].replace("_", " ").title(),
            "document_summary": None,
            "key_hooks": [],
            "chunks": chunks,
            "size_mb": float(payload.get("size_mb", 0.0)),
            "chunk_count": len(chunks),
            "last_conversation_topic": None,
            "last_opened_at": None,
            "messages": [],
            "message_feedback": {},
            "current_mode": "ask",
            "conversation_topic": None,
            "quiz_goal": None,
            "study_topic": None,
            "mastery_intro": None,
            "mastery_intro_citations": [],
            "remediation_message": None,
            "remediation_citations": [],
            "mastery_status": "idle",
            "current_quiz": None,
            "quiz_round": 0,
            "last_quiz_result": None,
            "weak_concepts": [],
            "study_plan": None,
            "study_plan_citations": [],
        }
        return [workspace], document_id
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return [], None
