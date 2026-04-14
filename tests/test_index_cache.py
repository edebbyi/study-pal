"""test_index_cache.py: Tests for test_index_cache.py."""

from __future__ import annotations

from pathlib import Path

import src.index_cache as index_cache
from src.app_state import IndexedDocument
from src.models import Chunk


def _build_indexed_document() -> IndexedDocument:
    return IndexedDocument(
        document_id="doc-123",
        session_id="session-123",
        filename="anatomy_example.pdf",
        document_title="Anatomy Example",
        document_topic="Human Anatomy",
        document_summary="A survey of the major systems of the human body.",
        size_mb=69.6,
        chunks=[
            Chunk(
                id="session-123-0",
                text="The brain is the control center of the nervous system.",
                filename="anatomy_example.pdf",
                page=12,
                chunk_id=0,
                session_id="session-123",
                citation="anatomy_example.pdf, Chapter 1, page 12",
                source_type="pdf",
                topic="Human Anatomy",
                chapter="Chapter 1",
            )
        ],
        last_conversation_topic="Brain",
        last_opened_at="2026-03-15T12:00:00+00:00",
    )


def test_index_cache_round_trip(tmp_path: Path, monkeypatch) -> None:
    """Test index cache round trip.
    
    Args:
        tmp_path (Path): Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / "document_library.json"
    monkeypatch.setattr(index_cache, "cache_directory", cache_dir)
    monkeypatch.setattr(index_cache, "document_library_path", cache_file)

    indexed_document = _build_indexed_document()
    workspace = {
        "document_id": indexed_document.document_id,
        "session_id": indexed_document.session_id,
        "filename": indexed_document.filename,
        "document_title": indexed_document.document_title,
        "document_topic": indexed_document.document_topic,
        "document_summary": indexed_document.document_summary,
        "chunks": indexed_document.chunks,
        "size_mb": indexed_document.size_mb,
        "chunk_count": len(indexed_document.chunks),
        "last_conversation_topic": indexed_document.last_conversation_topic,
        "last_opened_at": indexed_document.last_opened_at,
        "messages": [],
        "current_mode": "ask",
        "conversation_topic": None,
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
    index_cache.persist_document_library([workspace], indexed_document.document_id)
    restored_library, restored_active_document_id = index_cache.restore_document_library()

    assert restored_active_document_id == indexed_document.document_id
    assert restored_library[0]["document_id"] == indexed_document.document_id
    assert restored_library[0]["filename"] == indexed_document.filename
    assert restored_library[0]["document_title"] == indexed_document.document_title
    assert restored_library[0]["document_topic"] == indexed_document.document_topic
    assert restored_library[0]["chunks"] == indexed_document.chunks


def test_index_cache_returns_none_for_invalid_payload(tmp_path: Path, monkeypatch) -> None:
    """Test index cache returns none for invalid payload.
    
    Args:
        tmp_path (Path): Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / "document_library.json"
    cache_dir.mkdir()
    cache_file.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(index_cache, "cache_directory", cache_dir)
    monkeypatch.setattr(index_cache, "document_library_path", cache_file)

    assert index_cache.restore_document_library() == ([], None)


def test_index_cache_restores_legacy_single_document_payload(tmp_path: Path, monkeypatch) -> None:
    """Test index cache restores legacy single document payload.
    
    Args:
        tmp_path (Path): Test fixture or parameter.
        monkeypatch: Test fixture or parameter.
    """

    cache_dir = tmp_path / "cache"
    library_file = cache_dir / "document_library.json"
    legacy_file = cache_dir / "last_indexed_document.json"
    cache_dir.mkdir()
    legacy_file.write_text(
        """
        {
              "session_id": "legacy-session",
              "filename": "anatomy_example.pdf",
              "chunks": [
            {
              "id": "legacy-session-0",
              "text": "The brain is the control center.",
              "filename": "anatomy_example.pdf",
              "page": 12,
              "chunk_id": 0,
              "session_id": "legacy-session",
              "citation": "anatomy_example.pdf, page 12",
              "source_type": "pdf",
              "topic": null,
              "chapter": null
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(index_cache, "cache_directory", cache_dir)
    monkeypatch.setattr(index_cache, "document_library_path", library_file)
    monkeypatch.setattr(index_cache, "legacy_indexed_document_path", legacy_file)

    restored_library, active_document_id = index_cache.restore_document_library()

    assert active_document_id == "legacy-doc"
    assert len(restored_library) == 1
    assert restored_library[0]["filename"] == "anatomy_example.pdf"
    assert restored_library[0]["document_title"] == "Anatomy Example"
    assert restored_library[0]["document_topic"] == "Anatomy Example"
    assert restored_library[0]["chunk_count"] == 1
