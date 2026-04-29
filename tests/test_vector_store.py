"""test_vector_store.py: Tests for test_vector_store.py."""

from __future__ import annotations

import builtins

from src.data.vector_store import rebuild_document_library_from_remote
import src.data.vector_store as vector_store
from src.core.models import Chunk


class FakeVector:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class FakeFetchResponse:
    def __init__(self, vectors: dict[str, FakeVector]) -> None:
        self.vectors = vectors


class FakeIndex:
    def list(self, limit: int = 100):
        """List.
        
        Args:
            limit (int): Test fixture or parameter.
        """

        yield ["session-1-0", "session-1-1"]

    def fetch(self, ids: builtins.list[str]):
        """Fetch.
        
        Args:
            ids (list[str]): Test fixture or parameter.
        """

        return FakeFetchResponse(
            {
                "session-1-0": FakeVector(
                    {
                        "text": "The brain is part of the nervous system.",
                        "filename": "anatomy_example.pdf",
                        "page": 12,
                        "chunk_id": 0,
                        "session_id": "session-1",
                        "citation": "anatomy_example.pdf, Chapter 1, page 12",
                        "source_type": "pdf",
                        "document_id": "doc-1",
                        "document_title": "Anatomy Example",
                        "document_summary": "A survey of anatomy.",
                        "topic": "Human Anatomy",
                        "chapter": "Chapter 1",
                    }
                ),
                "session-1-1": FakeVector(
                    {
                        "text": "The cerebellum coordinates movement.",
                        "filename": "anatomy_example.pdf",
                        "page": 13,
                        "chunk_id": 1,
                        "session_id": "session-1",
                        "citation": "anatomy_example.pdf, Chapter 1, page 13",
                        "source_type": "pdf",
                        "document_id": "doc-1",
                        "document_title": "Anatomy Example",
                        "document_summary": "A survey of anatomy.",
                        "topic": "Human Anatomy",
                        "chapter": "Chapter 1",
                    }
                ),
            }
        )

    def query(self, vector: builtins.list[float], top_k: int, include_metadata: bool, filter: dict):
        class _Match:
            def __init__(self) -> None:
                self.score = 0.91
                self.metadata = {
                    "text": "Brain states include wakefulness and sleep.",
                    "filename": "neuro.pdf",
                    "page": 2,
                    "chunk_id": 0,
                    "citation": "neuro.pdf, page 2",
                    "topic": "Neuroscience",
                }

        class _Response:
            def __init__(self) -> None:
                self.matches = [_Match()]

        self.last_filter = filter
        return _Response()

    def upsert(self, vectors: builtins.list[dict]) -> None:
        self.last_upsert_vectors = vectors


class FakeFailingUpsertIndex(FakeIndex):
    def upsert(self, vectors: builtins.list[dict]) -> None:
        raise vector_store.PineconeApiException(status=503, reason="Service Unavailable")


def test_rebuild_document_library_from_remote_groups_chunks_into_workspace(monkeypatch) -> None:
    """Test rebuild document library from remote groups chunks into workspace.
    
    Args:
        monkeypatch: Test fixture or parameter.
    """

    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: FakeIndex())

    workspaces = rebuild_document_library_from_remote()

    assert len(workspaces) == 1
    assert workspaces[0]["document_id"] == "doc-1"
    assert workspaces[0]["document_title"] == "Anatomy Example"
    assert workspaces[0]["document_topic"] == "Human Anatomy"
    assert workspaces[0]["chunk_count"] == 2


def test_query_remote_chunks_filters_by_document_and_user_without_session(monkeypatch) -> None:
    """Document-scoped retrieval should not require session-id match."""
    fake_index = FakeIndex()
    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: fake_index)

    results = vector_store.query_remote_chunks(
        {"values": [0.1, 0.2, 0.3]},
        session_id="new-session-id",
        top_k=4,
        document_id="doc-1",
        user_id="user-1",
    )

    assert len(results) == 1
    assert fake_index.last_filter == {
        "$and": [
            {"document_id": {"$eq": "doc-1"}},
            {"user_id": {"$eq": "user-1"}},
        ]
    }


def test_query_remote_chunks_returns_empty_when_document_scope_has_no_user(monkeypatch) -> None:
    """Document-scoped retrieval requires user scope in multitenant mode."""
    fake_index = FakeIndex()
    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: fake_index)

    results = vector_store.query_remote_chunks(
        {"values": [0.1, 0.2, 0.3]},
        session_id="session-x",
        top_k=4,
        document_id="doc-1",
        user_id=None,
    )

    assert results == []


def test_upsert_remote_chunks_returns_true_on_success(monkeypatch) -> None:
    """Upsert should report success when Pinecone accepts payload."""
    fake_index = FakeIndex()
    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: fake_index)
    chunks = [
        Chunk(
            id="c-1",
            text="Brain states include wakefulness and sleep.",
            filename="neuro.pdf",
            page=1,
            chunk_id=0,
            session_id="s-1",
            citation="neuro.pdf, page 1",
            source_type="pdf",
            document_id="doc-1",
            user_id="user-1",
        )
    ]
    vectors = [{"values": [0.1, 0.2, 0.3]}]

    success = vector_store.upsert_remote_chunks(chunks, vectors)

    assert success is True


def test_upsert_remote_chunks_returns_false_on_pinecone_error(monkeypatch) -> None:
    """Upsert should report failure when Pinecone rejects payload."""
    fake_index = FakeFailingUpsertIndex()
    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: fake_index)
    chunks = [
        Chunk(
            id="c-1",
            text="Brain states include wakefulness and sleep.",
            filename="neuro.pdf",
            page=1,
            chunk_id=0,
            session_id="s-1",
            citation="neuro.pdf, page 1",
            source_type="pdf",
            document_id="doc-1",
            user_id="user-1",
        )
    ]
    vectors = [{"values": [0.1, 0.2, 0.3]}]

    success = vector_store.upsert_remote_chunks(chunks, vectors)

    assert success is False
