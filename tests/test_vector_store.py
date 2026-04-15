"""test_vector_store.py: Tests for test_vector_store.py."""

from __future__ import annotations

from src.data.vector_store import rebuild_document_library_from_remote
import src.data.vector_store as vector_store


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

    def fetch(self, ids: list[str]):
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
