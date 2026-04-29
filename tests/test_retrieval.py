"""test_retrieval.py: Tests for test_retrieval.py."""

from src.core.models import Chunk
from src.data.retrieval import retrieve_chunks


def test_retrieve_chunks_returns_best_match_for_session() -> None:
    """Test retrieve chunks returns best match for session.
    """

    chunks = [
        Chunk(
            id="a",
            text="Photosynthesis uses sunlight to make glucose.",
            filename="bio.txt",
            page=1,
            chunk_id=0,
            session_id="session-a",
            citation="bio.txt, page 1",
            source_type="txt",
        ),
        Chunk(
            id="b",
            text="The mitochondria generate ATP through cellular respiration.",
            filename="bio.txt",
            page=2,
            chunk_id=1,
            session_id="session-b",
            citation="bio.txt, page 2",
            source_type="txt",
        ),
    ]

    results = retrieve_chunks("How does photosynthesis use sunlight?", chunks, session_id="session-a")

    assert len(results) == 1
    assert results[0].citation == "bio.txt, page 1"


def test_retrieve_chunks_falls_back_to_loaded_chunks_when_session_and_document_metadata_drift() -> None:
    """Local retrieval should still work when metadata no longer matches current session/document."""

    chunks = [
        Chunk(
            id="a",
            text="Brain states include wakefulness and different sleep stages.",
            filename="neuro.txt",
            page=3,
            chunk_id=0,
            session_id="legacy-session",
            citation="neuro.txt, page 3",
            source_type="txt",
            document_id=None,
        ),
        Chunk(
            id="b",
            text="The medulla helps regulate breathing and heart rate.",
            filename="neuro.txt",
            page=4,
            chunk_id=1,
            session_id="legacy-session",
            citation="neuro.txt, page 4",
            source_type="txt",
            document_id=None,
        ),
    ]

    results = retrieve_chunks(
        "what are brain states?",
        chunks,
        session_id="current-session",
        document_id="current-document",
    )

    assert results
    assert any("brain states" in chunk.text.lower() for chunk in results)
