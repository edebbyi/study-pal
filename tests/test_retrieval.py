from src.models import Chunk
from src.retrieval import retrieve_chunks


def test_retrieve_chunks_returns_best_match_for_session() -> None:
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
