from __future__ import annotations

from collections import defaultdict

from pinecone import Pinecone, PineconeApiException, PineconeConfigurationError, PineconeProtocolError

from src.config import settings
from src.embeddings import EmbeddingVector
from src.models import Chunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        existing_ids = {chunk.id for chunk in self._chunks}
        for chunk in chunks:
            if chunk.id not in existing_ids:
                self._chunks.append(chunk)

    def query(self, session_id: str) -> list[Chunk]:
        return [chunk for chunk in self._chunks if chunk.session_id == session_id]


def normalize_retrieved_chunks(scored_chunks: list[tuple[Chunk, float]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=chunk.text,
            filename=chunk.filename,
            page=chunk.page,
            citation=chunk.citation,
            score=score,
            chunk_id=chunk.chunk_id,
            chapter=chunk.chapter,
            topic=chunk.topic,
        )
        for chunk, score in scored_chunks
    ]


def get_pinecone_index():
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        return None
    client = Pinecone(api_key=settings.pinecone_api_key)
    return client.Index(settings.pinecone_index_name)


def upsert_remote_chunks(chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
    index = get_pinecone_index()
    if index is None or not chunks:
        return

    payload = []
    for chunk, vector_payload in zip(chunks, vectors, strict=False):
        payload.append(
            {
                "id": chunk.id,
                "values": vector_payload["values"],
                "metadata": {
                    "text": chunk.text,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "session_id": chunk.session_id,
                    "citation": chunk.citation,
                    "source_type": chunk.source_type,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "document_summary": chunk.document_summary,
                    "chapter": chunk.chapter,
                    "topic": chunk.topic,
                },
            }
        )

    try:
        index.upsert(vectors=payload)
    except (PineconeApiException, PineconeConfigurationError, PineconeProtocolError):
        return


def query_remote_chunks(query_vector: EmbeddingVector, session_id: str, top_k: int) -> list[RetrievedChunk]:
    index = get_pinecone_index()
    if index is None:
        return []

    try:
        response = index.query(
            vector=query_vector["values"],
            top_k=top_k,
            include_metadata=True,
            filter={"session_id": {"$eq": session_id}},
        )
    except (PineconeApiException, PineconeConfigurationError, PineconeProtocolError):
        return []

    # Normalize SDK objects into our own retrieval shape before leaving the module.
    matches = []
    for match in response.matches:
        metadata = match.metadata or {}
        matches.append(
            RetrievedChunk(
                text=str(metadata.get("text", "")),
                filename=str(metadata.get("filename", "")),
                page=int(metadata.get("page", 0)),
                citation=str(metadata.get("citation", "")),
                score=float(match.score or 0.0),
                chunk_id=int(metadata.get("chunk_id", 0)),
                chapter=str(metadata.get("chapter")) if metadata.get("chapter") else None,
                topic=str(metadata.get("topic")) if metadata.get("topic") else None,
            )
        )
    return matches


def _chunk_from_metadata(vector_id: str, metadata: dict) -> Chunk:
    return Chunk(
        id=vector_id,
        text=str(metadata.get("text", "")),
        filename=str(metadata.get("filename", "")),
        page=int(metadata.get("page", 0)),
        chunk_id=int(metadata.get("chunk_id", 0)),
        session_id=str(metadata.get("session_id", "")),
        citation=str(metadata.get("citation", "")),
        source_type=str(metadata.get("source_type", "pdf")),
        document_id=str(metadata.get("document_id")) if metadata.get("document_id") else None,
        document_title=str(metadata.get("document_title")) if metadata.get("document_title") else None,
        document_summary=str(metadata.get("document_summary")) if metadata.get("document_summary") else None,
        topic=str(metadata.get("topic")) if metadata.get("topic") else None,
        chapter=str(metadata.get("chapter")) if metadata.get("chapter") else None,
    )


def rebuild_document_library_from_remote(max_vectors: int = 1000) -> list[dict[str, object]]:
    index = get_pinecone_index()
    if index is None:
        return []

    try:
        vector_ids: list[str] = []
        for id_batch in index.list(limit=100):
            vector_ids.extend(id_batch)
            if len(vector_ids) >= max_vectors:
                vector_ids = vector_ids[:max_vectors]
                break

        if not vector_ids:
            return []

        remote_chunks: list[Chunk] = []
        for start in range(0, len(vector_ids), 100):
            batch_ids = vector_ids[start : start + 100]
            fetch_response = index.fetch(ids=batch_ids)
            vectors = getattr(fetch_response, "vectors", {}) or {}
            for vector_id, vector_payload in vectors.items():
                metadata = getattr(vector_payload, "metadata", None) or {}
                if not metadata:
                    continue
                remote_chunks.append(_chunk_from_metadata(vector_id, metadata))
    except (PineconeApiException, PineconeConfigurationError, PineconeProtocolError, AttributeError, ValueError):
        return []

    grouped_chunks: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in remote_chunks:
        grouped_chunks[(chunk.session_id, chunk.filename)].append(chunk)

    workspaces: list[dict[str, object]] = []
    for (session_id, filename), chunks in grouped_chunks.items():
        first_chunk = sorted(chunks, key=lambda chunk: chunk.chunk_id)[0]
        document_title = first_chunk.document_title or filename.rsplit(".", 1)[0].replace("_", " ").title()
        document_topic = first_chunk.topic or document_title
        document_summary = first_chunk.document_summary or f"Recovered workspace for {filename} from Pinecone metadata."
        workspaces.append(
            {
                "document_id": first_chunk.document_id or f"remote-{session_id}",
                "session_id": session_id,
                "filename": filename,
                "document_title": document_title,
                "document_topic": document_topic,
                "document_summary": document_summary,
                "chunks": sorted(chunks, key=lambda chunk: chunk.chunk_id),
                "size_mb": 0.0,
                "chunk_count": len(chunks),
                "last_conversation_topic": None,
                "last_opened_at": None,
                "messages": [],
                "message_feedback": {},
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
        )

    workspaces.sort(key=lambda workspace: workspace["filename"])
    return workspaces
