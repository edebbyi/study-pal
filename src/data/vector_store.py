"""vector_store.py: Pinecone vector store helpers and remote workspace rebuilds."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from pinecone import Pinecone, PineconeApiException, PineconeConfigurationError, PineconeProtocolError

from src.core.config import settings
from src.core.models import Chunk, RetrievedChunk, SourceType
from src.data.embeddings import EmbeddingVector


class InMemoryVectorStore:
    """Store and retrieve chunks in memory for a session."""

    def __init__(self) -> None:
        """Initialize an empty in-memory chunk store."""
        self._chunks: list[Chunk] = []

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Add new chunks without duplicating existing ones.

        Args:
            chunks (list[Chunk]): Chunks to store.
        """
        existing_ids = {chunk.id for chunk in self._chunks}
        for chunk in chunks:
            if chunk.id not in existing_ids:
                self._chunks.append(chunk)

    def query(self, session_id: str) -> list[Chunk]:
        """Return all chunks matching the given session.

        Args:
            session_id (str): Session identifier to filter on.

        Returns:
            list[Chunk]: Chunks matching the session.
        """
        return [chunk for chunk in self._chunks if chunk.session_id == session_id]


def normalize_retrieved_chunks(scored_chunks: list[tuple[Chunk, float]]) -> list[RetrievedChunk]:
    """Convert scored chunk pairs into the retrieval response shape.

    Args:
        scored_chunks (list[tuple[Chunk, float]]): Chunk-score pairs from retrieval.

    Returns:
        list[RetrievedChunk]: Normalized retrieval chunks.
    """
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
    """Return a Pinecone index client when credentials are configured.

    Returns:
        Pinecone.Index | None: The configured index client or None.
    """
    if not settings.pinecone_api_key:
        return None
    client = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_host:
        try:
            return client.Index(host=settings.pinecone_host)
        except Exception:
            return None
    if not settings.pinecone_index_name:
        return None
    try:
        return client.Index(settings.pinecone_index_name)
    except Exception:
        return None


def upsert_remote_chunks(chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
    """Store chunk embeddings in the remote vector index.

    Args:
        chunks (list[Chunk]): Chunks to upsert.
        vectors (list[EmbeddingVector]): Embeddings for each chunk.
    """
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
                    "user_id": chunk.user_id,
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


def query_remote_chunks(
    query_vector: EmbeddingVector,
    session_id: str,
    top_k: int,
    *,
    document_id: str | None = None,
    user_id: str | None = None,
) -> list[RetrievedChunk]:
    """Query Pinecone and normalize results for a session or document.

    Args:
        query_vector (EmbeddingVector): Embedding payload for the query.
        session_id (str): Session identifier to filter on.
        top_k (int): Number of results to return.
        document_id (str | None): Optional document filter.

    Returns:
        list[RetrievedChunk]: Retrieved chunk results.
    """
    index = get_pinecone_index()
    if index is None:
        return []

    def _run_query(filter_payload: dict) -> list[RetrievedChunk]:
        try:
            response = index.query(
                vector=query_vector["values"],
                top_k=top_k,
                include_metadata=True,
                filter=filter_payload,
            )
        except (PineconeApiException, PineconeConfigurationError, PineconeProtocolError, Exception):
            return []
        return _normalize_matches(response.matches)

    if document_id:
        filter_payload: dict[str, object] = {"document_id": {"$eq": document_id}}
        if user_id:
            filter_payload = {"$and": [filter_payload, {"user_id": {"$eq": user_id}}]}
        if session_id:
            filter_payload = {"$and": [filter_payload, {"session_id": {"$eq": session_id}}]}
        return _run_query(filter_payload)

    if session_id:
        if user_id:
            return _run_query({"$and": [{"session_id": {"$eq": session_id}}, {"user_id": {"$eq": user_id}}]})
        return _run_query({"session_id": {"$eq": session_id}})
    return []


def _normalize_matches(matches: list[object]) -> list[RetrievedChunk]:
    """Normalize SDK matches into the RetrievedChunk shape.

    Args:
        matches (list[object]): Pinecone match objects.

    Returns:
        list[RetrievedChunk]: Normalized match objects.
    """
    normalized: list[RetrievedChunk] = []
    for match in matches:
        metadata = getattr(match, "metadata", None) or {}
        normalized.append(
            RetrievedChunk(
                text=str(metadata.get("text", "")),
                filename=str(metadata.get("filename", "")),
                page=int(metadata.get("page", 0)),
                citation=str(metadata.get("citation", "")),
                score=float(getattr(match, "score", 0.0) or 0.0),
                chunk_id=int(metadata.get("chunk_id", 0)),
                chapter=str(metadata.get("chapter")) if metadata.get("chapter") else None,
                topic=str(metadata.get("topic")) if metadata.get("topic") else None,
            )
        )
    return normalized


def _chunk_from_metadata(vector_id: str, metadata: dict) -> Chunk:
    """Build a Chunk object from Pinecone metadata.

    Args:
        vector_id (str): Vector identifier from Pinecone.
        metadata (dict): Metadata payload for the chunk.

    Returns:
        Chunk: Parsed chunk instance.
    """
    source_type_raw = metadata.get("source_type", "pdf")
    source_type: SourceType = cast(SourceType, source_type_raw if source_type_raw in {"pdf", "txt", "md"} else "pdf")
    return Chunk(
        id=vector_id,
        text=str(metadata.get("text", "")),
        filename=str(metadata.get("filename", "")),
        page=int(metadata.get("page", 0)),
        chunk_id=int(metadata.get("chunk_id", 0)),
        session_id=str(metadata.get("session_id", "")),
        citation=str(metadata.get("citation", "")),
        source_type=source_type,
        document_id=str(metadata.get("document_id")) if metadata.get("document_id") else None,
        document_title=str(metadata.get("document_title")) if metadata.get("document_title") else None,
        document_summary=str(metadata.get("document_summary")) if metadata.get("document_summary") else None,
        topic=str(metadata.get("topic")) if metadata.get("topic") else None,
        chapter=str(metadata.get("chapter")) if metadata.get("chapter") else None,
        user_id=str(metadata.get("user_id")) if metadata.get("user_id") else None,
    )


def rebuild_document_library_from_remote(
    *,
    user_id: str | None = None,
    max_vectors: int = 1000,
) -> list[dict[str, object]]:
    """Rebuild workspace records from the remote vector store.

    Args:
        max_vectors (int): Maximum number of vectors to scan.

    Returns:
        list[dict[str, object]]: Workspace summaries rebuilt from Pinecone.
    """
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

    grouped_chunks: dict[tuple[str, str, str | None], list[Chunk]] = defaultdict(list)
    for chunk in remote_chunks:
        if user_id and chunk.user_id != user_id:
            continue
        grouped_chunks[(chunk.session_id, chunk.filename, chunk.user_id)].append(chunk)

    workspaces: list[dict[str, object]] = []
    for (session_id, filename, chunk_user_id), chunks in grouped_chunks.items():
        first_chunk = sorted(chunks, key=lambda chunk: chunk.chunk_id)[0]
        document_title = first_chunk.document_title or filename.rsplit(".", 1)[0].replace("_", " ").title()
        document_topic = first_chunk.topic or document_title
        document_summary = first_chunk.document_summary
        workspaces.append(
            {
                "document_id": first_chunk.document_id or f"remote-{session_id}",
                "session_id": session_id,
                "user_id": chunk_user_id,
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

    workspaces.sort(key=lambda workspace: str(workspace.get("filename", "")))
    return workspaces
