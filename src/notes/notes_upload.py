"""notes_upload.py: Index uploaded notes into chunks, metadata, and embeddings."""

from __future__ import annotations

import streamlit as st

from src.core.app_state import IndexedDocument
from src.data.chunking import chunk_document
from src.data.document_metadata import detect_chapters, extract_document_metadata
from src.data.embeddings import embed_texts, is_embedding_vector
from src.data.ingestion import build_document
from src.core.utils import generate_document_id
from src.data.vector_store import upsert_remote_chunks



def index_uploaded_file(uploaded_file) -> IndexedDocument:
    """Turn an uploaded file into an indexed document workspace.

    Args:
        uploaded_file: Streamlit file handle for the uploaded notes.

    Returns:
        IndexedDocument: Indexed document metadata and chunks.
    """

    document_id = generate_document_id()
    uploaded_file.seek(0, 2)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    document = build_document(uploaded_file, st.session_state.session_id)
    document_metadata = extract_document_metadata(document)
    user_id = st.session_state.user_id
    chunks = chunk_document(
        document,
        document_id=document_id,
        document_title=document_metadata.document_title,
        document_summary=document_metadata.document_summary,
        document_topic=document_metadata.document_topic,
        user_id=user_id,
        chapters_by_page=detect_chapters(document),
    )
    embeddings = embed_texts([chunk.text for chunk in chunks])
    remote_embeddings = [embedding for embedding in embeddings if is_embedding_vector(embedding)]

    remote_sync_success = False
    # Only push to Pinecone when every chunk has a real vector payload.
    if len(remote_embeddings) == len(chunks):  # Only upsert when every chunk has a real vector.
        remote_sync_success = upsert_remote_chunks(chunks, remote_embeddings)
    if len(remote_embeddings) != len(chunks):
        st.warning(
            "Indexed locally, but Pinecone sync was skipped because embeddings were unavailable."
        )
    elif not remote_sync_success:
        st.warning(
            "Indexed locally, but Pinecone sync failed. Use 'Reindex to Pinecone' to retry."
        )

    return IndexedDocument(
        document_id=document_id,
        session_id=document.session_id,
        user_id=user_id,
        filename=document.filename,
        document_title=document_metadata.document_title,
        document_topic=document_metadata.document_topic,
        document_summary=document_metadata.document_summary,
        key_hooks=document_metadata.key_hooks,
        chunks=chunks,
        size_mb=round(size_mb, 1),
    )
