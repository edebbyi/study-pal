# Architecture Overview

## Runtime Flow

1. User logs in with Supabase magic link.
2. User uploads notes.
3. App chunks and embeds notes, then indexes them in Pinecone.
4. Queries retrieve relevant chunks.
5. OpenRouter generates grounded answers with citations.
6. Mastery mode runs quiz -> grading -> reteach -> study plan.

## Main Modules

- `app.py`: Streamlit UI and app orchestration.
- `src/auth/`: Supabase auth + per-user OpenRouter key storage.
- `src/core/`: config, session state, shared models/utilities.
- `src/data/`: ingestion, chunking, embeddings, retrieval, vector/cache.
- `src/llm/`: prompt construction and model client.
- `src/modes/`: ask/mastery router and loop components.
- `src/feedback/`: feedback persistence.
- `db/schema.sql`: Postgres schema.

## Credentials Strategy

- User-level OpenRouter key is primary.
- Optional global fallback key can be enabled by config.
- Stored user keys are encrypted before database writes.
