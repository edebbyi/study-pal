# Configuration

Study Pal reads configuration in this order:

1. Environment variables
2. `.env` (local)
3. `.streamlit/secrets.toml` (Streamlit Cloud)

## Core Keys

- `SUPABASE_URL`
- `SUPABASE_PUBLIC_KEY`
- `SUPABASE_REDIRECT_URL`
- `OPENROUTER_KEY_ENCRYPTION_SECRET`
- `DATABASE_URL`
- `PINECONE_API_KEY`
- `PINECONE_HOST`
- `PINECONE_INDEX_NAME` (default: `study-pal`)

## OpenRouter Runtime Keys

- Preferred: users save personal keys in **Settings**.
- Optional fallback: set `OPENROUTER_API_KEY` and `OPENROUTER_ALLOW_GLOBAL_FALLBACK=true`.
- If no user key exists and fallback is disabled, model calls are blocked.

## Key Save Requirements

Per-user key save/delete requires:

- `DATABASE_URL`
- `OPENROUTER_KEY_ENCRYPTION_SECRET`

Save behavior:

- Key format is checked (`sk-or-v1-...`).
- Key is validated against OpenRouter (`/models`) before persisting.
- Key is encrypted before writing to Postgres.

## Optional Keys

- `OPENROUTER_CHAT_MODEL`
- `OPENROUTER_EMBEDDING_MODEL`
- `OPENROUTER_BASE_URL`
- `OPENROUTER_RERANK_MODEL`
- `OPENROUTER_RERANK_CANDIDATES`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_BASE_URL`
- `LANGFUSE_PROMPT_*`

## Apply Schema

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

## Restart Note

Restart Streamlit after changing `.env` or `.streamlit/secrets.toml`.
