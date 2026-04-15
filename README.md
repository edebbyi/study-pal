# Study Pal

Study Pal is a Streamlit study app for note-grounded Q&A and guided mastery loops.

- Ask questions against uploaded notes with citations.
- Run a bounded mastery loop (quiz -> grade -> reteach -> reinforce -> study plan).
- Track traces/prompts in Langfuse (optional).
- Store feedback in Postgres when configured, with local fallback.

## Walkthrough

![Study Pal demo](docs/demo.gif)

### Ask Mode
![Ask Mode](docs/ask-mode.gif)

### Mastery Mode
![Mastery Mode](docs/mastery-mode.gif)

## Quick Start (Local)

### Prerequisites

- Python 3.11
- `pip`
- OpenRouter API key
- Pinecone index + host (for remote vector retrieval)

### 1) Clone and install

```bash
git clone <your-repo-url>
cd StudyPal
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

### 2) Configure settings

Use either `.env` (local dev) or `.streamlit/secrets.toml` (Streamlit Cloud).  
Environment variables override `secrets.toml` locally.

Minimum keys to run core app:

```toml
OPENROUTER_API_KEY = "..."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_API_KEY = "..."
PINECONE_HOST = "https://<your-index-host>.pinecone.io"
PINECONE_INDEX_NAME = "studypal"
```

If you want magic-link login:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_PUBLIC_KEY = "sb_publishable_..."
SUPABASE_REDIRECT_URL = "http://localhost:8501"
```

Optional (recommended for production):

```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
OPENROUTER_RERANK_MODEL = "cohere/rerank-4-pro"
OPENROUTER_RERANK_CANDIDATES = "12"
```

Reference template: `.streamlit/secrets.example.toml`

Complete `secrets.toml` example:

```toml
OPENROUTER_API_KEY = ""
OPENROUTER_CHAT_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-3-small"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_RERANK_MODEL = "cohere/rerank-4-pro"
OPENROUTER_RERANK_CANDIDATES = "12"
EMBEDDING_DIMENSIONS = "512"

STUDYPAL_CHUNK_SIZE = "900"
STUDYPAL_CHUNK_OVERLAP = "150"
STUDYPAL_TOP_K = "4"
STUDYPAL_QUIZ_QUESTIONS = "3"
STUDYPAL_MAX_QUIZ_ROUNDS = "3"
STUDYPAL_MAX_CHAT_TOKENS = "600"
STUDYPAL_MAX_FILE_SIZE_MB = "150"

SUPABASE_URL = ""
SUPABASE_PUBLIC_KEY = ""
SUPABASE_REDIRECT_URL = "http://localhost:8501"

PINECONE_API_KEY = ""
PINECONE_HOST = ""
PINECONE_INDEX_NAME = "studypal"

DATABASE_URL = ""

LANGFUSE_SECRET_KEY = ""
LANGFUSE_PUBLIC_KEY = ""
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
LANGFUSE_PROMPT_ANSWER = "study_pal_answer"
LANGFUSE_PROMPT_STRUCTURED_ANSWER = "study_pal_structured_answer"
LANGFUSE_PROMPT_DOCUMENT_METADATA = "study_pal_document_metadata"
LANGFUSE_PROMPT_QUIZ = "study_pal_quiz"
LANGFUSE_PROMPT_RETEACH = "study_pal_reteach"
LANGFUSE_PROMPT_STUDY_PLAN = "study_pal_study_plan"
LANGFUSE_PROMPT_FOLLOW_UP = "study_pal_follow_up"
LANGFUSE_PROMPT_VERSION = ""
```

### 3) Run

```bash
make run
```

App starts at `http://localhost:8501`.

## Quick Start (Docker)

```bash
docker compose up --build
```

or with Make targets:

```bash
make dev
```

Useful dev commands:

```bash
make dev-up      # recreate/start existing image/container
make dev-down    # stop and remove containers/network
```

## Configuration Notes

- `PINECONE_INDEX_NAME` must match your real index name exactly.
- `SUPABASE_REDIRECT_URL` must match the URL users click back into:
  - local: `http://localhost:8501`
  - cloud: `https://<your-app>.streamlit.app`
- Keep secrets out of git (`.env`, `.streamlit/secrets.toml` are ignored).

## Database Setup (Feedback)

If you set `DATABASE_URL`, apply schema before running:

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

If DB is not configured, app falls back to local SQLite/JSONL storage.

## Langfuse Prompt Seeding (Optional)

```bash
python scripts/seed_langfuse_prompts.py
```

Force-update prompt versions:

```bash
python scripts/seed_langfuse_prompts.py --force --label production
```

## Streamlit Cloud Deployment

1. Push repo to GitHub.
2. Create Streamlit app pointing to `app.py`.
3. Paste your config into Streamlit **Secrets**.
4. Set Supabase auth URLs:
   - Site URL: current app URL
   - Redirect URLs: local + production URL
5. Deploy.

Magic-link template should resolve back to your app URL with token params, for example:

```html
<a href="http://localhost:8501/?token_hash={{ .TokenHash }}&type=magiclink&email={{ .Email }}">
  Log In
</a>
```

Use your Streamlit Cloud URL instead of localhost in production.

## Development

```bash
make test
make lint
```

CI runs Ruff, Mypy, and Pytest on pushes and pull requests.

## Project Structure

- `app.py`: Streamlit UI + controller
- `src/notes/`: upload/index + answer orchestration
- `src/data/`: ingestion, chunking, retrieval, vector/cache
- `src/llm/`: prompts + model client
- `src/modes/`: ask/mastery router + quiz/grading/remediation/planning
- `src/feedback/`: feedback persistence
- `src/auth/`: Supabase magic-link helpers
- `db/schema.sql`: Postgres schema

## Limitations

- Answers are limited to uploaded notes.
- Retrieval quality depends on note quality/chunking.
- Mastery loop is intentionally bounded.
- Citations are chunk-level (not exact sentence spans).

## Next Version Focus

- Richer reteach interface and interaction flow in Mastery Mode.
- Higher-quality reteach content with clearer misconception contrast.

## License

MIT
