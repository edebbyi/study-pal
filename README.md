# Study Pal

Study Pal is a Streamlit study coach that lets you upload notes, ask cited questions, and run a bounded mastery loop with quizzes, reteaching, and a final study plan.

## Demo

![Study Pal demo](docs/demo.gif)

## What It Does

- Upload `pdf`, `txt`, or `md` notes
- Chunk and embed note content
- Retrieve relevant note passages for Ask Mode answers
- Show source citations for Ask Mode and Mastery Mode outputs
- Run a bounded mastery loop:
  - introduce a topic
  - generate a quiz
  - grade answers
  - reteach weak concepts
  - generate a reinforcement quiz
  - stop on mastery, stop action, or max rounds
  - build a study plan

## Architecture

- [app.py](app.py): Streamlit controller and UI
- [src/core/config.py](src/core/config.py): typed settings loader
- [src/notes/notes_upload.py](src/notes/notes_upload.py): upload and indexing workflow
- [src/notes/notes_answering.py](src/notes/notes_answering.py): Ask Mode answer building and citation lookup
- [src/modes/mode_router.py](src/modes/mode_router.py): Ask vs Mastery routing
- [src/modes/mastery.py](src/modes/mastery.py): mastery session and progression rules
- [src/modes/agent.py](src/modes/agent.py): bounded loop orchestration
- [src/modes/quiz.py](src/modes/quiz.py): quiz generation and validation
- [src/modes/grading.py](src/modes/grading.py): quiz grading
- [src/modes/remediation.py](src/modes/remediation.py): reteaching generation and fallback
- [src/modes/planning.py](src/modes/planning.py): study-plan generation and fallback
- [src/llm/llm_client.py](src/llm/llm_client.py): OpenRouter-backed generation
- [src/data/vector_store.py](src/data/vector_store.py): Pinecone operations

## Architecture Overview

At a high level, the app moves from upload to retrieval to tutoring, then optionally into the mastery loop.

```
Upload notes
  -> Ingestion + chunking
  -> Embeddings + Pinecone (optional)
  -> Workspace cache

Ask Mode
  -> Retrieve relevant chunks
  -> LLM answer from notes
  -> Citations

Mastery Mode
  -> Topic detection
  -> Quiz generation
  -> Grading
  -> Reteach weak concepts
  -> Reinforcement quiz
  -> Study plan
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
make install
```

3. Add secrets in `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "..."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_API_KEY = "..."
PINECONE_HOST = "https://studypal-3w8u9cl.svc.aped-4627-b74a.pinecone.io"
PINECONE_INDEX_NAME = "studypal"
DATABASE_URL = "postgresql://user:password@host:5432/studypal"
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
LANGFUSE_PROMPT_ANSWER = "study_pal_answer"
LANGFUSE_PROMPT_STRUCTURED_ANSWER = "study_pal_structured_answer"
LANGFUSE_PROMPT_DOCUMENT_METADATA = "study_pal_document_metadata"
LANGFUSE_PROMPT_QUIZ = "study_pal_quiz"
LANGFUSE_PROMPT_RETEACH = "study_pal_reteach"
LANGFUSE_PROMPT_STUDY_PLAN = "study_pal_study_plan"
LANGFUSE_PROMPT_FOLLOW_UP = "study_pal_follow_up"
LANGFUSE_PROMPT_VERSION = ""
OPENROUTER_RERANK_MODEL = ""
OPENROUTER_RERANK_CANDIDATES = "12"
```

> Note: `.streamlit/secrets.toml` is git-ignored on purpose. Use
> `.streamlit/secrets.example.toml` as a starting point.

Optional: seed Langfuse prompt templates from the repository defaults:

```bash
python scripts/seed_langfuse_prompts.py
```

5. Optional but recommended for production: apply the feedback schema from [db/schema.sql](db/schema.sql) to your Postgres database before starting the app.

4. Run the app:

```bash
make run
```

## Docker

Build and run with Docker:

```bash
docker build -t studypal .
docker run -p 8501:8501 --env-file .env studypal
```

Or use Docker Compose:

```bash
docker compose up --build
```

Copy `.env.example` to `.env` and fill in values, or mount
`.streamlit/secrets.toml` into the container if you prefer that format.

## Streamlit Cloud

Deploy on Streamlit Cloud:

1. Push this repo to GitHub.
2. Create a new app in Streamlit Cloud and point it at `app.py`.
3. In **App settings → Secrets**, paste the same keys you would put in `.streamlit/secrets.toml`.
4. Deploy to get a public URL like `https://studypal-<random>.streamlit.app`.

## Development

Run tests:

```bash
make test
```

Run linting:

```bash
make lint
```

## CI

GitHub Actions runs on every push and pull request and executes:

- Ruff (`ruff check .`)
- Mypy (`mypy .`)
- Pytest (`pytest`)

The project uses the repository-root `ruff.toml` for lint configuration.

## Demo Flow

1. Upload one of the sample files from [data_samples/](data_samples).
2. Ask a normal question about the notes to use Ask Mode.
3. Enter a mastery-style prompt such as `Help me study photosynthesis`.
4. Complete the quiz and review the reteaching feedback.
5. Continue until the app generates a study plan, or use `Stop and build study plan`.

## Limitations

- Responses are limited to the uploaded notes; the app does not browse the web.
- Retrieval quality depends on chunking and the notes' clarity.
- The mastery loop is intentionally bounded to avoid endless cycles.
- No user authentication or multi-user storage is included in this version.
- Citations are chunk-level and may not match exact sentence boundaries.

## Current Notes

- The mastery loop is bounded in code, not autonomous in an open-ended sense.
- Generated quiz, remediation, and study-plan content is validated before use and falls back safely when needed.
- Integration-style app flow tests cover Ask Mode, Mastery start, progression, completion, and stop behavior.
- Response feedback is stored in Postgres when `DATABASE_URL` is configured, with local SQLite/JSONL fallback for development.
- The sidebar includes a small `Feedback Admin` page for browsing recent saved feedback records.

## Secrets

Sensitive values should live in `.streamlit/secrets.toml` or environment
variables. The secrets file is ignored by git by default to keep keys out of
version control.
