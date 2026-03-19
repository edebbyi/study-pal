# Study Pal

Study Pal is a Streamlit study coach that lets you upload notes, ask cited questions, and run a bounded mastery loop with quizzes, reteaching, and a final study plan.

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

- [app.py](/Users/esosa/Documents/GitHub/StudyPal/app.py): Streamlit controller and UI
- [src/config.py](/Users/esosa/Documents/GitHub/StudyPal/src/config.py): typed settings loader
- [src/notes_upload.py](/Users/esosa/Documents/GitHub/StudyPal/src/notes_upload.py): upload and indexing workflow
- [src/notes_answering.py](/Users/esosa/Documents/GitHub/StudyPal/src/notes_answering.py): Ask Mode answer building and citation lookup
- [src/mode_router.py](/Users/esosa/Documents/GitHub/StudyPal/src/mode_router.py): Ask vs Mastery routing
- [src/mastery.py](/Users/esosa/Documents/GitHub/StudyPal/src/mastery.py): mastery session and progression rules
- [src/agent.py](/Users/esosa/Documents/GitHub/StudyPal/src/agent.py): bounded loop orchestration
- [src/quiz.py](/Users/esosa/Documents/GitHub/StudyPal/src/quiz.py): quiz generation and validation
- [src/grading.py](/Users/esosa/Documents/GitHub/StudyPal/src/grading.py): quiz grading
- [src/remediation.py](/Users/esosa/Documents/GitHub/StudyPal/src/remediation.py): reteaching generation and fallback
- [src/planning.py](/Users/esosa/Documents/GitHub/StudyPal/src/planning.py): study-plan generation and fallback
- [src/llm_client.py](/Users/esosa/Documents/GitHub/StudyPal/src/llm_client.py): OpenRouter-backed generation
- [src/vector_store.py](/Users/esosa/Documents/GitHub/StudyPal/src/vector_store.py): Pinecone operations

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
make install
```

3. Add secrets in `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "studypal"
DATABASE_URL = "postgresql://user:password@host:5432/studypal"
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

5. Optional but recommended for production: apply the feedback schema from [db/schema.sql](/Users/esosa/Documents/GitHub/StudyPal/db/schema.sql) to your Postgres database before starting the app.

4. Run the app:

```bash
make run
```

## Development

Run tests:

```bash
make test
```

Run linting:

```bash
make lint
```

## Demo Flow

1. Upload one of the sample files from [data_samples/](/Users/esosa/Documents/GitHub/StudyPal/data_samples).
2. Ask a normal question about the notes to use Ask Mode.
3. Enter a mastery-style prompt such as `Help me study photosynthesis`.
4. Complete the quiz and review the reteaching feedback.
5. Continue until the app generates a study plan, or use `Stop and build study plan`.

## Current Notes

- The mastery loop is bounded in code, not autonomous in an open-ended sense.
- Generated quiz, remediation, and study-plan content is validated before use and falls back safely when needed.
- Integration-style app flow tests cover Ask Mode, Mastery start, progression, completion, and stop behavior.
- Response feedback is stored in Postgres when `DATABASE_URL` is configured, with local SQLite/JSONL fallback for development.
- The sidebar includes a small `Feedback Admin` page for browsing recent saved feedback records.
