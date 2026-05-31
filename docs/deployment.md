# Deployment

## Local

```bash
make install
make run
```

## Docker

```bash
make dev
make dev-up
make dev-down
```

Run only the API container locally:

```bash
make docker-api-build
make docker-api-run
```

API health check:

```bash
curl http://localhost:8000/api/health
```

## Streamlit Cloud

1. Push repo to GitHub.
2. Create a Streamlit app pointing to `app.py`.
3. Add secrets in Streamlit `Settings -> Secrets`.
4. Configure Supabase auth URLs to match your Streamlit app URL.
5. Deploy.

For full key definitions, see [`configuration.md`](configuration.md). For auth redirect setup, see [`auth_setup.md`](auth_setup.md).

## Render (FastAPI backend, recommended)

1. Push the repo (including `Dockerfile.api`, `requirements.api.txt`, and `render.yaml`) to GitHub.
2. In Render, click `New +` -> `Blueprint` and select this repository.
3. Confirm service name `studypal-api` and deploy.
4. In Render service settings, set secret env vars:

- `OPENROUTER_API_KEY`
- `OPENROUTER_KEY_ENCRYPTION_SECRET`
- `PINECONE_API_KEY`
- `PINECONE_HOST`
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_PUBLIC_KEY`

5. Verify API health:

```bash
curl https://<your-render-service>.onrender.com/api/health
```

6. Update Streamlit secrets:

- `STUDYPAL_API_BASE_URL=https://<your-render-service>.onrender.com/api`

7. Prevent free-tier cold starts by pinging every 5-10 minutes:

- Monitor URL: `https://<your-render-service>.onrender.com/api/health`
- Method: `GET`
- Expected status: `200`

## Cloud Run (optional)

Use Cloud Shell in GCP if `gcloud` is not installed locally.

1. Enable APIs:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

2. Create an Artifact Registry Docker repo (one-time):

```bash
gcloud artifacts repositories create studypal \
  --repository-format=docker \
  --location=us-central1 \
  --description="StudyPal containers"
```

3. Build and push API image from repo root:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/study-pal-497721/studypal/studypal-api:latest \
  -f Dockerfile.api .
```

4. Deploy Cloud Run service:

```bash
gcloud run deploy studypal-api \
  --image us-central1-docker.pkg.dev/study-pal-497721/studypal/studypal-api:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

5. Set runtime env vars on Cloud Run (minimum):

- `OPENROUTER_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_HOST`
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_PUBLIC_KEY`
- `SUPABASE_REDIRECT_URL` (Streamlit URL)
- `OPENROUTER_KEY_ENCRYPTION_SECRET`

6. Update Streamlit secrets:

- `STUDYPAL_API_BASE_URL=https://<your-cloud-run-url>/api`

## Required Secrets for Cloud

- `SUPABASE_URL`
- `SUPABASE_PUBLIC_KEY`
- `SUPABASE_REDIRECT_URL` (your Streamlit app URL)
- `OPENROUTER_KEY_ENCRYPTION_SECRET`
- `DATABASE_URL`
- `PINECONE_API_KEY`
- `PINECONE_HOST`

## Post-Deploy Checks

- Magic link returns to the same app URL.
- Settings page can save and delete a per-user key.
- Upload, ask-mode response, and mastery loop complete successfully.
