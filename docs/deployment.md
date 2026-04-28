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

## Streamlit Cloud

1. Push repo to GitHub.
2. Create a Streamlit app pointing to `app.py`.
3. Add secrets in Streamlit `Settings -> Secrets`.
4. Configure Supabase auth URLs to match your Streamlit app URL.
5. Deploy.

For full key definitions, see [`configuration.md`](configuration.md). For auth redirect setup, see [`auth_setup.md`](auth_setup.md).

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
