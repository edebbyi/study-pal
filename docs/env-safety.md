# Env Safety: Never Overwrite `.env`

This project treats `.env` as user-owned state. It may contain private API keys
and local-only configuration. **Never overwrite, replace, or clear `.env`** in
automation or assistant workflows.

## Hard Rules

- **Do not overwrite `.env`.** If it exists, leave it untouched.
- **Do not delete `.env`.** Ever.
- **Do not auto-fill `.env` from `.env.example`.** That can erase real values.
- **Never commit `.env`.** It must remain local and private.

## Safe Alternatives

- Use `.env.example` for defaults and documentation.
- If `.env` is missing, **create a new file** only after explicit user approval.
- Prefer **adding new keys** (append only) with user consent instead of rewriting.

## Approved Workflow

1. Check if `.env` exists.
2. If it exists, **stop** and ask the user before any changes.
3. If it does not exist, ask the user whether to:
   - copy `.env.example` as a starting point, or
   - create a minimal `.env` with only the required keys.

## Recovery Guidance

If `.env` was overwritten:

- Try restoring from `.streamlit/secrets.toml` if it contains the same values.
- Check shell history or secrets manager for previous values.
- Ask the user before writing anything back to `.env`.

## Reminder for Assistants

If you are an automated agent or LLM: **you must never erase contents from
`.env`**. Treat it as sensitive user data and only modify it with explicit,
recorded consent.
