# Supabase Auth Setup

Study Pal uses Supabase magic-link authentication.

## Required Env Vars

- `SUPABASE_URL`
- `SUPABASE_PUBLIC_KEY`
- `SUPABASE_REDIRECT_URL`

Local redirect example:

```env
SUPABASE_REDIRECT_URL=http://localhost:8501
```

## Supabase URL Configuration

In Supabase:

1. Go to `Authentication -> URL Configuration`.
2. Set **Site URL** to your app URL.
3. Add your app URL(s) in **Redirect URLs**.

Local:

- `http://localhost:8501`

Production example:

- `https://<your-app>.streamlit.app`

## Email Template Link

Use a link that returns to your app URL:

```html
<a href="http://localhost:8501/?token_hash={{ .TokenHash }}&type=magiclink&email={{ .Email }}">
  Log In
</a>
```

Replace localhost with your production app URL in production.

## Common Issues

- `otp_expired`: the link is stale. Request a new magic link.
- Redirect goes to the wrong app: ensure Site URL and Redirect URLs match exactly.
- Login page says auth is not configured: verify `SUPABASE_URL` and `SUPABASE_PUBLIC_KEY`.
