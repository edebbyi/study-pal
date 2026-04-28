-- schema.sql: Local feedback storage schema for Study Pal.

CREATE TABLE IF NOT EXISTS response_feedback (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,
    document_id TEXT,
    filename TEXT,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    rating TEXT NOT NULL,
    feedback_text TEXT,
    topic TEXT,
    mode TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Speed up recent feedback queries in the admin view.
CREATE INDEX IF NOT EXISTS response_feedback_created_at_idx
    ON response_feedback (created_at DESC);

-- Support quick filtering by rating.
CREATE INDEX IF NOT EXISTS response_feedback_rating_idx
    ON response_feedback (rating);

-- Support topic-based filtering in analytics.
CREATE INDEX IF NOT EXISTS response_feedback_topic_idx
    ON response_feedback (topic);

-- Support user-level filtering when multi-user auth is enabled.
CREATE INDEX IF NOT EXISTS response_feedback_user_id_idx
    ON response_feedback (user_id);

-- Persist encrypted per-user OpenRouter keys for BYOK settings.
CREATE TABLE IF NOT EXISTS user_openrouter_keys (
    user_id TEXT PRIMARY KEY,
    encrypted_api_key TEXT NOT NULL,
    key_hint TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS user_openrouter_keys_updated_at_idx
    ON user_openrouter_keys (updated_at DESC);
