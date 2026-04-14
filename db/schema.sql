-- schema.sql: Local feedback storage schema for Study Pal.

CREATE TABLE IF NOT EXISTS response_feedback (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
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
