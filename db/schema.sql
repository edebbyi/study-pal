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

CREATE INDEX IF NOT EXISTS response_feedback_created_at_idx
    ON response_feedback (created_at DESC);

CREATE INDEX IF NOT EXISTS response_feedback_rating_idx
    ON response_feedback (rating);

CREATE INDEX IF NOT EXISTS response_feedback_topic_idx
    ON response_feedback (topic);
