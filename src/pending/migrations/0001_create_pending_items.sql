-- SQLite dialect, STRICT mode (requires SQLite 3.37+)
CREATE TABLE IF NOT EXISTS pending_items (
    -- Identity
    id TEXT PRIMARY KEY
    CHECK (length(id) = 36),
    -- Timing: ISO-8601; sqlx maps to DateTime<Utc> via chrono feature
    created_at TEXT NOT NULL
    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Retry bookkeeping
    retry_count INTEGER NOT NULL DEFAULT 0
    CHECK (retry_count >= 0),
    last_retry_at TEXT,
    -- Full incoming message snapshot (JSON)
    incoming TEXT NOT NULL
    CHECK (json_valid(incoming)),
    -- Enrichment context captured at fallback time
    url_contents TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(url_contents)),
    tool_results TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(tool_results)),
    source_urls TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(source_urls)),
    fallback_title TEXT,
    -- Telegram status message ID for resume notifications
    telegram_status_msg_id INTEGER,
    -- Generated columns extracted from JSON for efficient querying
    -- SourceMetadata is externally-tagged by serde:
    --   {"Telegram":{...}}, {"Http":{...}}, {"Email":{...}}
    source TEXT GENERATED ALWAYS AS (
        CASE
            WHEN json_extract(incoming, '$.metadata.Telegram') IS NOT NULL
                THEN 'telegram'
            WHEN json_extract(incoming, '$.metadata.Email') IS NOT NULL
                THEN 'email'
            ELSE 'http'
        END
    ) STORED,
    received_at TEXT GENERATED ALWAYS AS (
        json_extract(incoming, '$.received_at')
    ) STORED,
    url_count INTEGER GENERATED ALWAYS AS (
        json_array_length(url_contents)
    ) STORED,
    tool_count INTEGER GENERATED ALWAYS AS (
        json_array_length(tool_results)
    ) STORED
) STRICT;

CREATE INDEX IF NOT EXISTS idx_pending_retry
ON pending_items (retry_count, received_at);

CREATE INDEX IF NOT EXISTS idx_pending_source
ON pending_items (source);
