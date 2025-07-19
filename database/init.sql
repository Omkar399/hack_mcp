-- Screen Memory Assistant Database Schema
-- Initialize pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable full text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Core screen events table
CREATE TABLE screen_events (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_title TEXT,
    app_name    TEXT,
    full_text   TEXT,
    ocr_conf    SMALLINT CHECK (ocr_conf BETWEEN 0 AND 100),
    clip_vec    VECTOR(512),  -- CLIP ViT-B/32 embeddings
    image_path  TEXT,         -- Path to saved screenshot
    scene_hash  VARCHAR(64),  -- For duplicate detection
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Derived commands table
CREATE TABLE commands (
    id          BIGSERIAL PRIMARY KEY,
    event_id    BIGINT REFERENCES screen_events(id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL,
    cmd         TEXT NOT NULL,
    args        TEXT,
    exit_code   SMALLINT,
    shell       TEXT,
    working_dir TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Calendar entries table
CREATE TABLE calendar_entries (
    id          BIGSERIAL PRIMARY KEY,
    event_id    BIGINT REFERENCES screen_events(id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL,
    title       TEXT NOT NULL,
    event_time  TIMESTAMPTZ,
    end_time    TIMESTAMPTZ,
    source_app  TEXT,
    location    TEXT,
    attendees   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Error events (derived from screen_events)
CREATE TABLE error_events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    BIGINT REFERENCES screen_events(id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL,
    error_type  TEXT,
    error_msg   TEXT,
    app_name    TEXT,
    severity    TEXT CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_screen_events_ts ON screen_events (ts DESC);
CREATE INDEX idx_screen_events_app ON screen_events (app_name);
CREATE INDEX idx_screen_events_conf ON screen_events (ocr_conf) WHERE ocr_conf IS NOT NULL;

-- Full text search index
CREATE INDEX idx_screen_events_text_gin ON screen_events USING gin(to_tsvector('english', full_text));

-- Vector similarity search index (HNSW for fast approximate nearest neighbor)
CREATE INDEX idx_screen_events_clip_vec ON screen_events USING hnsw (clip_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Commands indexes
CREATE INDEX idx_commands_ts ON commands (ts DESC);
CREATE INDEX idx_commands_cmd ON commands USING gin(to_tsvector('english', cmd));

-- Calendar indexes
CREATE INDEX idx_calendar_event_time ON calendar_entries (event_time);
CREATE INDEX idx_calendar_ts ON calendar_entries (ts DESC);

-- Error events indexes
CREATE INDEX idx_error_events_ts ON error_events (ts DESC);
CREATE INDEX idx_error_events_severity ON error_events (severity);

-- Create a view for recent events with all derived data
CREATE VIEW recent_events AS
SELECT 
    se.id,
    se.ts,
    se.window_title,
    se.app_name,
    se.full_text,
    se.ocr_conf,
    se.image_path,
    c.cmd,
    c.exit_code,
    cal.title as calendar_title,
    cal.event_time,
    ee.error_type,
    ee.severity
FROM screen_events se
LEFT JOIN commands c ON se.id = c.event_id
LEFT JOIN calendar_entries cal ON se.id = cal.event_id  
LEFT JOIN error_events ee ON se.id = ee.event_id
ORDER BY se.ts DESC;

-- Function to clean up old data (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_events() RETURNS void AS $$
BEGIN
    DELETE FROM screen_events WHERE ts < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Insert some sample data for testing
INSERT INTO screen_events (ts, window_title, app_name, full_text, ocr_conf) VALUES 
    (NOW() - INTERVAL '1 hour', 'Terminal', 'Terminal', 'docker run -p 5432:5432 postgres:13', 85),
    (NOW() - INTERVAL '30 minutes', 'VS Code', 'Visual Studio Code', 'def capture_screen():\n    return pyautogui.screenshot()', 92),
    (NOW() - INTERVAL '15 minutes', 'Chrome', 'Google Chrome', 'Error: Cannot connect to database', 78);

-- Grant permissions to the app user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hack;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hack; 