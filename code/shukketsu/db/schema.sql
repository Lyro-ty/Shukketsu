-- Shukketsu Database Schema (Phase 1)
--
-- Prerequisites:
--   - sqlite-vec extension loaded (for vec0 virtual table)
--   - FTS5 available (built into SQLite 3.9+)
--
-- Apply via: code.shukketsu.db.connection.init_db()

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version) VALUES (1);

-- ============================================================
-- Ingested Content
-- ============================================================

-- Web pages, guides, forum posts we've ingested
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    source_type TEXT,                              -- "guide", "forum", "wiki", etc.
    trust_score REAL NOT NULL DEFAULT 0.3,         -- 0.0–1.0
    fetched_at TEXT,                                -- ISO 8601
    content_hash TEXT,                              -- SHA-256 for dedup
    chunk_count INTEGER NOT NULL DEFAULT 0,
    -- Freshness tracking
    last_checked TEXT,
    check_interval_hours INTEGER NOT NULL DEFAULT 168,  -- 1 week
    content_hash_previous TEXT,
    change_count INTEGER NOT NULL DEFAULT 0,
    is_stale INTEGER NOT NULL DEFAULT 0            -- SQLite has no BOOLEAN
);

-- Text chunks extracted from sources (semantic boundaries, ~400 tokens)
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,                  -- order within source
    metadata_json TEXT                              -- {"author": "...", "topic": "..."}
);

CREATE INDEX idx_chunks_source ON chunks(source_id, chunk_index);

-- 768-dim embeddings for semantic search (nomic-embed-text-v2, cosine distance)
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

-- Full-text search index (BM25 ranking, content-synced with chunks table)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with chunks table
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ============================================================
-- Knowledge Management
-- ============================================================

-- Wiki article metadata (articles themselves are Markdown in knowledge/)
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,                     -- "specs/combat/overview.md"
    title TEXT NOT NULL,
    last_updated TEXT,                              -- ISO 8601
    confidence_score REAL,                         -- 0.0–1.0
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0        -- 0 or 1
);
