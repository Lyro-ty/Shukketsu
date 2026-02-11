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

INSERT INTO schema_version (version) VALUES (3);

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
    path TEXT UNIQUE NOT NULL,                     -- "combat/gear/trinkets.md"
    title TEXT NOT NULL,
    spec TEXT NOT NULL,                            -- "combat", "assassination", "subtlety", "general"
    category TEXT NOT NULL,                        -- "gear", "rotation", "mechanics", etc.
    status TEXT NOT NULL DEFAULT 'draft',           -- "draft", "review", "published"
    confidence_score REAL NOT NULL DEFAULT 0.0,    -- 0.0-1.0, average of finding confidences
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_articles_status ON articles(status);
CREATE INDEX idx_articles_spec ON articles(spec);

-- ============================================================
-- Knowledge Graph (Phase 2)
-- ============================================================

-- Entity type catalog (seeded from code on first use)
CREATE TABLE entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

-- Named entities extracted from source content
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,                                -- original surface form
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id),
    canonical_name TEXT NOT NULL,                      -- normalized for dedup
    properties_json TEXT,                              -- {"ilvl": 141, ...}
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(canonical_name, entity_type_id)
);

CREATE INDEX idx_entities_type ON entities(entity_type_id);
CREATE INDEX idx_entities_canonical ON entities(canonical_name);

-- Typed edges between entities
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,                       -- from RelationType enum
    properties_json TEXT,                              -- {"phase": 1, ...}
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relation_type);
