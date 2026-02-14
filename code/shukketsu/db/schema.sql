-- Shukketsu Database Schema v7
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

INSERT INTO schema_version (version) VALUES (7);

-- ============================================================
-- Ingested Content
-- ============================================================

-- Web pages, guides, forum posts we've ingested
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    source_type TEXT,                              -- "guide", "forum", "wiki", etc.
    trust_score REAL NOT NULL DEFAULT 0.3,         -- 0.0-1.0
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
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,                  -- order within source
    metadata_json TEXT                              -- {"author": "...", "topic": "..."}
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, chunk_index);

-- 768-dim embeddings for semantic search (nomic-embed-text-v2, cosine distance)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

-- Full-text search index (BM25 ranking, content-synced with chunks table)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ============================================================
-- Knowledge Management
-- ============================================================

-- Wiki article metadata (articles themselves are Markdown in knowledge/)
CREATE TABLE IF NOT EXISTS articles (
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

CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_spec ON articles(spec);

-- ============================================================
-- Knowledge Graph (Phase 2)
-- ============================================================

-- Entity type catalog (seeded from code on first use)
CREATE TABLE IF NOT EXISTS entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

-- Named entities extracted from source content
CREATE TABLE IF NOT EXISTS entities (
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

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

-- Typed edges between entities
CREATE TABLE IF NOT EXISTS relationships (
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

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);

-- ============================================================
-- Memory (Phase 3)
-- ============================================================

-- Facts extracted from conversations
CREATE TABLE IF NOT EXISTS session_memories (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    answer_summary TEXT NOT NULL,
    key_facts_json TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    user_feedback TEXT,
    retrieval_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_memories_created ON session_memories(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS session_memories_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS strategy_memories (
    id INTEGER PRIMARY KEY,
    query_pattern TEXT NOT NULL,
    strategy_type TEXT NOT NULL DEFAULT 'routing',
    successful_tools TEXT NOT NULL DEFAULT '[]',
    failed_tools TEXT NOT NULL DEFAULT '[]',
    best_sources TEXT NOT NULL DEFAULT '[]',
    times_reinforced INTEGER NOT NULL DEFAULT 1,
    avg_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_used TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_strategy_memories_pattern ON strategy_memories(query_pattern);
CREATE INDEX IF NOT EXISTS idx_strategy_memories_type ON strategy_memories(strategy_type);

-- ============================================================
-- Trust Events (Phase 3)
-- ============================================================

CREATE TABLE IF NOT EXISTS trust_events (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    delta REAL NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trust_events_source ON trust_events(source_id);

-- ============================================================
-- Warcraft Logs API (Phase 4)
-- ============================================================

-- Tracked characters for personal monitoring
CREATE TABLE IF NOT EXISTS wcl_tracked_characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wcl_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    server_slug TEXT NOT NULL,
    server_region TEXT NOT NULL,
    endpoint TEXT NOT NULL DEFAULT 'fresh',
    class_id INTEGER,
    last_sync_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Encounter rankings (top 100 per encounter)
CREATE TABLE IF NOT EXISTS wcl_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    spec TEXT NOT NULL,
    dps REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    guild_name TEXT,
    server_name TEXT NOT NULL,
    server_region TEXT NOT NULL,
    faction INTEGER,
    raid_size INTEGER,
    bracket_data INTEGER,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(encounter_id, player_name, server_name, report_code)
);

-- Report metadata
CREATE TABLE IF NOT EXISTS wcl_reports (
    code TEXT PRIMARY KEY,
    title TEXT,
    zone_id INTEGER,
    zone_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    is_archived INTEGER NOT NULL DEFAULT 0,
    endpoint TEXT NOT NULL DEFAULT 'fresh',
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Fight metadata within reports
CREATE TABLE IF NOT EXISTS wcl_fights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    kill INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    boss_percentage REAL,
    avg_item_level REAL,
    raid_size INTEGER,
    difficulty INTEGER,
    UNIQUE(report_code, fight_id)
);

-- CombatantInfo per player per fight
CREATE TABLE IF NOT EXISTS wcl_combatants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    player_name TEXT,
    spec_id INTEGER,
    faction INTEGER,
    strength INTEGER, agility INTEGER, stamina INTEGER,
    intellect INTEGER, spirit INTEGER,
    crit_melee INTEGER, crit_ranged INTEGER, crit_spell INTEGER,
    haste_melee INTEGER, haste_ranged INTEGER, haste_spell INTEGER,
    hit_melee INTEGER, hit_ranged INTEGER, hit_spell INTEGER,
    expertise INTEGER, dodge INTEGER, parry INTEGER, block INTEGER, armor INTEGER,
    gear_json TEXT,
    talents_json TEXT,
    auras_json TEXT,
    UNIQUE(report_code, fight_id, source_id)
);

-- Damage breakdown per player per fight
CREATE TABLE IF NOT EXISTS wcl_damage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    player_type TEXT,
    total_damage INTEGER NOT NULL,
    active_time_ms INTEGER,
    abilities_json TEXT,
    targets_json TEXT,
    UNIQUE(report_code, fight_id, player_name)
);

-- Buff uptimes per player per fight
CREATE TABLE IF NOT EXISTS wcl_buffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    buff_name TEXT NOT NULL,
    buff_guid INTEGER NOT NULL,
    total_uptime_ms INTEGER NOT NULL,
    total_uses INTEGER NOT NULL,
    bands_json TEXT,
    UNIQUE(report_code, fight_id, buff_guid)
);

-- Cast counts per player per fight
CREATE TABLE IF NOT EXISTS wcl_casts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    ability_name TEXT NOT NULL,
    cast_count INTEGER NOT NULL,
    UNIQUE(report_code, fight_id, player_name, ability_name)
);

-- Per-fight rankings (percentiles)
CREATE TABLE IF NOT EXISTS wcl_fight_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    server_name TEXT,
    class TEXT NOT NULL,
    spec TEXT,
    dps REAL NOT NULL,
    rank_percent INTEGER,
    total_parses INTEGER,
    UNIQUE(report_code, fight_id, player_name)
);

-- Character sync log
CREATE TABLE IF NOT EXISTS wcl_character_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_wcl_id INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT,
    dps REAL,
    rank_percent INTEGER,
    kill INTEGER,
    timestamp INTEGER,
    UNIQUE(character_wcl_id, report_code, fight_id)
);

CREATE INDEX IF NOT EXISTS idx_wcl_rankings_encounter ON wcl_rankings(encounter_id);
CREATE INDEX IF NOT EXISTS idx_wcl_rankings_player ON wcl_rankings(player_name, server_name);
CREATE INDEX IF NOT EXISTS idx_wcl_combatants_report ON wcl_combatants(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_damage_report ON wcl_damage(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_character_log_char ON wcl_character_log(character_wcl_id);

-- ============================================================
-- Validation (Phase 4)
-- ============================================================

CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT NOT NULL,
    run_type TEXT NOT NULL,
    total_fights INTEGER NOT NULL,
    included_fights INTEGER NOT NULL,
    overall_dps_drift_pct REAL NOT NULL,
    overall_status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
