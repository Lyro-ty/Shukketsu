"""SQLite database connection factory with WAL mode and extensions."""

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from code.shukketsu import config

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).parent


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Create a configured SQLite connection with extensions loaded.

    Args:
        db_path: Path to the database file. Defaults to config.DB_PATH.

    Returns:
        A connection with WAL mode, foreign keys, and sqlite-vec enabled.
    """
    path = str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _configure(conn)
    _load_extensions(conn)
    return conn


_GRAPH_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id),
    canonical_name TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(canonical_name, entity_type_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema from schema.sql. Idempotent.

    Checks schema_version table before applying. Safe to call multiple times.
    """
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            if version < 2:
                _migrate_v1_to_v2(conn)
            if version < 3:
                _migrate_v2_to_v3(conn)
            if version < 4:
                _migrate_v3_to_v4(conn)
            if version < 5:
                _migrate_v4_to_v5(conn)
            if version < 6:
                _migrate_v5_to_v6(conn)
            if version < 7:
                _migrate_v6_to_v7(conn)
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    # Apply any migrations beyond base schema version
    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    if version < 5:
        _migrate_v4_to_v5(conn)
    if version < 6:
        _migrate_v5_to_v6(conn)
    if version < 7:
        _migrate_v6_to_v7(conn)
    logger.info("Database schema initialized (version 7)")


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate v1 schema to v2: add knowledge graph tables."""
    conn.executescript(_GRAPH_TABLES_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.commit()
    logger.info("Database migrated from v1 to v2 (knowledge graph tables)")


_ARTICLES_V3_SQL = """
DROP TABLE IF EXISTS articles;

CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    spec TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    confidence_score REAL NOT NULL DEFAULT 0.0,
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_spec ON articles(spec);
"""


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate v2 schema to v3: replace articles table with richer columns."""
    conn.executescript(_ARTICLES_V3_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    conn.commit()
    logger.info("Database migrated from v2 to v3 (articles table v3)")


_MEMORY_V4_SQL = """
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

CREATE TABLE IF NOT EXISTS trust_events (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    delta REAL NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trust_events_source ON trust_events(source_id);
"""


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate v3 schema to v4: add memory tables and trust events."""
    conn.executescript(_MEMORY_V4_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (4)")
    conn.commit()
    logger.info("Database migrated from v3 to v4 (memory + trust_events tables)")


_WCL_SCHEMA = Path(__file__).resolve().parent.parent / "apis" / "wcl" / "schema.sql"


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Migrate v4 schema to v5: add WCL API data tables."""
    wcl_sql = _WCL_SCHEMA.read_text()
    conn.executescript(wcl_sql)
    conn.execute("INSERT INTO schema_version (version) VALUES (5)")
    conn.commit()
    logger.info("Database migrated from v4 to v5 (WCL API tables)")


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Migrate v5 to v6: add player_name to wcl_combatants."""
    try:
        conn.execute("ALTER TABLE wcl_combatants ADD COLUMN player_name TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists (idempotent)
    conn.execute("INSERT INTO schema_version (version) VALUES (6)")
    conn.commit()
    logger.info("Database migrated from v5 to v6 (wcl_combatants.player_name)")


_VALIDATION_V7_SQL = """
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
"""


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Migrate v6 to v7: add validation_runs table."""
    conn.executescript(_VALIDATION_V7_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (7)")
    conn.commit()
    logger.info("Database migrated from v6 to v7 (validation_runs table)")


def _configure(conn: sqlite3.Connection) -> None:
    """Apply SQLite pragmas for WAL mode, safety, and performance."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")


def _load_extensions(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec extension for vector similarity search."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
