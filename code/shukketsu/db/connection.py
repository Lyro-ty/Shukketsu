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
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    logger.info("Database schema initialized (version 3)")


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
