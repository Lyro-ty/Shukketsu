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
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _configure(conn)
    _load_extensions(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema from schema.sql. Idempotent.

    Checks schema_version table before applying. Safe to call multiple times.
    """
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    logger.info("Database schema initialized (version 1)")


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
