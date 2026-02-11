"""KnowledgeManager for wiki article CRUD.

Handles both filesystem (Markdown + YAML frontmatter) and database
(articles table) operations. Articles are the source of truth on disk;
the DB row is for querying and status tracking.
"""

import logging
import re
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArticleStatus(StrEnum):
    """Lifecycle status of a wiki article."""

    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"


class Spec(StrEnum):
    """Valid spec values for article classification."""

    COMBAT = "combat"
    ASSASSINATION = "assassination"
    SUBTLETY = "subtlety"
    GENERAL = "general"


class SourceRef(BaseModel):
    """A source referenced in an article."""

    url: str
    trust: float = Field(ge=0.0, le=1.0)


class ClaimRef(BaseModel):
    """A factual claim in an article, tracked for verification."""

    text: str
    verified: bool = False
    evidence: list[str] = Field(default_factory=list)


class ArticleSummary(BaseModel):
    """Lightweight article record from DB queries."""

    path: str
    title: str
    spec: str
    category: str
    status: ArticleStatus
    confidence_score: float
    last_updated: str


class ArticleMeta(BaseModel):
    """Pydantic model for article YAML frontmatter."""

    title: str
    spec: Spec
    category: str
    status: ArticleStatus = ArticleStatus.DRAFT
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime
    updated_at: datetime
    sources: list[SourceRef] = Field(default_factory=list)
    claims: list[ClaimRef] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Convert a title to a URL-safe slug.

    Lowercase, strip apostrophes, replace non-alphanumeric runs
    with hyphens, strip leading/trailing hyphens.
    """
    cleaned = title.lower().replace("'", "")
    return _SLUG_RE.sub("-", cleaned).strip("-")


def derive_path(spec: str, category: str, title: str) -> str:
    """Derive the relative article path from spec, category, and title."""
    return f"{spec}/{category}/{slugify(title)}.md"


class KnowledgeManager:
    """Article CRUD: filesystem (Markdown) + database (articles table)."""

    def __init__(self, conn: sqlite3.Connection, knowledge_dir: Path) -> None:
        self._conn = conn
        self._knowledge_dir = knowledge_dir

    def create_draft(self, meta: ArticleMeta, content: str) -> str:
        """Write markdown file + insert DB row. Returns relative path.

        Checks DB uniqueness before writing to prevent orphaned files.
        """
        path = derive_path(meta.spec, meta.category, meta.title)

        # Check DB first to prevent orphaned files
        existing = self._conn.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()
        if existing is not None:
            raise ValueError(f"Article already exists at path: {path}")

        # Write file
        full_path = self._knowledge_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        post = frontmatter.Post(content, **meta.model_dump(mode="json"))
        full_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Insert DB row
        try:
            self._conn.execute(
                """INSERT INTO articles
                (path, title, spec, category, status, confidence_score, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    path,
                    meta.title,
                    meta.spec,
                    meta.category,
                    meta.status,
                    meta.confidence,
                    meta.created_at.isoformat() if isinstance(meta.created_at, datetime) else meta.created_at,
                    meta.updated_at.isoformat() if isinstance(meta.updated_at, datetime) else meta.updated_at,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Race condition: another process inserted between check and insert
            full_path.unlink(missing_ok=True)
            raise ValueError(f"Article already exists at path: {path}")

        logger.info("Created draft article: %s", path)
        return path

    def read_article(self, path: str) -> tuple[ArticleMeta, str]:
        """Read and parse frontmatter + content from a markdown file."""
        full_path = self._knowledge_dir / path
        if not full_path.exists():
            raise FileNotFoundError(f"Article not found: {path}")

        post = frontmatter.loads(full_path.read_text(encoding="utf-8"))
        meta = ArticleMeta(**post.metadata)
        return meta, post.content
