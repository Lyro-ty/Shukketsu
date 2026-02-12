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

    id: int
    path: str
    title: str
    spec: str
    category: str
    status: ArticleStatus
    confidence_score: float
    verified_claims: int = 0
    unverified_claims: int = 0
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

    def update_draft(self, path: str, meta: ArticleMeta, content: str) -> None:
        """Overwrite a draft article's content and metadata.

        Only works on articles with status 'draft'. Raises ValueError
        for review or published articles.
        """
        row = self._conn.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Article not found: {path}")

        status = row["status"]
        if status != ArticleStatus.DRAFT:
            raise ValueError(f"Cannot update article with status '{status}'")

        # Write file
        full_path = self._knowledge_dir / path
        now = datetime.now(tz=meta.updated_at.tzinfo)
        meta_updated = meta.model_copy(update={"updated_at": now})

        post = frontmatter.Post(content, **meta_updated.model_dump(mode="json"))
        full_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Update DB row
        self._conn.execute(
            """UPDATE articles SET title = ?, confidence_score = ?, last_updated = ?
            WHERE path = ?""",
            (meta_updated.title, meta_updated.confidence, now.isoformat(), path),
        )
        self._conn.commit()
        logger.info("Updated draft article: %s", path)

    def list_articles(
        self,
        *,
        status: ArticleStatus | None = None,
        spec: Spec | None = None,
    ) -> list[ArticleSummary]:
        """Query articles with optional filters. Returns typed ArticleSummary objects."""
        query = "SELECT id, path, title, spec, category, status, confidence_score, verified_claims, unverified_claims, last_updated FROM articles"
        conditions: list[str] = []
        params: list[str] = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if spec is not None:
            conditions.append("spec = ?")
            params.append(spec)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY last_updated DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            ArticleSummary(
                id=row["id"],
                path=row["path"],
                title=row["title"],
                spec=row["spec"],
                category=row["category"],
                status=ArticleStatus(row["status"]),
                confidence_score=row["confidence_score"],
                verified_claims=row["verified_claims"],
                unverified_claims=row["unverified_claims"],
                last_updated=row["last_updated"],
            )
            for row in rows
        ]

    def set_status(self, path: str, new_status: ArticleStatus) -> None:
        """Transition article status. Enforces draft->review->published order."""
        row = self._conn.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Article not found: {path}")

        current = ArticleStatus(row["status"])
        valid_transitions = {
            ArticleStatus.DRAFT: ArticleStatus.REVIEW,
            ArticleStatus.REVIEW: ArticleStatus.PUBLISHED,
        }

        if valid_transitions.get(current) != new_status:
            raise ValueError(f"Invalid status transition: {current} -> {new_status}")

        self._conn.execute(
            "UPDATE articles SET status = ?, last_updated = ? WHERE path = ?",
            (new_status, datetime.now().isoformat(), path),
        )
        self._conn.commit()
        logger.info("Article %s status: %s -> %s", path, current, new_status)

    def exists(self, spec: str, category: str, title: str) -> str | None:
        """Check if an article exists at the derived path. Returns path or None."""
        path = derive_path(spec, category, title)
        row = self._conn.execute("SELECT path FROM articles WHERE path = ?", (path,)).fetchone()
        return row["path"] if row else None
