"""Source manifest: Pydantic-validated YAML source list with filtering."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SourceEntry(BaseModel):
    """A single source to ingest."""

    url: str
    title: str
    source_type: str = "guide"
    category: str = "general"
    priority: int = Field(default=2, ge=1, le=3)
    spec: str | None = None
    enabled: bool = True

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        return v


class Manifest(BaseModel):
    """Collection of sources with filtering."""

    sources: list[SourceEntry] = Field(default_factory=list)

    def filter(
        self,
        *,
        max_priority: int | None = None,
        category: str | None = None,
        spec: str | None = None,
        enabled_only: bool = True,
    ) -> list[SourceEntry]:
        """Return sources matching all provided criteria."""
        results = self.sources
        if enabled_only:
            results = [s for s in results if s.enabled]
        if max_priority is not None:
            results = [s for s in results if s.priority <= max_priority]
        if category is not None:
            results = [s for s in results if s.category == category]
        if spec is not None:
            results = [s for s in results if s.spec == spec]
        return results


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a YAML manifest file.

    Supports both list format (plain list of sources) and dict format
    (with a 'sources' key).
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return Manifest()

    if isinstance(data, list):
        return Manifest(sources=data)

    if isinstance(data, dict):
        return Manifest(**data)

    msg = f"Manifest must be a YAML list or dict, got {type(data).__name__}"
    raise ValueError(msg)
