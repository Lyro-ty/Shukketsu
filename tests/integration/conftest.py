"""Integration test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def seed_content_dir() -> Path:
    """Path to the eval seed content directory."""
    return Path(__file__).parent.parent.parent / "code" / "shukketsu" / "evals" / "datasets" / "seed_content"
