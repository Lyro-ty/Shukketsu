"""Tests for ingest manifest models and YAML loading."""

from pathlib import Path

import pytest
import yaml

from code.shukketsu.ingest.manifest import Manifest, SourceEntry, load_manifest


class TestSourceEntry:
    """Tests for SourceEntry model."""

    def test_defaults(self) -> None:
        entry = SourceEntry(url="https://example.com", title="Test")
        assert entry.source_type == "guide"
        assert entry.category == "general"
        assert entry.priority == 2
        assert entry.spec is None
        assert entry.enabled is True

    def test_all_fields(self) -> None:
        entry = SourceEntry(
            url="https://wowhead.com/guide",
            title="Combat Rogue Guide",
            source_type="guide",
            category="class_guide",
            priority=1,
            spec="combat",
            enabled=False,
        )
        assert entry.url == "https://wowhead.com/guide"
        assert entry.spec == "combat"
        assert entry.enabled is False

    def test_url_validation_rejects_bad_scheme(self) -> None:
        with pytest.raises(ValueError, match="http"):
            SourceEntry(url="ftp://example.com", title="Bad")

    def test_priority_bounds(self) -> None:
        with pytest.raises(ValueError):
            SourceEntry(url="https://example.com", title="Bad", priority=0)
        with pytest.raises(ValueError):
            SourceEntry(url="https://example.com", title="Bad", priority=4)


class TestManifest:
    """Tests for Manifest filtering."""

    @pytest.fixture
    def manifest(self) -> Manifest:
        return Manifest(
            sources=[
                SourceEntry(url="https://a.com", title="A", priority=1, category="guide", spec="combat"),
                SourceEntry(url="https://b.com", title="B", priority=2, category="gear", spec="combat"),
                SourceEntry(url="https://c.com", title="C", priority=3, category="guide", spec="assassination"),
                SourceEntry(url="https://d.com", title="D", priority=1, category="gear", enabled=False),
            ]
        )

    def test_filter_enabled_only(self, manifest: Manifest) -> None:
        result = manifest.filter()
        assert len(result) == 3

    def test_filter_by_priority(self, manifest: Manifest) -> None:
        result = manifest.filter(max_priority=1)
        assert len(result) == 1
        assert result[0].title == "A"

    def test_filter_by_category(self, manifest: Manifest) -> None:
        result = manifest.filter(category="gear")
        assert len(result) == 1
        assert result[0].title == "B"

    def test_filter_by_spec(self, manifest: Manifest) -> None:
        result = manifest.filter(spec="combat")
        assert len(result) == 2

    def test_filter_combined(self, manifest: Manifest) -> None:
        result = manifest.filter(max_priority=2, category="guide")
        assert len(result) == 1
        assert result[0].title == "A"

    def test_filter_includes_disabled(self, manifest: Manifest) -> None:
        result = manifest.filter(enabled_only=False)
        assert len(result) == 4


class TestLoadManifest:
    """Tests for YAML loading."""

    def test_load_list_format(self, tmp_path: Path) -> None:
        data = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B", "priority": 1},
        ]
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.dump(data))
        m = load_manifest(path)
        assert len(m.sources) == 2
        assert m.sources[1].priority == 1

    def test_load_dict_format(self, tmp_path: Path) -> None:
        data = {"sources": [{"url": "https://a.com", "title": "A"}]}
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.dump(data))
        m = load_manifest(path)
        assert len(m.sources) == 1

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text("")
        m = load_manifest(path)
        assert len(m.sources) == 0

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text("just a string")
        with pytest.raises(ValueError, match="list or dict"):
            load_manifest(path)
