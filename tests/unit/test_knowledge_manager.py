"""Tests for KnowledgeManager article CRUD."""

from code.shukketsu.knowledge.manager import derive_path, slugify


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Phase 1 BiS Gear Guide") == "phase-1-bis-gear-guide"

    def test_special_characters(self) -> None:
        assert slugify("Combat Swords: Rotation Priority") == "combat-swords-rotation-priority"
        assert slugify("What's BiS? (Phase 1)") == "whats-bis-phase-1"

    def test_multiple_hyphens_collapsed(self) -> None:
        assert slugify("foo---bar") == "foo-bar"
        assert slugify("hello   world") == "hello-world"

    def test_leading_trailing_stripped(self) -> None:
        assert slugify("--hello--") == "hello"
        assert slugify("  spaced  ") == "spaced"


class TestDerivePath:
    def test_basic(self) -> None:
        assert derive_path("combat", "gear", "Phase 1 BiS Gear Guide") == "combat/gear/phase-1-bis-gear-guide.md"

    def test_with_special_title(self) -> None:
        assert (
            derive_path("assassination", "rotation", "Mutilate: Priority List")
            == "assassination/rotation/mutilate-priority-list.md"
        )
