"""Tests for the curated item database (code.shukketsu.sim.items)."""

import pytest

from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    GearSlot,
    GemSlot,
    ProcTrigger,
    RogueSpec,
    WeaponType,
)


@pytest.fixture
def db() -> ItemDatabase:
    """Provide a fresh ItemDatabase instance."""
    return ItemDatabase()


class TestItemDatabaseInit:
    """Tests for database initialization."""

    def test_initializes_without_error(self, db: ItemDatabase) -> None:
        """ItemDatabase should construct successfully."""
        assert db is not None

    def test_at_least_50_items(self, db: ItemDatabase) -> None:
        """Database must contain at least 50 curated items."""
        assert len(db._items) >= 50

    def test_items_span_phases_1_through_5(self, db: ItemDatabase) -> None:
        """Items should cover all five TBC content phases."""
        phases = {item.phase for item in db._items.values()}
        for p in (1, 2, 3, 5):
            assert p in phases, f"Phase {p} missing from item database"

    def test_all_gear_slots_have_at_least_one_item(self, db: ItemDatabase) -> None:
        """Every GearSlot enum value must have at least one item."""
        for slot in GearSlot:
            items = db.items_for_slot(slot)
            assert len(items) >= 1, f"No items for slot {slot.value}"


class TestGetItem:
    """Tests for get_item lookups."""

    def test_get_dragonspine_trophy(self, db: ItemDatabase) -> None:
        """Dragonspine Trophy (28830) should be retrievable by ID."""
        item = db.get_item(28830)
        assert item is not None
        assert item.name == "Dragonspine Trophy"
        assert item.slot == GearSlot.TRINKET_1
        assert item.phase == 1

    def test_get_item_returns_none_for_unknown_id(self, db: ItemDatabase) -> None:
        """Unknown item IDs should return None."""
        assert db.get_item(999999) is None


class TestItemsForSlot:
    """Tests for items_for_slot filtering."""

    def test_trinket_slot_returns_items(self, db: ItemDatabase) -> None:
        """TRINKET_1 slot should contain multiple items (DST, Tsunami, etc.)."""
        trinkets = db.items_for_slot(GearSlot.TRINKET_1)
        assert len(trinkets) >= 2
        names = {t.name for t in trinkets}
        assert "Dragonspine Trophy" in names

    def test_phase_filter_excludes_later_phases(self, db: ItemDatabase) -> None:
        """Phase 1 filter should exclude items from phases 2+."""
        p1_heads = db.items_for_slot(GearSlot.HEAD, phase=1)
        for item in p1_heads:
            assert item.phase <= 1, f"{item.name} is phase {item.phase}, expected <= 1"

    def test_phase_filter_includes_current_and_earlier(self, db: ItemDatabase) -> None:
        """Phase 3 filter should include phases 1, 2, and 3."""
        p3_mh = db.items_for_slot(GearSlot.MAIN_HAND, phase=3)
        phases = {i.phase for i in p3_mh}
        assert 1 in phases
        assert 3 in phases

    def test_min_ilvl_filter(self, db: ItemDatabase) -> None:
        """min_ilvl filter should exclude items below the threshold."""
        high_ilvl = db.items_for_slot(GearSlot.MAIN_HAND, min_ilvl=140)
        for item in high_ilvl:
            assert item.item_level >= 140, f"{item.name} ilvl {item.item_level} < 140"

    def test_results_sorted_by_ilvl_descending(self, db: ItemDatabase) -> None:
        """items_for_slot should return results sorted by item_level descending."""
        items = db.items_for_slot(GearSlot.HEAD)
        ilvls = [i.item_level for i in items]
        assert ilvls == sorted(ilvls, reverse=True)


class TestSearch:
    """Tests for name-based search."""

    def test_search_case_insensitive(self, db: ItemDatabase) -> None:
        """Search for 'dragon' should find Dragonspine Trophy (case-insensitive)."""
        results = db.search("dragon")
        names = {r.name for r in results}
        assert "Dragonspine Trophy" in names

    def test_search_with_slot_filter(self, db: ItemDatabase) -> None:
        """Search with slot filter should narrow results."""
        all_results = db.search("netherblade")
        slot_results = db.search("netherblade", slot=GearSlot.HEAD)
        assert len(slot_results) <= len(all_results)
        for item in slot_results:
            assert item.slot == GearSlot.HEAD


class TestRogueItemsForSlot:
    """Tests for rogue_items_for_slot."""

    def test_returns_items_for_slot(self, db: ItemDatabase) -> None:
        """rogue_items_for_slot should return items for the given slot."""
        items = db.rogue_items_for_slot(GearSlot.MAIN_HAND, RogueSpec.COMBAT_SWORDS)
        assert len(items) >= 1
        for item in items:
            assert item.slot == GearSlot.MAIN_HAND


class TestSetBonuses:
    """Tests for set bonus data."""

    def test_netherblade_set_bonuses(self, db: ItemDatabase) -> None:
        """Netherblade should have 2pc and 4pc bonuses."""
        bonuses = db.get_set_bonuses("netherblade")
        assert len(bonuses) == 2
        pieces = {b.pieces_required for b in bonuses}
        assert pieces == {2, 4}

    def test_slayer_set_bonuses(self, db: ItemDatabase) -> None:
        """Slayer should have 2pc and 4pc bonuses."""
        bonuses = db.get_set_bonuses("slayer")
        assert len(bonuses) == 2

    def test_unknown_set_returns_empty(self, db: ItemDatabase) -> None:
        """Unknown set names should return an empty list."""
        assert db.get_set_bonuses("nonexistent_set") == []


class TestProcItems:
    """Tests for items with proc effects."""

    def test_dst_proc_data(self, db: ItemDatabase) -> None:
        """Dragonspine Trophy should have PPM 1.0 proc with ICD 20s."""
        dst = db.get_item(28830)
        assert dst is not None
        assert dst.proc is not None
        assert dst.proc.trigger == ProcTrigger.PPM
        assert dst.proc.rate == 1.0
        assert dst.proc.icd == 20.0
        assert dst.proc.duration == 10.0
        assert dst.proc.effect.get("haste_rating") == 325.0


class TestWeaponItems:
    """Tests for weapon stat data."""

    def test_blinkstrike_weapon_stats(self, db: ItemDatabase) -> None:
        """Blinkstrike should have correct weapon stats (2.6 speed sword)."""
        item = db.get_item(28573)
        assert item is not None
        assert item.weapon is not None
        assert item.weapon.speed == 2.6
        assert item.weapon.weapon_type == WeaponType.SWORD
        assert item.weapon.min_damage == 105
        assert item.weapon.max_damage == 196


class TestEnchants:
    """Tests for enchant data."""

    def test_get_mongoose_enchant(self, db: ItemDatabase) -> None:
        """Mongoose enchant (2673) should be retrievable."""
        enchant = db.get_enchant(2673)
        assert enchant is not None
        assert enchant.name == "Mongoose"
        assert enchant.slot == GearSlot.MAIN_HAND

    def test_mongoose_proc_effect(self, db: ItemDatabase) -> None:
        """Mongoose should have a PPM 1.0 proc granting agility."""
        enchant = db.get_enchant(2673)
        assert enchant is not None
        assert enchant.proc is not None
        assert enchant.proc.trigger == ProcTrigger.PPM
        assert enchant.proc.rate == 1.0
        assert enchant.proc.effect.get("agility") == 120.0

    def test_get_enchant_returns_none_for_unknown(self, db: ItemDatabase) -> None:
        """Unknown enchant IDs should return None."""
        assert db.get_enchant(999999) is None


class TestGems:
    """Tests for gem data."""

    def test_get_delicate_living_ruby(self, db: ItemDatabase) -> None:
        """Delicate Living Ruby (24028) should be retrievable with correct stats."""
        gem = db.get_gem(24028)
        assert gem is not None
        assert gem.name == "Delicate Living Ruby"
        assert gem.color == GemSlot.RED
        assert gem.stats.get("agility") == 8

    def test_meta_gem_has_condition(self, db: ItemDatabase) -> None:
        """Relentless Earthstorm Diamond should have a meta condition."""
        gem = db.get_gem(32409)
        assert gem is not None
        assert gem.color == GemSlot.META
        assert gem.meta_condition is not None
        assert "red" in gem.meta_condition.lower()

    def test_get_gem_returns_none_for_unknown(self, db: ItemDatabase) -> None:
        """Unknown gem IDs should return None."""
        assert db.get_gem(999999) is None
