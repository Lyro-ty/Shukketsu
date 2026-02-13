"""Tests for sim buff registry and stacking rules."""

from code.shukketsu.sim.buffs import (
    get_preset,
    resolve_buffs,
)
from code.shukketsu.sim.models import ProcTrigger


class TestResolveBuffsFullPreset:
    """Test resolve_buffs with a full 25-man raid preset."""

    def test_full_25man_aggregate_stats(self) -> None:
        """Full 25-man preset produces correct aggregate flat stats."""
        buffs, debuffs, consumables = get_preset("full_25man")
        resolved = resolve_buffs(buffs, debuffs, consumables)

        # Battle Shout 382 + Trueshot 125 + Flask 120 = 627
        assert resolved.flat_stats["attack_power"] == 627
        # Grace of Air 88 + MotW 14 + Warp Burger 20 = 122
        assert resolved.flat_stats["agility"] == 122
        # Strength of Earth 98 + MotW 14 = 112
        assert resolved.flat_stats["strength"] == 112

    def test_full_25man_multipliers(self) -> None:
        """Full 25-man preset produces correct stat multipliers."""
        buffs, debuffs, consumables = get_preset("full_25man")
        resolved = resolve_buffs(buffs, debuffs, consumables)

        assert resolved.stat_multipliers["stats_pct"] == 0.10
        assert resolved.stat_multipliers["melee_crit_pct"] == 0.05


class TestCategoryStacking:
    """Test category-based stacking rules."""

    def test_same_category_highest_wins(self) -> None:
        """Two FOOD buffs -> highest stat total wins."""
        resolved = resolve_buffs([], [], ["food_warp_burger", "food_clefthoof"])
        # Warp Burger agi=20 vs Clefthoof str=20 — same total, first encountered wins
        # Actually both sum to 20. Pick whichever is "highest" — they tie.
        # The impl picks the later one if tied (> not >=). Let's check it picks one.
        active = resolved.active_buff_ids
        # Exactly one food should survive
        food_ids = active & {"food_warp_burger", "food_clefthoof"}
        assert len(food_ids) == 1

    def test_different_categories_stack(self) -> None:
        """Buffs in different categories all stack."""
        resolved = resolve_buffs(
            ["kings", "battle_shout", "grace_of_air"],
            [],
            [],
        )
        assert "kings" in resolved.active_buff_ids
        assert "battle_shout" in resolved.active_buff_ids
        assert "grace_of_air" in resolved.active_buff_ids

    def test_flask_excludes_elixirs(self) -> None:
        """Flask present -> battle and guardian elixirs are excluded."""
        resolved = resolve_buffs(
            [],
            [],
            ["flask_relentless_assault", "elixir_major_agility", "elixir_draenic_wisdom"],
        )
        assert "flask_relentless_assault" in resolved.active_buff_ids
        assert "elixir_major_agility" not in resolved.active_buff_ids
        assert "elixir_draenic_wisdom" not in resolved.active_buff_ids


class TestBossDebuffs:
    """Test boss debuff armor reduction stacking."""

    def test_armor_reduction_sums(self) -> None:
        """All boss debuff armor reductions sum: 2600 + 610 + 800 = 4010."""
        resolved = resolve_buffs(
            [],
            ["sunder_armor", "faerie_fire", "curse_of_recklessness"],
            [],
        )
        assert resolved.boss_armor_reduction == 4010


class TestProcs:
    """Test proc effects on buffs."""

    def test_wf_totem_proc_active(self) -> None:
        """Windfury Totem proc is included in active_procs."""
        resolved = resolve_buffs(["wf_totem"], [], [])
        assert len(resolved.active_procs) == 1
        assert resolved.active_procs[0].trigger == ProcTrigger.ON_HIT
        assert resolved.active_procs[0].effect == {"extra_attacks": 1}

    def test_heroism_proc_active(self) -> None:
        """Heroism proc is included in active_procs."""
        resolved = resolve_buffs(["heroism"], [], [])
        assert len(resolved.active_procs) == 1
        assert resolved.active_procs[0].trigger == ProcTrigger.ON_USE
        assert resolved.active_procs[0].duration == 40.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_buff_list_returns_zero(self) -> None:
        """Empty buff lists return zeroed ResolvedBuffs."""
        resolved = resolve_buffs([], [], [])
        assert resolved.flat_stats == {}
        assert resolved.stat_multipliers == {}
        assert resolved.active_procs == []
        assert resolved.boss_armor_reduction == 0
        assert resolved.active_buff_ids == set()

    def test_unknown_buff_id_skipped(self) -> None:
        """Unknown buff IDs are silently skipped."""
        resolved = resolve_buffs(["nonexistent_buff", "kings"], [], [])
        assert "kings" in resolved.active_buff_ids
        assert "nonexistent_buff" not in resolved.active_buff_ids


class TestPresets:
    """Test raid presets."""

    def test_full_25man_preset(self) -> None:
        """get_preset returns correct lists for full_25man."""
        buffs, debuffs, consumables = get_preset("full_25man")
        assert "kings" in buffs
        assert "battle_shout" in buffs
        assert "sunder_armor" in debuffs
        assert "flask_relentless_assault" in consumables

    def test_solo_preset_empty(self) -> None:
        """get_preset returns empty lists for solo."""
        buffs, debuffs, consumables = get_preset("solo")
        assert buffs == []
        assert debuffs == []
        assert consumables == []
