"""Tests for simulation engine data models, enums, and error types."""

from enum import StrEnum

from code.shukketsu.resilience.errors import (
    FailureMode,
    InvalidSimConfigError,
    ItemNotFoundError,
    ShukketsuError,
    SimError,
    SimTimeoutError,
)
from code.shukketsu.sim.models import (
    AbilityBreakdown,
    AbilityFlag,
    BossConfig,
    BuffCategory,
    CooldownUsage,
    DpsTimeline,
    Enchant,
    FightType,
    GearSlot,
    Gem,
    GemSlot,
    HitOutcome,
    Item,
    OpenerAbility,
    PoisonConfig,
    PoisonStats,
    PoisonType,
    ProcEffect,
    ProcTrigger,
    ProcUptime,
    Race,
    ResourceStats,
    RogueSpec,
    SetBonus,
    SimConfig,
    SimResult,
    StatWeight,
    WeaponStats,
    WeaponType,
)

# --- Helper: minimal SimConfig for result tests ---


def _minimal_config() -> SimConfig:
    """Return a SimConfig with only required fields for use in result model tests."""
    return SimConfig(
        spec=RogueSpec.COMBAT_SWORDS,
        talents="20/41/0",
        gear={GearSlot.MAIN_HAND: 28438, GearSlot.OFF_HAND: 28439},
    )


def _minimal_resource_stats() -> ResourceStats:
    """Return a ResourceStats with plausible values."""
    return ResourceStats(
        energy_per_second=10.0,
        avg_energy_waste=1.5,
        combo_points_per_second=0.8,
        combo_point_overcap_pct=3.2,
        gcd_utilization_pct=92.0,
        energy_starved_pct=12.0,
    )


# =============================================================================
# Enum tests
# =============================================================================


class TestWeaponType:
    """Tests for WeaponType enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(WeaponType, StrEnum)

    def test_has_all_values(self) -> None:
        expected = {"sword", "dagger", "fist", "mace"}
        assert {v.value for v in WeaponType} == expected

    def test_member_count(self) -> None:
        assert len(WeaponType) == 4


class TestRogueSpec:
    """Tests for RogueSpec enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(RogueSpec, StrEnum)

    def test_has_all_specs(self) -> None:
        expected = {"combat_swords", "combat_fists", "combat_daggers", "assassination_mutilate"}
        assert {v.value for v in RogueSpec} == expected

    def test_member_count(self) -> None:
        assert len(RogueSpec) == 4


class TestGearSlot:
    """Tests for GearSlot enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(GearSlot, StrEnum)

    def test_has_exactly_17_slots(self) -> None:
        assert len(GearSlot) == 17

    def test_has_weapon_slots(self) -> None:
        assert GearSlot.MAIN_HAND == "main_hand"
        assert GearSlot.OFF_HAND == "off_hand"
        assert GearSlot.RANGED == "ranged"

    def test_has_jewelry_slots(self) -> None:
        assert GearSlot.RING_1 == "ring_1"
        assert GearSlot.RING_2 == "ring_2"
        assert GearSlot.TRINKET_1 == "trinket_1"
        assert GearSlot.TRINKET_2 == "trinket_2"


class TestRace:
    """Tests for Race enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(Race, StrEnum)

    def test_has_all_races(self) -> None:
        expected = {"human", "orc", "night_elf", "blood_elf", "undead", "dwarf", "gnome", "troll"}
        assert {v.value for v in Race} == expected

    def test_member_count(self) -> None:
        assert len(Race) == 8


class TestHitOutcome:
    """Tests for HitOutcome enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(HitOutcome, StrEnum)

    def test_has_all_outcomes(self) -> None:
        expected = {"hit", "crit", "miss", "dodge", "glancing", "block"}
        assert {v.value for v in HitOutcome} == expected

    def test_member_count(self) -> None:
        assert len(HitOutcome) == 6


class TestAbilityFlag:
    """Tests for AbilityFlag enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(AbilityFlag, StrEnum)

    def test_has_all_flags(self) -> None:
        expected = {
            "builder",
            "finisher",
            "normalized",
            "cannot_be_dodged",
            "applies_lethality",
            "physical",
            "nature",
            "ignores_armor",
            "main_hand",
            "off_hand",
            "snapshot",
            "off_gcd",
        }
        assert {v.value for v in AbilityFlag} == expected

    def test_member_count(self) -> None:
        assert len(AbilityFlag) == 12


class TestBuffCategory:
    """Tests for BuffCategory enum."""

    def test_is_str_enum(self) -> None:
        assert issubclass(BuffCategory, StrEnum)

    def test_has_all_categories(self) -> None:
        expected = {
            "attack_power",
            "stats_pct",
            "agility_flat",
            "strength_flat",
            "melee_crit",
            "ap_pct",
            "flask",
            "battle_elixir",
            "guardian_elixir",
            "food",
            "uncategorized",
        }
        assert {v.value for v in BuffCategory} == expected

    def test_member_count(self) -> None:
        assert len(BuffCategory) == 11


class TestPoisonType:
    """Tests for PoisonType enum."""

    def test_values(self) -> None:
        assert PoisonType.INSTANT == "instant_poison"
        assert PoisonType.DEADLY == "deadly_poison"
        assert PoisonType.NONE == "none"


class TestOpenerAbility:
    """Tests for OpenerAbility enum."""

    def test_values(self) -> None:
        assert OpenerAbility.GARROTE == "garrote"
        assert OpenerAbility.AMBUSH == "ambush"
        assert OpenerAbility.NONE == "none"


class TestProcTrigger:
    """Tests for ProcTrigger enum."""

    def test_values(self) -> None:
        assert ProcTrigger.ON_HIT == "on_hit"
        assert ProcTrigger.PPM == "ppm"
        assert ProcTrigger.ON_USE == "on_use"


class TestFightType:
    """Tests for FightType enum."""

    def test_values(self) -> None:
        assert FightType.PATCHWERK == "patchwerk"
        assert FightType.CLEAVE == "cleave"
        assert FightType.MOVEMENT == "movement"


class TestGemSlot:
    """Tests for GemSlot enum."""

    def test_values(self) -> None:
        assert GemSlot.RED == "red"
        assert GemSlot.META == "meta"
        assert len(GemSlot) == 4


# =============================================================================
# Item model tests
# =============================================================================


class TestWeaponStats:
    """Tests for WeaponStats model."""

    def test_weapon_stats_with_type(self) -> None:
        ws = WeaponStats(min_damage=80.0, max_damage=150.0, speed=2.7, weapon_type=WeaponType.SWORD, dps=42.6)
        assert ws.weapon_type == WeaponType.SWORD
        assert ws.speed == 2.7


class TestProcEffect:
    """Tests for ProcEffect model."""

    def test_proc_on_hit(self) -> None:
        proc = ProcEffect(trigger=ProcTrigger.ON_HIT, rate=1.0, icd=20.0, duration=10.0, effect={"haste_rating": 325})
        assert proc.trigger == ProcTrigger.ON_HIT
        assert proc.effect["haste_rating"] == 325

    def test_proc_defaults(self) -> None:
        proc = ProcEffect(trigger=ProcTrigger.PPM, rate=1.0)
        assert proc.icd == 0.0
        assert proc.duration == 0.0
        assert proc.effect == {}
        assert proc.stacks == 1


class TestItem:
    """Tests for Item model."""

    def test_minimal_item(self) -> None:
        item = Item(id=28830, name="Dragonspine Trophy", slot=GearSlot.TRINKET_1, item_level=138, phase=1)
        assert item.weapon is None
        assert item.proc is None
        assert item.sockets == []
        assert item.stats == {}
        assert item.set_id is None

    def test_item_with_weapon(self) -> None:
        ws = WeaponStats(min_damage=80.0, max_damage=150.0, speed=2.7, weapon_type=WeaponType.SWORD, dps=42.6)
        item = Item(
            id=28438,
            name="Latro's Shifting Sword",
            slot=GearSlot.MAIN_HAND,
            item_level=115,
            phase=1,
            stats={"agility": 21, "hit_rating": 14},
            weapon=ws,
        )
        assert item.weapon is not None
        assert item.weapon.weapon_type == WeaponType.SWORD

    def test_item_with_proc(self) -> None:
        proc = ProcEffect(trigger=ProcTrigger.ON_HIT, rate=1.0, icd=20.0, duration=10.0, effect={"haste_rating": 325})
        item = Item(id=28830, name="Dragonspine Trophy", slot=GearSlot.TRINKET_1, item_level=138, phase=1, proc=proc)
        assert item.proc is not None
        assert item.proc.effect["haste_rating"] == 325

    def test_item_with_sockets(self) -> None:
        item = Item(
            id=30450,
            name="Wristguards of Determination",
            slot=GearSlot.WRIST,
            item_level=141,
            phase=3,
            stats={"agility": 25, "stamina": 30},
            sockets=[GemSlot.RED, GemSlot.YELLOW],
            socket_bonus={"agility": 3},
        )
        assert len(item.sockets) == 2
        assert item.socket_bonus["agility"] == 3


class TestSetBonus:
    """Tests for SetBonus model."""

    def test_set_bonus(self) -> None:
        bonus = SetBonus(set_name="Slayer's Armor", pieces_required=2, effect={"snd_haste_bonus": 0.05})
        assert bonus.pieces_required == 2
        assert bonus.effect["snd_haste_bonus"] == 0.05


class TestEnchant:
    """Tests for Enchant model."""

    def test_enchant_with_stats(self) -> None:
        ench = Enchant(id=2673, name="Enchant Weapon - Mongoose", slot=GearSlot.MAIN_HAND)
        assert ench.stats == {}
        assert ench.proc is None

    def test_enchant_with_proc(self) -> None:
        proc = ProcEffect(trigger=ProcTrigger.PPM, rate=1.0, duration=15.0, effect={"agility": 120, "haste_pct": 2.0})
        ench = Enchant(id=2673, name="Enchant Weapon - Mongoose", slot=GearSlot.MAIN_HAND, proc=proc)
        assert ench.proc is not None
        assert ench.proc.effect["agility"] == 120


class TestGem:
    """Tests for Gem model."""

    def test_gem_basic(self) -> None:
        gem = Gem(id=24028, name="Delicate Living Ruby", color=GemSlot.RED, stats={"agility": 8})
        assert gem.meta_condition is None
        assert gem.stats["agility"] == 8

    def test_gem_meta_condition(self) -> None:
        gem = Gem(
            id=25899,
            name="Relentless Earthstorm Diamond",
            color=GemSlot.META,
            stats={"agility": 12, "crit_damage_pct": 3},
            meta_condition="more red than blue, more red than yellow",
        )
        assert gem.meta_condition is not None
        assert "red" in gem.meta_condition


# =============================================================================
# Config model tests
# =============================================================================


class TestPoisonConfig:
    """Tests for PoisonConfig defaults."""

    def test_defaults(self) -> None:
        pc = PoisonConfig()
        assert pc.main_hand == PoisonType.INSTANT
        assert pc.off_hand == PoisonType.DEADLY


class TestBossConfig:
    """Tests for BossConfig defaults."""

    def test_defaults(self) -> None:
        bc = BossConfig()
        assert bc.name == "Raid Boss"
        assert bc.level == 73
        assert bc.armor == 7700
        assert bc.debuffs == []

    def test_custom_boss(self) -> None:
        bc = BossConfig(name="Brutallus", armor=6200, debuffs=["sunder_armor", "faerie_fire"])
        assert bc.name == "Brutallus"
        assert len(bc.debuffs) == 2


class TestSimConfig:
    """Tests for SimConfig model."""

    def test_required_fields_only(self) -> None:
        cfg = SimConfig(
            spec=RogueSpec.COMBAT_SWORDS,
            talents="20/41/0",
            gear={GearSlot.MAIN_HAND: 28438, GearSlot.OFF_HAND: 28439},
        )
        assert cfg.spec == RogueSpec.COMBAT_SWORDS
        assert cfg.talents == "20/41/0"
        assert len(cfg.gear) == 2

    def test_defaults_are_correct(self) -> None:
        cfg = _minimal_config()
        assert cfg.race == Race.ORC
        assert cfg.fight_length == 300
        assert cfg.iterations == 10000
        assert cfg.fight_type == FightType.PATCHWERK
        assert cfg.target_count == 1
        assert cfg.raid_preset == "full_25man"
        assert cfg.latency_ms == 0
        assert cfg.opener == OpenerAbility.GARROTE
        assert cfg.expose_armor is False
        assert cfg.use_premeditation is False
        assert cfg.buffs == []
        assert cfg.consumables == []
        assert cfg.enchants == {}
        assert cfg.gems == {}

    def test_poison_config_default(self) -> None:
        cfg = _minimal_config()
        assert cfg.poisons.main_hand == PoisonType.INSTANT
        assert cfg.poisons.off_hand == PoisonType.DEADLY

    def test_boss_config_default(self) -> None:
        cfg = _minimal_config()
        assert cfg.boss.level == 73
        assert cfg.boss.armor == 7700

    def test_all_fields_populated(self) -> None:
        cfg = SimConfig(
            spec=RogueSpec.ASSASSINATION_MUTILATE,
            race=Race.BLOOD_ELF,
            talents="41/20/0",
            gear={GearSlot.MAIN_HAND: 32526, GearSlot.OFF_HAND: 32526},
            enchants={GearSlot.MAIN_HAND: 2673},
            gems={GearSlot.CHEST: [24028, 24028, 24028]},
            poisons=PoisonConfig(main_hand=PoisonType.INSTANT, off_hand=PoisonType.DEADLY),
            opener=OpenerAbility.GARROTE,
            expose_armor=False,
            use_premeditation=True,
            buffs=["blessing_of_kings", "windfury_totem"],
            consumables=["flask_of_relentless_assault"],
            boss=BossConfig(name="Brutallus", armor=6200),
            fight_type=FightType.PATCHWERK,
            target_count=1,
            fight_length=360,
            iterations=25000,
            raid_preset="full_25man",
            latency_ms=50,
        )
        assert cfg.spec == RogueSpec.ASSASSINATION_MUTILATE
        assert cfg.race == Race.BLOOD_ELF
        assert cfg.iterations == 25000
        assert cfg.latency_ms == 50
        assert cfg.use_premeditation is True
        assert len(cfg.gems[GearSlot.CHEST]) == 3


# =============================================================================
# Result model tests
# =============================================================================


class TestAbilityBreakdown:
    """Tests for AbilityBreakdown model."""

    def test_basic(self) -> None:
        ab = AbilityBreakdown(
            name="Sinister Strike",
            damage_total=150000.0,
            damage_pct=35.2,
            casts=210.0,
            hit_pct=72.0,
            crit_pct=20.0,
            miss_pct=5.0,
            dodge_pct=3.0,
        )
        assert ab.name == "Sinister Strike"
        assert ab.glancing_pct == 0.0

    def test_with_glancing(self) -> None:
        ab = AbilityBreakdown(
            name="MH Auto",
            damage_total=100000.0,
            damage_pct=25.0,
            casts=111.0,
            hit_pct=50.0,
            crit_pct=18.0,
            miss_pct=8.0,
            dodge_pct=0.0,
            glancing_pct=24.0,
        )
        assert ab.glancing_pct == 24.0


class TestStatWeight:
    """Tests for StatWeight model."""

    def test_basic(self) -> None:
        sw = StatWeight(stat="agility", ep_value=1.8, dps_per_point=0.45)
        assert sw.is_capped is False

    def test_capped(self) -> None:
        sw = StatWeight(stat="hit_rating", ep_value=2.2, dps_per_point=0.55, is_capped=True)
        assert sw.is_capped is True


class TestResourceStats:
    """Tests for ResourceStats model."""

    def test_all_fields(self) -> None:
        rs = _minimal_resource_stats()
        assert rs.energy_per_second == 10.0
        assert rs.avg_energy_waste == 1.5
        assert rs.combo_points_per_second == 0.8
        assert rs.combo_point_overcap_pct == 3.2
        assert rs.gcd_utilization_pct == 92.0
        assert rs.energy_starved_pct == 12.0


class TestDpsTimeline:
    """Tests for DpsTimeline model."""

    def test_defaults(self) -> None:
        dt = DpsTimeline()
        assert dt.bucket_seconds == 5
        assert dt.buckets == []

    def test_with_buckets(self) -> None:
        dt = DpsTimeline(bucket_seconds=10, buckets=[1200.0, 1350.0, 1400.0, 1380.0])
        assert len(dt.buckets) == 4


class TestProcUptime:
    """Tests for ProcUptime model."""

    def test_basic(self) -> None:
        pu = ProcUptime(name="Dragonspine Trophy", source="trinket_1", uptime_pct=23.5, avg_procs_per_fight=4.2)
        assert pu.avg_stacks == 1.0

    def test_with_stacks(self) -> None:
        pu = ProcUptime(
            name="Deadly Poison", source="off_hand", uptime_pct=95.0, avg_procs_per_fight=50.0, avg_stacks=4.3
        )
        assert pu.avg_stacks == 4.3


class TestPoisonStats:
    """Tests for PoisonStats model."""

    def test_instant_poison(self) -> None:
        ps = PoisonStats(
            poison_type=PoisonType.INSTANT,
            hand="main_hand",
            procs_per_fight=45.0,
            damage_total=25000.0,
            damage_pct=6.0,
        )
        assert ps.avg_deadly_stacks == 0.0

    def test_deadly_poison(self) -> None:
        ps = PoisonStats(
            poison_type=PoisonType.DEADLY,
            hand="off_hand",
            procs_per_fight=30.0,
            damage_total=18000.0,
            damage_pct=4.2,
            avg_deadly_stacks=4.1,
        )
        assert ps.avg_deadly_stacks == 4.1


class TestCooldownUsage:
    """Tests for CooldownUsage model."""

    def test_basic(self) -> None:
        cu = CooldownUsage(name="Blade Flurry", casts_per_fight=2.5, avg_uptime_pct=12.5)
        assert cu.name == "Blade Flurry"


class TestSimResult:
    """Tests for SimResult model."""

    def test_full_result(self) -> None:
        cfg = _minimal_config()
        result = SimResult(
            dps_mean=1850.5,
            dps_std=45.2,
            dps_median=1848.0,
            dps_min=1720.0,
            dps_max=1980.0,
            iterations=10000,
            fight_length=300,
            ability_breakdown=[
                AbilityBreakdown(
                    name="Sinister Strike",
                    damage_total=150000.0,
                    damage_pct=35.0,
                    casts=210.0,
                    hit_pct=72.0,
                    crit_pct=20.0,
                    miss_pct=5.0,
                    dodge_pct=3.0,
                )
            ],
            resource_stats=_minimal_resource_stats(),
            config=cfg,
        )
        assert result.dps_mean == 1850.5
        assert result.iterations == 10000
        assert len(result.ability_breakdown) == 1
        assert result.stat_weights == []
        assert result.proc_uptimes == []
        assert result.poison_stats == []
        assert result.cooldown_usage == []
        assert result.buff_uptimes == {}
        assert result.dps_distribution == []
        assert result.dps_timeline.buckets == []

    def test_empty_stat_weights_default(self) -> None:
        cfg = _minimal_config()
        result = SimResult(
            dps_mean=1000.0,
            dps_std=30.0,
            dps_median=1000.0,
            dps_min=900.0,
            dps_max=1100.0,
            iterations=1000,
            fight_length=300,
            ability_breakdown=[],
            resource_stats=_minimal_resource_stats(),
            config=cfg,
        )
        assert result.stat_weights == []


# =============================================================================
# Error tests
# =============================================================================


class TestSimError:
    """Tests for SimError."""

    def test_has_correct_failure_mode(self) -> None:
        err = SimError("RNG seed collision")
        assert err.failure_mode == FailureMode.SIM_ERROR

    def test_inherits_shukketsu_error(self) -> None:
        err = SimError("event loop crash")
        assert isinstance(err, ShukketsuError)

    def test_message(self) -> None:
        err = SimError("combat state invalid")
        assert "combat state invalid" in str(err)


class TestInvalidSimConfigError:
    """Tests for InvalidSimConfigError."""

    def test_has_correct_failure_mode(self) -> None:
        err = InvalidSimConfigError("missing main hand weapon")
        assert err.failure_mode == FailureMode.SIM_VALIDATION

    def test_inherits_shukketsu_error(self) -> None:
        err = InvalidSimConfigError("bad talent string")
        assert isinstance(err, ShukketsuError)


class TestSimTimeoutError:
    """Tests for SimTimeoutError."""

    def test_has_correct_failure_mode(self) -> None:
        err = SimTimeoutError("exceeded 60s budget")
        assert err.failure_mode == FailureMode.SIM_TIMEOUT

    def test_inherits_shukketsu_error(self) -> None:
        err = SimTimeoutError("timed out")
        assert isinstance(err, ShukketsuError)


class TestItemNotFoundError:
    """Tests for ItemNotFoundError."""

    def test_has_correct_failure_mode(self) -> None:
        err = ItemNotFoundError("item 99999 not in database")
        assert err.failure_mode == FailureMode.TOOL_EXECUTION_ERROR

    def test_inherits_shukketsu_error(self) -> None:
        err = ItemNotFoundError("unknown item")
        assert isinstance(err, ShukketsuError)


# =============================================================================
# Config constants tests
# =============================================================================


class TestSimConfigConstants:
    """Tests for sim-related config values in config.py."""

    def test_sim_defaults_exist(self) -> None:
        from code.shukketsu import config

        assert config.SIM_DEFAULT_ITERATIONS == 10000
        assert config.SIM_DEFAULT_FIGHT_LENGTH == 300
        assert config.SIM_STAT_WEIGHT_DELTA == 80
        assert config.SIM_CACHE_ENABLED is True
        assert config.ANALYST_MAX_ITERATIONS == 5


class TestFailureModeSimValues:
    """Tests for new FailureMode enum values."""

    def test_sim_error_value(self) -> None:
        assert FailureMode.SIM_ERROR == "sim_error"

    def test_sim_validation_value(self) -> None:
        assert FailureMode.SIM_VALIDATION == "sim_validation"

    def test_sim_timeout_value(self) -> None:
        assert FailureMode.SIM_TIMEOUT == "sim_timeout"
