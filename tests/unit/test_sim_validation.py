"""Tests for simulation validation profiles."""

from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import GearSlot, RogueSpec, SimConfig
from code.shukketsu.sim.validation import (
    VALIDATION_PROFILES,
    ValidationProfile,
    ValidationResult,
    get_validation_profiles,
    run_validation,
)


class TestValidationProfiles:
    """Tests for the canonical validation profile definitions."""

    def test_profiles_not_empty(self) -> None:
        """get_validation_profiles returns at least 5 profiles."""
        profiles = get_validation_profiles()
        assert len(profiles) >= 5

    def test_get_validation_profiles_returns_same_as_constant(self) -> None:
        """get_validation_profiles() returns the module-level constant."""
        assert get_validation_profiles() is VALIDATION_PROFILES

    def test_all_profiles_have_valid_config(self) -> None:
        """Every profile has a valid SimConfig instance."""
        for profile in VALIDATION_PROFILES:
            assert isinstance(profile.config, SimConfig), f"{profile.name}: config is not SimConfig"

    def test_all_profiles_have_reasonable_dps_range(self) -> None:
        """DPS ranges are within sane bounds and ordered correctly."""
        for profile in VALIDATION_PROFILES:
            low, high = profile.expected_dps_range
            assert low > 500, f"{profile.name}: min DPS too low ({low})"
            assert high < 5000, f"{profile.name}: max DPS too high ({high})"
            assert low < high, f"{profile.name}: min >= max ({low} >= {high})"

    def test_combat_profiles_use_2pct_tolerance(self) -> None:
        """All Combat spec profiles default to 2% tolerance."""
        combat_specs = {RogueSpec.COMBAT_SWORDS, RogueSpec.COMBAT_DAGGERS, RogueSpec.COMBAT_FISTS}
        for profile in VALIDATION_PROFILES:
            if profile.spec in combat_specs:
                assert profile.tolerance_pct == 2.0, f"{profile.name}: expected 2.0, got {profile.tolerance_pct}"

    def test_mutilate_profiles_use_4pct_tolerance(self) -> None:
        """Mutilate profiles use a wider 4% tolerance (more variable DPS)."""
        for profile in VALIDATION_PROFILES:
            if profile.spec == RogueSpec.ASSASSINATION_MUTILATE:
                assert profile.tolerance_pct == 4.0, f"{profile.name}: expected 4.0, got {profile.tolerance_pct}"

    def test_profile_names_are_unique(self) -> None:
        """No two profiles share the same name."""
        names = [p.name for p in VALIDATION_PROFILES]
        assert len(names) == len(set(names)), f"Duplicate names found: {names}"

    def test_profiles_cover_multiple_specs(self) -> None:
        """Profiles span at least combat swords and mutilate specs."""
        specs = {p.spec for p in VALIDATION_PROFILES}
        assert RogueSpec.COMBAT_SWORDS in specs
        assert RogueSpec.ASSASSINATION_MUTILATE in specs

    def test_profiles_cover_multiple_phases(self) -> None:
        """Profiles span at least 2 different content phases."""
        phases = {p.phase for p in VALIDATION_PROFILES}
        assert len(phases) >= 2, f"Only {len(phases)} phase(s) covered"

    def test_all_gear_slots_filled(self) -> None:
        """Every profile has all 17 gear slots populated."""
        all_slots = set(GearSlot)
        for profile in VALIDATION_PROFILES:
            filled = set(profile.config.gear.keys())
            missing = all_slots - filled
            assert not missing, f"{profile.name}: missing gear slots {missing}"

    def test_gear_item_ids_are_in_database(self) -> None:
        """Gear item IDs reference items that exist in the ItemDatabase."""
        db = ItemDatabase()
        for profile in VALIDATION_PROFILES:
            for slot, item_id in profile.config.gear.items():
                item = db.get_item(item_id)
                assert item is not None, f"{profile.name}: item {item_id} for {slot} not in ItemDatabase"

    def test_higher_phases_have_higher_dps(self) -> None:
        """P3 combat swords should have higher expected DPS than P1."""
        by_phase: dict[int, ValidationProfile] = {}
        for p in VALIDATION_PROFILES:
            if p.spec == RogueSpec.COMBAT_SWORDS and p.config.raid_preset == "full_25man":
                by_phase[p.phase] = p
        if 1 in by_phase and 3 in by_phase:
            p1_mid = sum(by_phase[1].expected_dps_range) / 2
            p3_mid = sum(by_phase[3].expected_dps_range) / 2
            assert p3_mid > p1_mid, f"P3 midpoint ({p3_mid}) should exceed P1 ({p1_mid})"


class TestValidationResult:
    """Tests for the ValidationResult model."""

    def test_result_model_within_range(self) -> None:
        """A result within range reports within_range=True."""
        result = ValidationResult(
            profile_name="Test",
            our_dps=1500.0,
            expected_range=(1200.0, 1800.0),
            within_range=True,
            drift_pct=0.0,
        )
        assert result.within_range is True

    def test_result_model_outside_range(self) -> None:
        """A result outside range reports within_range=False with positive drift."""
        result = ValidationResult(
            profile_name="Test",
            our_dps=1800.0,
            expected_range=(1200.0, 1600.0),
            within_range=False,
            drift_pct=28.57,
        )
        assert result.within_range is False
        assert result.drift_pct > 0

    def test_result_serialization_roundtrip(self) -> None:
        """ValidationResult survives JSON serialization."""
        result = ValidationResult(
            profile_name="Round Trip",
            our_dps=1234.5,
            expected_range=(1000.0, 1500.0),
            within_range=True,
            drift_pct=-1.24,
        )
        data = result.model_dump()
        restored = ValidationResult(**data)
        assert restored == result


class TestRunValidation:
    """Tests for the run_validation async function."""

    async def test_run_validation_pass(self) -> None:
        """run_validation returns within_range=True when DPS is in range."""

        class FakeRunner:
            async def sim_run(self, config: SimConfig) -> object:
                class FakeResult:
                    dps_mean = 1300.0

                return FakeResult()

        profile = VALIDATION_PROFILES[0]
        result = await run_validation(FakeRunner(), profile)
        assert result.profile_name == profile.name
        assert result.within_range is True

    async def test_run_validation_fail_high(self) -> None:
        """run_validation returns within_range=False when DPS exceeds range."""

        class FakeRunner:
            async def sim_run(self, config: SimConfig) -> object:
                class FakeResult:
                    dps_mean = 99999.0

                return FakeResult()

        profile = VALIDATION_PROFILES[0]
        result = await run_validation(FakeRunner(), profile)
        assert result.within_range is False
        assert result.drift_pct > 0

    async def test_run_validation_drift_sign(self) -> None:
        """Drift is negative when DPS is below midpoint, positive when above."""

        class FakeRunner:
            async def sim_run(self, config: SimConfig) -> object:
                class FakeResult:
                    dps_mean = 900.0  # Below P1 combat swords midpoint (1300)

                return FakeResult()

        profile = VALIDATION_PROFILES[0]
        result = await run_validation(FakeRunner(), profile)
        assert result.drift_pct < 0, "Drift should be negative when below midpoint"
