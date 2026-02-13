"""Tests for WCL API Pydantic models.

Uses realistic WoW TBC data: Brutallus encounters, Warglaive of Azzinoth,
Sunwell Plateau, etc.
"""

from code.shukketsu.apis.wcl.models import (
    WCLAbilitySummary,
    WCLActor,
    WCLAuraEntry,
    WCLBuffAura,
    WCLBuffBand,
    WCLCastEntry,
    WCLCharacter,
    WCLCombatantInfo,
    WCLDamageEntry,
    WCLDamageEvent,
    WCLEndpoint,
    WCLFight,
    WCLFightRanking,
    WCLGearItem,
    WCLGem,
    WCLGuildInfo,
    WCLHitType,
    WCLRanking,
    WCLReport,
    WCLReportRef,
    WCLServerInfo,
    WCLTalentEntry,
    WCLTargetSummary,
    WCLZoneInfo,
    WCLZoneRanking,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestWCLEndpoint:
    """Tests for WCLEndpoint enum."""

    def test_classic_value(self) -> None:
        assert WCLEndpoint.CLASSIC == "classic"

    def test_fresh_value(self) -> None:
        assert WCLEndpoint.FRESH == "fresh"

    def test_string_comparison(self) -> None:
        assert WCLEndpoint.CLASSIC == "classic"
        assert WCLEndpoint.FRESH == "fresh"

    def test_all_members(self) -> None:
        members = list(WCLEndpoint)
        assert len(members) == 2


class TestWCLHitType:
    """Tests for WCLHitType enum."""

    def test_normal_hit(self) -> None:
        assert WCLHitType.NORMAL_HIT == 1

    def test_crit(self) -> None:
        assert WCLHitType.CRIT == 2

    def test_absorb(self) -> None:
        assert WCLHitType.ABSORB == 3

    def test_dodge(self) -> None:
        assert WCLHitType.DODGE == 7

    def test_miss(self) -> None:
        assert WCLHitType.MISS == 8

    def test_immune(self) -> None:
        assert WCLHitType.IMMUNE == 10

    def test_partial_resist(self) -> None:
        assert WCLHitType.PARTIAL_RESIST == 16

    def test_all_members(self) -> None:
        members = list(WCLHitType)
        assert len(members) == 7

    def test_int_comparison(self) -> None:
        assert WCLHitType.CRIT == 2
        assert WCLHitType(2) is WCLHitType.CRIT


# ---------------------------------------------------------------------------
# Shared / Reference models
# ---------------------------------------------------------------------------


class TestWCLGuildInfo:
    """Tests for WCLGuildInfo model."""

    def test_required_fields_only(self) -> None:
        guild = WCLGuildInfo(id=12345, name="Nihilum")
        assert guild.id == 12345
        assert guild.name == "Nihilum"
        assert guild.faction is None

    def test_all_fields(self) -> None:
        guild = WCLGuildInfo(id=12345, name="Nihilum", faction=0)
        assert guild.faction == 0


class TestWCLServerInfo:
    """Tests for WCLServerInfo model."""

    def test_required_fields_only(self) -> None:
        server = WCLServerInfo(id=501, name="Whitemane")
        assert server.id == 501
        assert server.name == "Whitemane"
        assert server.region is None

    def test_all_fields(self) -> None:
        server = WCLServerInfo(id=501, name="Whitemane", region="US")
        assert server.region == "US"


class TestWCLReportRef:
    """Tests for WCLReportRef model."""

    def test_from_camel_case(self) -> None:
        ref = WCLReportRef(code="abc123", fightID=5, startTime=1700000000000)
        assert ref.code == "abc123"
        assert ref.fight_id == 5
        assert ref.start_time == 1700000000000

    def test_from_snake_case(self) -> None:
        ref = WCLReportRef(code="xyz789", fight_id=3, start_time=1700000000000)
        assert ref.fight_id == 3
        assert ref.start_time == 1700000000000

    def test_serialization_uses_snake_case(self) -> None:
        ref = WCLReportRef(code="abc123", fightID=5, startTime=1700000000000)
        data = ref.model_dump()
        assert "fight_id" in data
        assert "start_time" in data

    def test_serialization_by_alias(self) -> None:
        ref = WCLReportRef(code="abc123", fightID=5, startTime=1700000000000)
        data = ref.model_dump(by_alias=True)
        assert "fightID" in data
        assert "startTime" in data


# ---------------------------------------------------------------------------
# Rankings & Characters
# ---------------------------------------------------------------------------


class TestWCLRanking:
    """Tests for WCLRanking model."""

    def test_required_fields_only(self) -> None:
        ranking = WCLRanking(
            name="Lyroo",
            **{"class": "Rogue"},
            spec="Combat",
            amount=2847.5,
            duration=180000,
        )
        assert ranking.name == "Lyroo"
        assert ranking.class_name == "Rogue"
        assert ranking.spec == "Combat"
        assert ranking.amount == 2847.5
        assert ranking.duration == 180000

    def test_class_alias(self) -> None:
        """'class' is reserved in Python, so we use class_name with alias."""
        ranking = WCLRanking(name="Lyroo", class_name="Rogue", spec="Combat", amount=2847.5, duration=180000)
        assert ranking.class_name == "Rogue"

    def test_all_fields(self) -> None:
        ranking = WCLRanking(
            name="Lyroo",
            **{"class": "Rogue"},
            spec="Combat",
            amount=2847.5,
            duration=180000,
            hardModeLevel=0,
            report=WCLReportRef(code="rpt1", fightID=2, startTime=1700000000000),
            guild=WCLGuildInfo(id=100, name="Nihilum"),
            server=WCLServerInfo(id=501, name="Whitemane", region="US"),
            faction=0,
            size=25,
            bracketData=0,
        )
        assert ranking.hard_mode_level == 0
        assert ranking.report is not None
        assert ranking.report.code == "rpt1"
        assert ranking.guild is not None
        assert ranking.guild.name == "Nihilum"
        assert ranking.server is not None
        assert ranking.server.region == "US"
        assert ranking.faction == 0
        assert ranking.size == 25
        assert ranking.bracket_data == 0

    def test_nested_report_guild_server(self) -> None:
        """Verify deep nesting with realistic data."""
        data = {
            "name": "Lyroo",
            "class": "Rogue",
            "spec": "Combat",
            "amount": 3150.8,
            "duration": 165000,
            "report": {"code": "SWP2024x", "fightID": 7, "startTime": 1700001234567},
            "guild": {"id": 999, "name": "Death and Taxes", "faction": 0},
            "server": {"id": 42, "name": "Whitemane", "region": "US"},
        }
        ranking = WCLRanking.model_validate(data)
        assert ranking.report is not None
        assert ranking.report.fight_id == 7
        assert ranking.guild is not None
        assert ranking.guild.faction == 0
        assert ranking.server is not None
        assert ranking.server.name == "Whitemane"

    def test_optional_fields_default_none(self) -> None:
        ranking = WCLRanking(name="Lyroo", **{"class": "Rogue"}, spec="Combat", amount=2847.5, duration=180000)
        assert ranking.hard_mode_level is None
        assert ranking.report is None
        assert ranking.guild is None
        assert ranking.server is None
        assert ranking.faction is None
        assert ranking.size is None
        assert ranking.bracket_data is None


class TestWCLZoneRanking:
    """Tests for WCLZoneRanking model."""

    def test_from_camel_case(self) -> None:
        zr = WCLZoneRanking(
            encounterID=725,
            encounterName="Brutallus",
            bestAmount=3200.5,
            totalKills=15,
        )
        assert zr.encounter_id == 725
        assert zr.encounter_name == "Brutallus"
        assert zr.best_amount == 3200.5
        assert zr.total_kills == 15
        assert zr.median_percent is None
        assert zr.rank_percent is None

    def test_all_fields(self) -> None:
        zr = WCLZoneRanking(
            encounterID=725,
            encounterName="Brutallus",
            bestAmount=3200.5,
            medianPercent=95.2,
            totalKills=15,
            rankPercent=98.7,
        )
        assert zr.median_percent == 95.2
        assert zr.rank_percent == 98.7

    def test_from_snake_case(self) -> None:
        zr = WCLZoneRanking(
            encounter_id=725,
            encounter_name="Brutallus",
            best_amount=3200.5,
            total_kills=15,
        )
        assert zr.encounter_id == 725


class TestWCLCharacter:
    """Tests for WCLCharacter model."""

    def test_from_camel_case(self) -> None:
        char = WCLCharacter(id=104956434, name="Lyroo", classID=4)
        assert char.class_id == 4

    def test_from_snake_case(self) -> None:
        char = WCLCharacter(id=104956434, name="Lyroo", class_id=4)
        assert char.class_id == 4

    def test_with_server(self) -> None:
        char = WCLCharacter(
            id=104956434,
            name="Lyroo",
            classID=4,
            server=WCLServerInfo(id=501, name="Whitemane", region="US"),
        )
        assert char.server is not None
        assert char.server.name == "Whitemane"

    def test_server_defaults_none(self) -> None:
        char = WCLCharacter(id=1, name="Test", classID=4)
        assert char.server is None


# ---------------------------------------------------------------------------
# Report-Level
# ---------------------------------------------------------------------------


class TestWCLZoneInfo:
    """Tests for WCLZoneInfo model."""

    def test_construction(self) -> None:
        zone = WCLZoneInfo(id=1012, name="Sunwell Plateau")
        assert zone.id == 1012
        assert zone.name == "Sunwell Plateau"


class TestWCLReport:
    """Tests for WCLReport model."""

    def test_required_fields(self) -> None:
        report = WCLReport(code="SWP2024x", startTime=1700000000000, endTime=1700003600000)
        assert report.code == "SWP2024x"
        assert report.start_time == 1700000000000
        assert report.end_time == 1700003600000
        assert report.title is None
        assert report.zone is None

    def test_all_fields(self) -> None:
        report = WCLReport(
            code="SWP2024x",
            title="Sunwell Plateau - Full Clear",
            startTime=1700000000000,
            endTime=1700003600000,
            zone=WCLZoneInfo(id=1012, name="Sunwell Plateau"),
        )
        assert report.title == "Sunwell Plateau - Full Clear"
        assert report.zone is not None
        assert report.zone.name == "Sunwell Plateau"

    def test_from_snake_case(self) -> None:
        report = WCLReport(code="abc", start_time=100, end_time=200)
        assert report.start_time == 100
        assert report.end_time == 200


class TestWCLFight:
    """Tests for WCLFight model."""

    def test_required_fields(self) -> None:
        fight = WCLFight(id=7, encounterID=725, name="Brutallus")
        assert fight.id == 7
        assert fight.encounter_id == 725
        assert fight.name == "Brutallus"
        assert fight.kill is None
        assert fight.duration is None

    def test_all_fields(self) -> None:
        fight = WCLFight(
            id=7,
            encounterID=725,
            name="Brutallus",
            kill=True,
            duration=180000,
            bossPercentage=0.0,
            averageItemLevel=159.2,
            size=25,
            difficulty=0,
        )
        assert fight.kill is True
        assert fight.duration == 180000
        assert fight.boss_percentage == 0.0
        assert fight.avg_item_level == 159.2
        assert fight.size == 25
        assert fight.difficulty == 0

    def test_from_snake_case(self) -> None:
        fight = WCLFight(id=1, encounter_id=725, name="Brutallus")
        assert fight.encounter_id == 725

    def test_wipe(self) -> None:
        fight = WCLFight(
            id=3,
            encounterID=725,
            name="Brutallus",
            kill=False,
            bossPercentage=15.3,
        )
        assert fight.kill is False
        assert fight.boss_percentage == 15.3


class TestWCLActor:
    """Tests for WCLActor model."""

    def test_required_fields(self) -> None:
        actor = WCLActor(id=1, name="Lyroo", type="Player")
        assert actor.id == 1
        assert actor.name == "Lyroo"
        assert actor.type == "Player"
        assert actor.sub_type is None

    def test_with_sub_type(self) -> None:
        actor = WCLActor(id=1, name="Lyroo", type="Player", subType="Rogue")
        assert actor.sub_type == "Rogue"

    def test_npc_actor(self) -> None:
        actor = WCLActor(id=50, name="Brutallus", type="NPC")
        assert actor.type == "NPC"


# ---------------------------------------------------------------------------
# Per-Player (CombatantInfo)
# ---------------------------------------------------------------------------


class TestWCLGem:
    """Tests for WCLGem model."""

    def test_required_fields(self) -> None:
        gem = WCLGem(id=32409, itemLevel=70)
        assert gem.id == 32409
        assert gem.item_level == 70
        assert gem.icon is None

    def test_all_fields(self) -> None:
        # Relentless Earthstorm Diamond
        gem = WCLGem(id=32409, itemLevel=70, icon="inv_misc_gem_diamond_07")
        assert gem.icon == "inv_misc_gem_diamond_07"

    def test_from_snake_case(self) -> None:
        gem = WCLGem(id=32409, item_level=70)
        assert gem.item_level == 70


class TestWCLGearItem:
    """Tests for WCLGearItem model."""

    def test_required_fields(self) -> None:
        # Warglaive of Azzinoth (MH)
        item = WCLGearItem(id=32837, slot=16, itemLevel=156)
        assert item.id == 32837
        assert item.slot == 16
        assert item.item_level == 156
        assert item.quality is None
        assert item.name is None
        assert item.permanent_enchant is None
        assert item.gems == []
        assert item.set_id is None

    def test_all_fields(self) -> None:
        item = WCLGearItem(
            id=32837,
            slot=16,
            quality=5,
            name="Warglaive of Azzinoth",
            itemLevel=156,
            permanentEnchant=2673,
            permanentEnchantName="Mongoose",
            gems=[
                WCLGem(id=32409, itemLevel=70, icon="inv_misc_gem_diamond_07"),
            ],
            setID=0,
        )
        assert item.quality == 5
        assert item.name == "Warglaive of Azzinoth"
        assert item.permanent_enchant == 2673
        assert item.permanent_enchant_name == "Mongoose"
        assert len(item.gems) == 1
        assert item.gems[0].id == 32409
        assert item.set_id == 0

    def test_multiple_gems(self) -> None:
        # Belt of One-Hundred Deaths with 2 gems
        item = WCLGearItem(
            id=34556,
            slot=6,
            itemLevel=154,
            gems=[
                WCLGem(id=32220, itemLevel=70),  # Glinting Pyrestone
                WCLGem(id=32218, itemLevel=70),  # Shifting Shadowsong Amethyst
            ],
        )
        assert len(item.gems) == 2
        assert item.gems[0].id == 32220
        assert item.gems[1].id == 32218

    def test_from_dict_nested(self) -> None:
        data = {
            "id": 32837,
            "slot": 16,
            "itemLevel": 156,
            "gems": [{"id": 32409, "itemLevel": 70}],
        }
        item = WCLGearItem.model_validate(data)
        assert item.item_level == 156
        assert len(item.gems) == 1
        assert item.gems[0].item_level == 70


class TestWCLTalentEntry:
    """Tests for WCLTalentEntry model."""

    def test_required_fields(self) -> None:
        talent = WCLTalentEntry(guid=14983)
        assert talent.guid == 14983
        assert talent.type is None
        assert talent.name is None
        assert talent.ability_icon is None

    def test_all_fields(self) -> None:
        talent = WCLTalentEntry(guid=14983, type=2, name="Blade Flurry", abilityIcon="ability_warrior_punishingblow")
        assert talent.name == "Blade Flurry"
        assert talent.ability_icon == "ability_warrior_punishingblow"


class TestWCLAuraEntry:
    """Tests for WCLAuraEntry model."""

    def test_all_none(self) -> None:
        aura = WCLAuraEntry()
        assert aura.source is None
        assert aura.ability is None
        assert aura.stacks is None
        assert aura.icon is None
        assert aura.name is None

    def test_all_fields(self) -> None:
        # Slice and Dice
        aura = WCLAuraEntry(source=1, ability=6774, stacks=1, icon="ability_rogue_slicedice", name="Slice and Dice")
        assert aura.ability == 6774
        assert aura.name == "Slice and Dice"


class TestWCLCombatantInfo:
    """Tests for WCLCombatantInfo model."""

    def test_required_fields(self) -> None:
        info = WCLCombatantInfo(sourceID=1)
        assert info.source_id == 1
        assert info.spec_id is None
        assert info.gear == []
        assert info.talents == []
        assert info.auras == []

    def test_full_stat_block(self) -> None:
        """Realistic TBC Rogue stat block."""
        info = WCLCombatantInfo(
            sourceID=1,
            specID=260,  # Combat
            faction=0,
            strength=120,
            agility=780,
            stamina=650,
            intellect=45,
            spirit=55,
            critMelee=2800,
            critRanged=200,
            critSpell=50,
            hasteMelee=215,
            hasteRanged=0,
            hasteSpell=0,
            hitMelee=350,
            hitRanged=100,
            hitSpell=50,
            expertise=26,
            dodge=400,
            parry=0,
            block=0,
            armor=5500,
        )
        assert info.agility == 780
        assert info.crit_melee == 2800
        assert info.haste_melee == 215
        assert info.hit_melee == 350
        assert info.expertise == 26

    def test_with_gear_talents_auras(self) -> None:
        """Full combatant info with nested gear, talents, auras."""
        info = WCLCombatantInfo(
            sourceID=1,
            specID=260,
            gear=[
                WCLGearItem(
                    id=32837,
                    slot=16,
                    itemLevel=156,
                    name="Warglaive of Azzinoth",
                    quality=5,
                    permanentEnchant=2673,
                    gems=[WCLGem(id=32409, itemLevel=70)],
                ),
                WCLGearItem(id=34556, slot=6, itemLevel=154, name="Belt of One-Hundred Deaths"),
            ],
            talents=[
                WCLTalentEntry(guid=14983, name="Blade Flurry"),
                WCLTalentEntry(guid=31124, name="Combat Potency"),
            ],
            auras=[
                WCLAuraEntry(ability=6774, name="Slice and Dice"),
                WCLAuraEntry(ability=35696, name="Drums of Battle"),
            ],
        )
        assert len(info.gear) == 2
        assert info.gear[0].name == "Warglaive of Azzinoth"
        assert len(info.gear[0].gems) == 1
        assert len(info.talents) == 2
        assert info.talents[1].name == "Combat Potency"
        assert len(info.auras) == 2
        assert info.auras[1].name == "Drums of Battle"

    def test_from_dict(self) -> None:
        """Verify model_validate with a realistic API-like dict."""
        data = {
            "sourceID": 3,
            "specID": 259,  # Assassination
            "faction": 0,
            "agility": 720,
            "critMelee": 2600,
            "hitMelee": 300,
            "expertise": 20,
            "armor": 5000,
            "gear": [
                {
                    "id": 34541,
                    "slot": 0,
                    "itemLevel": 154,
                    "name": "Cowl of the Illidari High Lord",
                    "gems": [
                        {"id": 32409, "itemLevel": 70},
                        {"id": 32220, "itemLevel": 70},
                    ],
                },
            ],
            "talents": [{"guid": 14177, "name": "Mutilate"}],
            "auras": [{"ability": 1784, "name": "Stealth"}],
        }
        info = WCLCombatantInfo.model_validate(data)
        assert info.source_id == 3
        assert info.agility == 720
        assert len(info.gear) == 1
        assert len(info.gear[0].gems) == 2
        assert info.talents[0].name == "Mutilate"
        assert info.auras[0].name == "Stealth"

    def test_snake_case_construction(self) -> None:
        info = WCLCombatantInfo(source_id=5, spec_id=260, crit_melee=2800)
        assert info.source_id == 5
        assert info.spec_id == 260
        assert info.crit_melee == 2800


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestWCLAbilitySummary:
    """Tests for WCLAbilitySummary model."""

    def test_required_fields(self) -> None:
        ability = WCLAbilitySummary(name="Sinister Strike", total=450000)
        assert ability.name == "Sinister Strike"
        assert ability.total == 450000
        assert ability.type is None

    def test_with_type(self) -> None:
        ability = WCLAbilitySummary(name="Sinister Strike", total=450000, type=1)
        assert ability.type == 1


class TestWCLTargetSummary:
    """Tests for WCLTargetSummary model."""

    def test_construction(self) -> None:
        target = WCLTargetSummary(name="Brutallus", total=580000)
        assert target.name == "Brutallus"
        assert target.total == 580000


class TestWCLDamageEntry:
    """Tests for WCLDamageEntry model."""

    def test_required_fields(self) -> None:
        entry = WCLDamageEntry(name="Lyroo", id=1, total=580000)
        assert entry.name == "Lyroo"
        assert entry.id == 1
        assert entry.total == 580000
        assert entry.abilities == []
        assert entry.targets == []
        assert entry.gear == []
        assert entry.talents == []

    def test_all_fields(self) -> None:
        entry = WCLDamageEntry(
            name="Lyroo",
            id=1,
            guid=12345,
            type="Player",
            icon="class_rogue",
            itemLevel=159.2,
            total=580000,
            activeTime=175000,
            activeTimeReduced=172000,
        )
        assert entry.guid == 12345
        assert entry.type == "Player"
        assert entry.item_level == 159.2
        assert entry.active_time == 175000
        assert entry.active_time_reduced == 172000

    def test_nested_abilities_targets(self) -> None:
        """Damage entry with nested ability and target breakdowns."""
        entry = WCLDamageEntry(
            name="Lyroo",
            id=1,
            total=580000,
            abilities=[
                WCLAbilitySummary(name="Sinister Strike", total=200000),
                WCLAbilitySummary(name="Blade Flurry", total=150000),
                WCLAbilitySummary(name="Melee", total=130000),
                WCLAbilitySummary(name="Deadly Poison VII", total=60000),
                WCLAbilitySummary(name="Instant Poison VII", total=40000),
            ],
            targets=[
                WCLTargetSummary(name="Brutallus", total=580000),
            ],
        )
        assert len(entry.abilities) == 5
        assert entry.abilities[0].name == "Sinister Strike"
        assert entry.abilities[0].total == 200000
        assert len(entry.targets) == 1
        assert entry.targets[0].name == "Brutallus"

    def test_with_gear_talents(self) -> None:
        """Damage entry includes gear and talents for detailed analysis."""
        entry = WCLDamageEntry(
            name="Lyroo",
            id=1,
            total=580000,
            gear=[
                WCLGearItem(id=32837, slot=16, itemLevel=156),
            ],
            talents=[
                WCLTalentEntry(guid=14983, name="Blade Flurry"),
            ],
        )
        assert len(entry.gear) == 1
        assert len(entry.talents) == 1

    def test_from_dict(self) -> None:
        data = {
            "name": "Lyroo",
            "id": 1,
            "total": 580000,
            "activeTime": 175000,
            "abilities": [
                {"name": "Sinister Strike", "total": 200000},
                {"name": "Melee", "total": 130000},
            ],
            "targets": [{"name": "Brutallus", "total": 580000}],
        }
        entry = WCLDamageEntry.model_validate(data)
        assert entry.active_time == 175000
        assert len(entry.abilities) == 2
        assert len(entry.targets) == 1


class TestWCLBuffBand:
    """Tests for WCLBuffBand model."""

    def test_from_camel_case(self) -> None:
        band = WCLBuffBand(startTime=0, endTime=180000)
        assert band.start_time == 0
        assert band.end_time == 180000

    def test_from_snake_case(self) -> None:
        band = WCLBuffBand(start_time=5000, end_time=35000)
        assert band.start_time == 5000
        assert band.end_time == 35000


class TestWCLBuffAura:
    """Tests for WCLBuffAura model."""

    def test_required_fields(self) -> None:
        aura = WCLBuffAura(
            name="Slice and Dice",
            guid=6774,
            totalUptime=170000,
            totalUses=8,
        )
        assert aura.name == "Slice and Dice"
        assert aura.guid == 6774
        assert aura.total_uptime == 170000
        assert aura.total_uses == 8
        assert aura.bands == []

    def test_with_bands(self) -> None:
        """Buff aura with time bands showing uptime windows."""
        aura = WCLBuffAura(
            name="Slice and Dice",
            guid=6774,
            type=1,
            abilityIcon="ability_rogue_slicedice",
            totalUptime=170000,
            totalUses=8,
            bands=[
                WCLBuffBand(startTime=0, endTime=30000),
                WCLBuffBand(startTime=30500, endTime=60000),
                WCLBuffBand(startTime=60200, endTime=90000),
            ],
        )
        assert aura.ability_icon == "ability_rogue_slicedice"
        assert len(aura.bands) == 3
        assert aura.bands[0].start_time == 0
        assert aura.bands[0].end_time == 30000
        assert aura.bands[2].end_time == 90000

    def test_from_dict(self) -> None:
        data = {
            "name": "Blade Flurry",
            "guid": 13877,
            "totalUptime": 15000,
            "totalUses": 1,
            "bands": [
                {"startTime": 5000, "endTime": 20000},
            ],
        }
        aura = WCLBuffAura.model_validate(data)
        assert aura.total_uptime == 15000
        assert len(aura.bands) == 1
        assert aura.bands[0].start_time == 5000


class TestWCLCastEntry:
    """Tests for WCLCastEntry model."""

    def test_required_fields(self) -> None:
        cast = WCLCastEntry(name="Lyroo", id=1, total=85)
        assert cast.name == "Lyroo"
        assert cast.total == 85
        assert cast.abilities == []

    def test_with_abilities(self) -> None:
        cast = WCLCastEntry(
            name="Lyroo",
            id=1,
            type="Player",
            total=85,
            abilities=[
                WCLAbilitySummary(name="Sinister Strike", total=45),
                WCLAbilitySummary(name="Slice and Dice", total=8),
                WCLAbilitySummary(name="Blade Flurry", total=2),
                WCLAbilitySummary(name="Adrenaline Rush", total=1),
            ],
        )
        assert len(cast.abilities) == 4
        assert cast.abilities[0].total == 45


# ---------------------------------------------------------------------------
# Raw Events
# ---------------------------------------------------------------------------


class TestWCLDamageEvent:
    """Tests for WCLDamageEvent model."""

    def test_required_fields(self) -> None:
        event = WCLDamageEvent(
            timestamp=1700000050000,
            sourceID=1,
            targetID=50,
            abilityGameID=1752,  # Sinister Strike
            hitType=2,  # CRIT
        )
        assert event.timestamp == 1700000050000
        assert event.type == "damage"
        assert event.source_id == 1
        assert event.target_id == 50
        assert event.ability_game_id == 1752
        assert event.hit_type == 2
        assert event.amount == 0  # default

    def test_all_fields(self) -> None:
        event = WCLDamageEvent(
            timestamp=1700000050000,
            type="damage",
            sourceID=1,
            targetID=50,
            abilityGameID=1752,
            fight=7,
            hitType=2,
            amount=4500,
            mitigated=500,
            unmitigatedAmount=5000,
            resisted=0,
            buffs="6774.13877.35696",
            isAoE=False,
        )
        assert event.fight == 7
        assert event.amount == 4500
        assert event.mitigated == 500
        assert event.unmitigated_amount == 5000
        assert event.resisted == 0
        assert event.buffs == "6774.13877.35696"
        assert event.is_aoe is False

    def test_hit_type_enum_compat(self) -> None:
        """Hit type values match WCLHitType enum."""
        event = WCLDamageEvent(timestamp=100, sourceID=1, targetID=2, abilityGameID=1752, hitType=WCLHitType.CRIT)
        assert event.hit_type == WCLHitType.CRIT
        assert event.hit_type == 2

    def test_miss_event(self) -> None:
        event = WCLDamageEvent(
            timestamp=100,
            sourceID=1,
            targetID=50,
            abilityGameID=1752,
            hitType=WCLHitType.MISS,
            amount=0,
        )
        assert event.hit_type == 8
        assert event.amount == 0

    def test_dodge_event(self) -> None:
        event = WCLDamageEvent(
            timestamp=100,
            sourceID=1,
            targetID=50,
            abilityGameID=1752,
            hitType=WCLHitType.DODGE,
            amount=0,
        )
        assert event.hit_type == 7

    def test_partial_resist(self) -> None:
        event = WCLDamageEvent(
            timestamp=100,
            sourceID=1,
            targetID=50,
            abilityGameID=26679,  # Deadly Poison
            hitType=WCLHitType.PARTIAL_RESIST,
            amount=350,
            resisted=150,
            unmitigatedAmount=500,
        )
        assert event.hit_type == 16
        assert event.resisted == 150

    def test_from_dict(self) -> None:
        data = {
            "timestamp": 1700000050000,
            "type": "damage",
            "sourceID": 1,
            "targetID": 50,
            "abilityGameID": 1752,
            "hitType": 2,
            "amount": 4500,
            "unmitigatedAmount": 5000,
            "buffs": "6774.13877",
        }
        event = WCLDamageEvent.model_validate(data)
        assert event.source_id == 1
        assert event.unmitigated_amount == 5000

    def test_optional_defaults(self) -> None:
        event = WCLDamageEvent(timestamp=100, sourceID=1, targetID=2, abilityGameID=1, hitType=1)
        assert event.fight is None
        assert event.mitigated is None
        assert event.unmitigated_amount is None
        assert event.resisted is None
        assert event.buffs is None
        assert event.is_aoe is None


# ---------------------------------------------------------------------------
# Fight Rankings
# ---------------------------------------------------------------------------


class TestWCLFightRanking:
    """Tests for WCLFightRanking model."""

    def test_required_fields(self) -> None:
        ranking = WCLFightRanking(
            name="Lyroo",
            **{"class": "Rogue"},
            amount=3150.8,
        )
        assert ranking.name == "Lyroo"
        assert ranking.class_name == "Rogue"
        assert ranking.amount == 3150.8
        assert ranking.server_name is None
        assert ranking.spec is None

    def test_all_fields(self) -> None:
        ranking = WCLFightRanking(
            name="Lyroo",
            server="Whitemane",
            **{"class": "Rogue"},
            spec="Combat",
            amount=3150.8,
            bracket=0,
            rank=5,
            rankPercent=99.2,
            totalParses=8500,
        )
        assert ranking.server_name == "Whitemane"
        assert ranking.spec == "Combat"
        assert ranking.bracket == 0
        assert ranking.rank == 5
        assert ranking.rank_percent == 99.2
        assert ranking.total_parses == 8500

    def test_class_alias_resolution(self) -> None:
        """Verify 'class' alias resolves to class_name."""
        data = {
            "name": "Lyroo",
            "class": "Rogue",
            "amount": 3150.8,
        }
        ranking = WCLFightRanking.model_validate(data)
        assert ranking.class_name == "Rogue"

    def test_server_alias_resolution(self) -> None:
        """Verify 'server' alias resolves to server_name."""
        data = {
            "name": "Lyroo",
            "class": "Rogue",
            "server": "Whitemane",
            "amount": 3150.8,
        }
        ranking = WCLFightRanking.model_validate(data)
        assert ranking.server_name == "Whitemane"

    def test_from_snake_case(self) -> None:
        ranking = WCLFightRanking(
            name="Lyroo",
            class_name="Rogue",
            amount=3150.8,
            rank_percent=99.2,
            total_parses=8500,
        )
        assert ranking.class_name == "Rogue"
        assert ranking.rank_percent == 99.2
        assert ranking.total_parses == 8500

    def test_serialization_by_alias(self) -> None:
        ranking = WCLFightRanking(name="Lyroo", class_name="Rogue", amount=3150.8, rank_percent=99.2)
        data = ranking.model_dump(by_alias=True)
        assert "class" in data
        assert "rankPercent" in data
        assert data["class"] == "Rogue"


# ---------------------------------------------------------------------------
# Integration / cross-model tests
# ---------------------------------------------------------------------------


class TestCrossModelIntegration:
    """Tests that verify models work together in realistic scenarios."""

    def test_full_ranking_with_nested_objects(self) -> None:
        """Simulate a full ranking response from WCL API."""
        data = {
            "name": "Lyroo",
            "class": "Rogue",
            "spec": "Combat",
            "amount": 3150.8,
            "duration": 165432,
            "hardModeLevel": 0,
            "faction": 0,
            "size": 25,
            "bracketData": 0,
            "report": {
                "code": "SWP2024xABC",
                "fightID": 7,
                "startTime": 1700001234567,
            },
            "guild": {
                "id": 999,
                "name": "Death and Taxes",
                "faction": 0,
            },
            "server": {
                "id": 42,
                "name": "Whitemane",
                "region": "US",
            },
        }
        ranking = WCLRanking.model_validate(data)
        assert ranking.class_name == "Rogue"
        assert ranking.report is not None
        assert ranking.report.code == "SWP2024xABC"
        assert ranking.report.fight_id == 7
        assert ranking.guild is not None
        assert ranking.guild.name == "Death and Taxes"
        assert ranking.server is not None
        assert ranking.server.region == "US"
        assert ranking.size == 25

    def test_full_damage_entry_from_api_dict(self) -> None:
        """Simulate a damage table entry from WCL API."""
        data = {
            "name": "Lyroo",
            "id": 1,
            "guid": 104956434,
            "type": "Player",
            "icon": "class_rogue",
            "itemLevel": 159.2,
            "total": 580000,
            "activeTime": 175000,
            "activeTimeReduced": 172000,
            "abilities": [
                {"name": "Sinister Strike", "total": 200000, "type": 1},
                {"name": "Blade Flurry", "total": 150000, "type": 1},
                {"name": "Melee", "total": 130000, "type": 1},
                {"name": "Deadly Poison VII", "total": 60000, "type": 8},
                {"name": "Instant Poison VII", "total": 40000, "type": 8},
            ],
            "targets": [
                {"name": "Brutallus", "total": 580000},
            ],
            "gear": [
                {
                    "id": 32837,
                    "slot": 16,
                    "quality": 5,
                    "name": "Warglaive of Azzinoth",
                    "itemLevel": 156,
                    "permanentEnchant": 2673,
                    "permanentEnchantName": "Mongoose",
                    "gems": [{"id": 32409, "itemLevel": 70}],
                },
            ],
            "talents": [
                {"guid": 14983, "name": "Blade Flurry", "abilityIcon": "ability_warrior_punishingblow"},
            ],
        }
        entry = WCLDamageEntry.model_validate(data)
        assert entry.item_level == 159.2
        assert entry.active_time == 175000
        assert len(entry.abilities) == 5
        assert entry.abilities[3].name == "Deadly Poison VII"
        assert entry.abilities[3].type == 8
        assert entry.targets[0].name == "Brutallus"
        assert len(entry.gear) == 1
        assert entry.gear[0].name == "Warglaive of Azzinoth"
        assert entry.gear[0].permanent_enchant_name == "Mongoose"
        assert len(entry.gear[0].gems) == 1
        assert len(entry.talents) == 1

    def test_full_combatant_info_from_api_dict(self) -> None:
        """Simulate a complete CombatantInfo response."""
        data = {
            "sourceID": 1,
            "specID": 260,
            "faction": 0,
            "strength": 120,
            "agility": 780,
            "stamina": 650,
            "intellect": 45,
            "spirit": 55,
            "critMelee": 2800,
            "critRanged": 200,
            "critSpell": 50,
            "hasteMelee": 215,
            "hasteRanged": 0,
            "hasteSpell": 0,
            "hitMelee": 350,
            "hitRanged": 100,
            "hitSpell": 50,
            "expertise": 26,
            "dodge": 400,
            "parry": 0,
            "block": 0,
            "armor": 5500,
            "gear": [
                {
                    "id": 32837,
                    "slot": 16,
                    "quality": 5,
                    "name": "Warglaive of Azzinoth",
                    "itemLevel": 156,
                    "permanentEnchant": 2673,
                    "gems": [
                        {"id": 32409, "itemLevel": 70, "icon": "inv_misc_gem_diamond_07"},
                    ],
                },
                {
                    "id": 32838,
                    "slot": 17,
                    "quality": 5,
                    "name": "Warglaive of Azzinoth",
                    "itemLevel": 156,
                    "permanentEnchant": 2673,
                    "gems": [
                        {"id": 32409, "itemLevel": 70},
                    ],
                },
            ],
            "talents": [
                {"guid": 14983, "type": 2, "name": "Blade Flurry"},
                {"guid": 31124, "type": 2, "name": "Combat Potency"},
                {"guid": 35551, "type": 2, "name": "Surprise Attacks"},
            ],
            "auras": [
                {"ability": 6774, "name": "Slice and Dice", "stacks": 1},
                {"ability": 35696, "name": "Drums of Battle"},
                {"ability": 25359, "name": "Grace of Air Totem"},
            ],
        }
        info = WCLCombatantInfo.model_validate(data)
        assert info.source_id == 1
        assert info.spec_id == 260
        assert info.agility == 780
        assert info.crit_melee == 2800
        assert info.expertise == 26
        assert len(info.gear) == 2
        assert info.gear[0].name == "Warglaive of Azzinoth"
        assert info.gear[1].slot == 17
        assert len(info.gear[0].gems) == 1
        assert len(info.talents) == 3
        assert info.talents[2].name == "Surprise Attacks"
        assert len(info.auras) == 3
        assert info.auras[0].stacks == 1

    def test_buff_aura_with_full_bands(self) -> None:
        """Simulate a buff uptime table entry with time bands."""
        data = {
            "name": "Slice and Dice",
            "guid": 6774,
            "type": 1,
            "abilityIcon": "ability_rogue_slicedice",
            "totalUptime": 170000,
            "totalUses": 8,
            "bands": [
                {"startTime": 0, "endTime": 28500},
                {"startTime": 29000, "endTime": 58000},
                {"startTime": 58200, "endTime": 87000},
                {"startTime": 87500, "endTime": 116000},
                {"startTime": 116200, "endTime": 145000},
                {"startTime": 145500, "endTime": 170000},
            ],
        }
        aura = WCLBuffAura.model_validate(data)
        assert aura.total_uptime == 170000
        assert aura.total_uses == 8
        assert len(aura.bands) == 6
        # Verify first and last band
        assert aura.bands[0].start_time == 0
        assert aura.bands[5].end_time == 170000
