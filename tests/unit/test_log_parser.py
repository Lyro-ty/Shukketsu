"""Tests for CLEU combat log parser."""

from code.shukketsu.sim.log_parser import CLEUParser

# Each line is a realistic TBC CLEU event; joined to form the test fixture.
_LOG_LINES = [
    "2/13 20:31:45.000  ENCOUNTER_START,652,Attumen the Huntsman,1,5",
    '2/13 20:31:45.500  SPELL_DAMAGE,0x060000001234,"Lyroo",0x511,0x0,'
    '0xF130000001234,"Boss",0x10A48,0x0,1752,"Sinister Strike",0x1,Boss,0,1234,1,0,0,nil,nil,nil,1',
    '2/13 20:31:46.000  SWING_DAMAGE,0x060000001234,"Lyroo",0x511,0x0,'
    '0xF130000001234,"Boss",0x10A48,0x0,Boss,0,500,1,0,0,nil,nil,nil,0',
    '2/13 20:31:46.500  SPELL_AURA_APPLIED,0x060000001234,"Lyroo",0x511,0x0,'
    '0x060000001234,"Lyroo",0x511,0x0,6774,"Slice and Dice",0x1,BUFF',
    '2/13 20:31:47.000  SPELL_CAST_SUCCESS,0x060000001234,"Lyroo",0x511,0x0,'
    '0xF130000001234,"Boss",0x10A48,0x0,1752,"Sinister Strike",0x1',
    '2/13 20:31:56.500  SPELL_AURA_REMOVED,0x060000001234,"Lyroo",0x511,0x0,'
    '0x060000001234,"Lyroo",0x511,0x0,6774,"Slice and Dice",0x1,BUFF',
    "2/13 20:32:45.000  ENCOUNTER_END,652,Attumen the Huntsman,1,1",
]

SAMPLE_LOG = "\n".join(_LOG_LINES) + "\n"


class TestCLEUParser:
    """Tests for CLEUParser."""

    def test_parse_returns_one_fight(self) -> None:
        """Single encounter pair yields one fight."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert len(fights) == 1

    def test_fight_duration(self) -> None:
        """Duration is ENCOUNTER_END minus ENCOUNTER_START in ms."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert fights[0].fight_duration_ms == 60000  # 20:32:45 - 20:31:45

    def test_damage_aggregation(self) -> None:
        """Total damage sums SPELL_DAMAGE and SWING_DAMAGE."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert fights[0].total_damage == 1734  # 1234 + 500

    def test_ability_breakdown(self) -> None:
        """SPELL_DAMAGE with spell ID 1752 maps to sinister_strike."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert "sinister_strike" in fights[0].ability_breakdown
        assert fights[0].ability_breakdown["sinister_strike"].damage_total == 1234

    def test_melee_tracked(self) -> None:
        """SWING_DAMAGE maps to melee via spell ID 1."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert "melee" in fights[0].ability_breakdown
        assert fights[0].ability_breakdown["melee"].damage_total == 500

    def test_buff_uptime(self) -> None:
        """Buff uptime tracks AURA_APPLIED to AURA_REMOVED duration."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert "slice_and_dice" in fights[0].buff_uptimes
        # Applied at 46.5s, removed at 56.5s = 10s out of 60s fight
        uptime = fights[0].buff_uptimes["slice_and_dice"]
        assert abs(uptime - (10000 / 60000)) < 0.01

    def test_cast_counting(self) -> None:
        """SPELL_CAST_SUCCESS increments cast_count on ability."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "Lyroo")
        assert fights[0].ability_breakdown["sinister_strike"].cast_count == 1

    def test_empty_log(self) -> None:
        """Empty input returns no fights."""
        parser = CLEUParser()
        fights = parser.parse("", "Lyroo")
        assert fights == []

    def test_wrong_character_filtered(self) -> None:
        """Events for other characters are excluded from aggregation."""
        parser = CLEUParser()
        fights = parser.parse(SAMPLE_LOG, "NotLyroo")
        assert len(fights) == 1
        assert fights[0].total_damage == 0

    def test_multiple_encounters(self) -> None:
        """Two encounter pairs yield two fights."""
        second = SAMPLE_LOG.replace("652", "653").replace("Attumen", "Moroes")
        double_log = SAMPLE_LOG + "\n" + second
        parser = CLEUParser()
        fights = parser.parse(double_log, "Lyroo")
        assert len(fights) == 2
