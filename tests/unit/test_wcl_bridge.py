"""Tests for WCL-to-SimConfig bridge."""

import json
import sqlite3

import pytest

from code.shukketsu.sim.wcl_bridge import WCLBridge


@pytest.fixture()
def bridge_db(test_db: sqlite3.Connection) -> sqlite3.Connection:
    """Populate DB with a minimal WCL fight for bridge testing."""
    test_db.execute("INSERT OR IGNORE INTO wcl_reports (code, endpoint) VALUES ('RPT1', 'fresh')")
    test_db.execute(
        """INSERT OR REPLACE INTO wcl_fights
           (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
           VALUES ('RPT1', 1, 725, 'Brutallus', 1, 180000)""",
    )
    gear = json.dumps(
        [
            {"id": 28189, "slot": 15, "itemLevel": 115},
            {"id": 28189, "slot": 16, "itemLevel": 115},
        ]
    )
    auras = json.dumps(
        [
            {"source": 0, "ability": 25898, "stacks": 0, "icon": "spell_magic.jpg"},
        ]
    )
    test_db.execute(
        """INSERT OR REPLACE INTO wcl_combatants
           (report_code, fight_id, source_id, player_name, spec_id,
            strength, agility, stamina, hit_melee, crit_melee, expertise, haste_melee,
            gear_json, talents_json, auras_json)
           VALUES ('RPT1', 1, 5, 'Lyroo', 260,
                   200, 800, 300, 142, 300, 60, 120,
                   ?, '[]', ?)""",
        (gear, auras),
    )
    test_db.commit()
    return test_db


class TestWCLBridge:
    def test_build_config_returns_sim_config(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import SimConfig

        assert isinstance(config, SimConfig)

    def test_build_config_has_race(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert config.race.value == "orc"

    def test_build_config_has_weapons(self, bridge_db: sqlite3.Connection) -> None:
        """Config must include MH and OH weapon item IDs."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot

        assert GearSlot.MAIN_HAND in config.gear
        assert GearSlot.OFF_HAND in config.gear

    def test_build_config_has_talents(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert len(config.talents) > 0

    def test_build_config_has_buffs_from_auras(self, bridge_db: sqlite3.Connection) -> None:
        """Kings aura in wcl_combatants should map to 'kings' buff."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert "kings" in config.buffs

    def test_build_config_synthetic_stat_item(self, bridge_db: sqlite3.Connection) -> None:
        """A synthetic item with unbuffed stats should be registered."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot

        assert GearSlot.CHEST in config.gear
        assert config.gear[GearSlot.CHEST] < 0

    def test_build_config_missing_combatant_raises(self, bridge_db: sqlite3.Connection) -> None:
        from code.shukketsu.resilience.errors import WCLBridgeError

        bridge = WCLBridge(bridge_db)
        with pytest.raises(WCLBridgeError, match="No combatant"):
            bridge.build_config("RPT1", 1, 999, race="orc")

    def test_spec_detection_combat(self, bridge_db: sqlite3.Connection) -> None:
        """spec_id 260 (Combat) maps to combat_swords."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert "combat" in config.spec.value

    def test_item_db_has_synthetic_items(self, bridge_db: sqlite3.Connection) -> None:
        """After build_config, bridge's item_db includes synthetic items."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot

        chest_id = config.gear[GearSlot.CHEST]
        item = bridge.item_db.get_item(chest_id)
        assert item is not None
        assert item.stats.get("agility", 0) > 0
