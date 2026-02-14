"""WCL-to-SimConfig bridge.

Reconstructs SimConfig from WCL combatant data using synthetic Item
objects. WCL stats are post-buff totals, so buff contributions are
subtracted to produce unbuffed gear stats.
"""

import json
import logging
import sqlite3

from code.shukketsu.resilience.errors import WCLBridgeError
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    BossConfig,
    GearSlot,
    Item,
    PoisonConfig,
    Race,
    RogueSpec,
    SimConfig,
    WeaponStats,
    WeaponType,
)
from code.shukketsu.sim.spell_map import wcl_buff_name

logger = logging.getLogger(__name__)

# WCL spec_id -> (RogueSpec, canonical talent string)
_SPEC_MAP: dict[int, tuple[RogueSpec, str]] = {
    259: (RogueSpec.ASSASSINATION_MUTILATE, "41/20/0"),
    260: (RogueSpec.COMBAT_SWORDS, "20/41/0"),
    261: (RogueSpec.COMBAT_SWORDS, "20/41/0"),  # Subtlety -> treat as combat for sim
}

# WCL gear slot index -> GearSlot (only weapon slots needed for resolution)
_WCL_WEAPON_SLOTS: dict[int, GearSlot] = {
    15: GearSlot.MAIN_HAND,
    16: GearSlot.OFF_HAND,
}


class WCLBridge:
    """Bridges WCL combatant data into SimConfig via synthetic Items."""

    def __init__(self, conn: sqlite3.Connection, item_db: ItemDatabase | None = None) -> None:
        self._conn = conn
        self._item_db = item_db or ItemDatabase()
        self._next_id = -1000

    def _next_synthetic_id(self) -> int:
        """Return the next synthetic item ID (always negative)."""
        self._next_id -= 1
        return self._next_id

    @property
    def item_db(self) -> ItemDatabase:
        """Return the item database used by this bridge."""
        return self._item_db

    def build_config(
        self,
        report_code: str,
        fight_id: int,
        source_id: int,
        *,
        race: str = "human",
        fight_length: int | None = None,
        iterations: int | None = None,
    ) -> SimConfig:
        """Build a SimConfig from WCL combatant data.

        Args:
            report_code: WCL report code.
            fight_id: Fight ID within the report.
            source_id: Player's source ID in the report.
            race: Player race string (lowercase).
            fight_length: Override fight length (default: use fight duration).
            iterations: Override iteration count.

        Returns:
            SimConfig ready for simulation.

        Raises:
            WCLBridgeError: If combatant data is missing or insufficient.
        """
        row = self._conn.execute(
            """SELECT * FROM wcl_combatants
               WHERE report_code = ? AND fight_id = ? AND source_id = ?""",
            (report_code, fight_id, source_id),
        ).fetchone()

        if row is None:
            raise WCLBridgeError(f"No combatant found for report={report_code} fight={fight_id} source={source_id}")

        # Resolve spec and talents
        spec_id = row["spec_id"] or 260
        spec, talents = _SPEC_MAP.get(spec_id, (RogueSpec.COMBAT_SWORDS, "20/41/0"))

        # Parse gear JSON for weapons
        gear_json: list[dict] = json.loads(row["gear_json"] or "[]")
        gear: dict[GearSlot, int] = {}
        self._resolve_weapons(gear_json, gear)

        # Parse auras JSON for buff detection
        auras_json: list[dict] = json.loads(row["auras_json"] or "[]")
        buffs = self._detect_buffs(auras_json)

        # Build synthetic stat body item (unbuffed gear stats)
        stat_item = self._build_stat_item(row, buffs)
        self._item_db.register_item(stat_item)
        gear[GearSlot.CHEST] = stat_item.id

        # Get fight duration for fight_length
        fight_row = self._conn.execute(
            "SELECT duration_ms FROM wcl_fights WHERE report_code = ? AND fight_id = ?",
            (report_code, fight_id),
        ).fetchone()
        duration_s = (fight_row["duration_ms"] // 1000) if fight_row else 300

        # Build race enum
        race_enum = Race(race) if race in [r.value for r in Race] else Race.HUMAN

        from code.shukketsu import config as cfg

        return SimConfig(
            spec=spec,
            race=race_enum,
            talents=talents,
            gear=gear,
            buffs=buffs,
            consumables=[],
            boss=BossConfig(),
            poisons=PoisonConfig(),
            fight_length=fight_length or duration_s,
            iterations=iterations or cfg.VALIDATION_ITERATIONS,
        )

    def _resolve_weapons(self, gear_json: list[dict], gear: dict[GearSlot, int]) -> None:
        """Resolve MH/OH weapons from gear_json, creating synthetic weapons if needed."""
        for entry in gear_json:
            slot_idx = entry.get("slot", -1)
            if slot_idx not in _WCL_WEAPON_SLOTS:
                continue
            gear_slot = _WCL_WEAPON_SLOTS[slot_idx]
            item_id = entry.get("id", 0)
            existing = self._item_db.get_item(item_id)
            if existing is not None and existing.weapon is not None:
                gear[gear_slot] = item_id
            else:
                synth = Item(
                    id=self._next_synthetic_id(),
                    name=f"WCL Weapon {item_id}",
                    slot=gear_slot,
                    item_level=entry.get("itemLevel", 100),
                    phase=1,
                    weapon=WeaponStats(
                        min_damage=100,
                        max_damage=200,
                        speed=2.6,
                        weapon_type=WeaponType.SWORD,
                        dps=57.7,
                    ),
                )
                self._item_db.register_item(synth)
                gear[gear_slot] = synth.id
                logger.warning("Created synthetic weapon for unknown item ID %d", item_id)

    def _detect_buffs(self, auras_json: list[dict]) -> list[str]:
        """Map WCL auras to sim buff IDs."""
        buffs: list[str] = []
        for aura in auras_json:
            ability_id = aura.get("ability", 0)
            buff = wcl_buff_name(ability_id)
            if buff is not None and buff not in buffs:
                buffs.append(buff)
        return buffs

    def _build_stat_item(self, row: sqlite3.Row, buffs: list[str]) -> Item:
        """Build a synthetic chest item carrying unbuffed gear stats.

        WCL stats are post-buff. We subtract known buff contributions
        to approximate the gear-only stats.

        Args:
            row: sqlite3.Row from wcl_combatants with stat columns.
            buffs: List of detected buff names from aura parsing.

        Returns:
            A synthetic Item placed in the CHEST slot with approximated gear stats.
        """
        agi = float(row["agility"] or 0)
        strength = float(row["strength"] or 0)
        stamina = float(row["stamina"] or 0)
        hit_rating = float(row["hit_melee"] or 0)
        crit_rating = float(row["crit_melee"] or 0)
        haste_rating = float(row["haste_melee"] or 0)
        expertise_rating = float(row["expertise"] or 0)

        # Remove Kings 10% multiplicative buff
        if "kings" in buffs:
            agi /= 1.10
            strength /= 1.10
            stamina /= 1.10

        # Subtract base stats (approximate naked Rogue values)
        agi -= 180.0
        strength -= 110.0
        stamina -= 100.0

        # Clamp to zero
        agi = max(0.0, agi)
        strength = max(0.0, strength)
        stamina = max(0.0, stamina)

        return Item(
            id=self._next_synthetic_id(),
            name="WCL Synthetic Stats",
            slot=GearSlot.CHEST,
            item_level=0,
            phase=1,
            stats={
                "agility": agi,
                "strength": strength,
                "stamina": stamina,
                "hit_rating": hit_rating,
                "crit_rating": crit_rating,
                "haste_rating": haste_rating,
                "expertise_rating": expertise_rating,
            },
        )
