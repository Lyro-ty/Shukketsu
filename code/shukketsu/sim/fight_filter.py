"""WCL fight filtering for sim validation.

Selects fights suitable for DPS calibration by excluding wipes,
deaths, short fights, and movement-heavy encounters.
"""

import logging
import sqlite3
from typing import Final

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Encounters where continuous combat assumption holds (encounter_id -> name)
PATCHWERK_ENCOUNTERS: Final[dict[int, str]] = {
    # Molten Core (Fresh)
    200663: "Lucifron",
    200664: "Magmadar",
    200670: "Golemagg the Incinerator",
    # Karazhan
    652: "Attumen the Huntsman",
    # Gruul's Lair
    649: "Gruul the Dragonkiller",
    # SSC
    623: "Hydross the Unstable",
    624: "The Lurker Below",
    # TK
    730: "Void Reaver",
    # Black Temple
    601: "Supremus",
    602: "Shade of Akama",
    # Sunwell Plateau
    725: "Brutallus",
    726: "Felmyst",
}

# If active_time / fight_duration < this, the player likely died
_DEATH_ACTIVE_RATIO: Final[float] = 0.85


class FilterCriteria(BaseModel):
    """Criteria for selecting valid calibration fights."""

    model_config = ConfigDict(frozen=True)

    kills_only: bool = True
    min_duration_ms: int = 30_000
    patchwerk_only: bool = True
    encounter_ids: set[int] | None = None


class ValidatedFight(BaseModel):
    """A WCL fight annotated with inclusion/exclusion status."""

    report_code: str
    fight_id: int
    encounter_id: int
    encounter_name: str
    duration_ms: int
    source_id: int
    wcl_active_dps: float
    wcl_total_damage: int
    included: bool
    exclusion_reason: str | None = None


class FightFilter:
    """Filters WCL fights for sim validation suitability."""

    def __init__(self, criteria: FilterCriteria | None = None) -> None:
        self._criteria = criteria or FilterCriteria()

    def filter_fights(
        self,
        conn: sqlite3.Connection,
        character_name: str,
    ) -> list[ValidatedFight]:
        """Return all fights for a character, annotated with inclusion/exclusion."""
        rows = conn.execute(
            """SELECT f.report_code, f.fight_id, f.encounter_id, f.encounter_name,
                      f.kill, f.duration_ms,
                      d.total_damage, d.active_time_ms,
                      c.source_id
               FROM wcl_fights f
               JOIN wcl_damage d ON d.report_code = f.report_code AND d.fight_id = f.fight_id
               JOIN wcl_combatants c ON c.report_code = f.report_code AND c.fight_id = f.fight_id
               WHERE d.player_name = ? AND c.player_name = ?""",
            (character_name, character_name),
        ).fetchall()

        results: list[ValidatedFight] = []
        for row in rows:
            exclusion_reason = self._check_exclusion(
                kill=row["kill"],
                duration_ms=row["duration_ms"],
                encounter_id=row["encounter_id"],
                active_time_ms=row["active_time_ms"],
            )

            active_time_ms = row["active_time_ms"] or 0
            active_dps = (
                row["total_damage"] / (active_time_ms / 1000)
                if active_time_ms > 0
                else 0.0
            )

            results.append(
                ValidatedFight(
                    report_code=row["report_code"],
                    fight_id=row["fight_id"],
                    encounter_id=row["encounter_id"],
                    encounter_name=row["encounter_name"],
                    duration_ms=row["duration_ms"],
                    source_id=row["source_id"],
                    wcl_active_dps=active_dps,
                    wcl_total_damage=row["total_damage"],
                    included=exclusion_reason is None,
                    exclusion_reason=exclusion_reason,
                )
            )

        return results

    def _check_exclusion(
        self,
        *,
        kill: int,
        duration_ms: int,
        encounter_id: int,
        active_time_ms: int | None,
    ) -> str | None:
        """Return exclusion reason or None if fight is valid."""
        c = self._criteria

        if c.kills_only and not kill:
            return "wipe"

        if duration_ms < c.min_duration_ms:
            return "short_fight"

        if c.patchwerk_only and c.encounter_ids is None:
            if encounter_id not in PATCHWERK_ENCOUNTERS:
                return "movement_boss"
        elif c.encounter_ids is not None:
            if encounter_id not in c.encounter_ids:
                return "not_in_whitelist"

        # Detect death: active time < 85% of fight duration
        if active_time_ms and duration_ms > 0:
            if active_time_ms / duration_ms < _DEATH_ACTIVE_RATIO:
                return "death"

        return None
