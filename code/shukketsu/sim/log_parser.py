"""TBC CLEU combat log parser.

Streaming parser that segments fights by ENCOUNTER_START/END,
aggregates damage, buff uptimes, and cast counts per character.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from code.shukketsu.sim.comparator import AbilityMetrics, WCLFightMetrics
from code.shukketsu.sim.spell_map import SPELL_ID_TO_ABILITY

logger = logging.getLogger(__name__)

# Timestamp regex: month/day HH:MM:SS.mmm followed by two-space delimiter
_TIMESTAMP_RE = re.compile(r"(\d+/\d+\s+\d+:\d+:\d+\.\d+)\s{2}(.+)")


@dataclass
class _BuffTracker:
    """Tracks aura application/removal for uptime calculation."""

    applied_at: float = 0.0
    total_uptime_ms: float = 0.0
    is_active: bool = False


@dataclass
class _FightAccumulator:
    """Accumulates data for a single encounter."""

    encounter_id: int = 0
    encounter_name: str = ""
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    damage_by_ability: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    casts_by_ability: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    buff_trackers: dict[str, _BuffTracker] = field(default_factory=dict)
    total_damage: int = 0


def _parse_timestamp_ms(ts_str: str) -> float:
    """Parse CLEU timestamp to milliseconds since midnight.

    Args:
        ts_str: Timestamp string in "M/D HH:MM:SS.mmm" format.

    Returns:
        Milliseconds since midnight (date portion ignored).
    """
    parts = ts_str.strip().split()
    if len(parts) < 2:
        return 0.0
    time_part = parts[1]
    h, m, rest = time_part.split(":")
    s, ms = rest.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


class CLEUParser:
    """Parses TBC combat log text into fight metrics.

    Segments log lines by ENCOUNTER_START/END boundaries and aggregates
    damage, buff uptimes, and cast counts for a specific character.
    Results use the same WCLFightMetrics type as the WCL API path so
    both sources flow through the same comparison pipeline.
    """

    def parse(self, log_text: str, character_name: str) -> list[WCLFightMetrics]:
        """Parse combat log and return metrics per encounter.

        Args:
            log_text: Full combat log text content.
            character_name: Character name to filter events for.

        Returns:
            List of WCLFightMetrics, one per completed encounter.
        """
        if not log_text.strip():
            return []

        fights: list[WCLFightMetrics] = []
        current: _FightAccumulator | None = None

        for line_num, line in enumerate(log_text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue

            match = _TIMESTAMP_RE.match(line)
            if not match:
                continue

            timestamp_str = match.group(1)
            payload = match.group(2)
            timestamp_ms = _parse_timestamp_ms(timestamp_str)

            fields = _split_fields(payload)
            if not fields:
                continue

            event_type = fields[0]

            if event_type == "ENCOUNTER_START":
                encounter_id = int(fields[1]) if len(fields) > 1 else 0
                encounter_name = fields[2] if len(fields) > 2 else "Unknown"
                current = _FightAccumulator(
                    encounter_id=encounter_id,
                    encounter_name=encounter_name,
                    start_time_ms=timestamp_ms,
                )
            elif event_type == "ENCOUNTER_END" and current is not None:
                current.end_time_ms = timestamp_ms
                # Finalize buff uptimes for any buffs still active at fight end
                for bt in current.buff_trackers.values():
                    if bt.is_active:
                        bt.total_uptime_ms += timestamp_ms - bt.applied_at
                        bt.is_active = False

                metrics = _build_metrics(current)
                fights.append(metrics)
                current = None
            elif current is not None:
                _process_event(event_type, fields, timestamp_ms, character_name, current)

        return fights


def _split_fields(payload: str) -> list[str]:
    """Split comma-separated CLEU fields, stripping surrounding quotes.

    Args:
        payload: Raw CLEU payload after the timestamp delimiter.

    Returns:
        List of field strings with quotes removed.
    """
    result: list[str] = []
    current = ""
    in_quotes = False
    for ch in payload:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            result.append(current.strip())
            current = ""
        else:
            current += ch
    if current:
        result.append(current.strip())
    return result


def _process_event(
    event_type: str,
    fields: list[str],
    timestamp_ms: float,
    character_name: str,
    fight: _FightAccumulator,
) -> None:
    """Process a single combat log event into the fight accumulator.

    Args:
        event_type: CLEU event type string.
        fields: Parsed comma-separated fields (index 0 is event_type).
        timestamp_ms: Event timestamp in milliseconds.
        character_name: Character to filter events for.
        fight: Accumulator for the current encounter.
    """
    # Common prefix (indices 0-8):
    #   event_type, sourceGUID, sourceName, sourceFlags, sourceRaidFlags,
    #   destGUID, destName, destFlags, destRaidFlags
    if len(fields) < 4:
        return

    source_name = fields[2].strip('"')

    if event_type == "SWING_DAMAGE":
        if source_name != character_name:
            return
        # SWING_DAMAGE layout (TBC advanced combat log):
        #   [0-8] common prefix
        #   [9] advanced unitName, [10] advanced unitFlags
        #   [11] amount, [12] overkill, [13] school, ...
        amount = _extract_damage_amount(fields, 11)
        ability_name = SPELL_ID_TO_ABILITY.get(1, "melee")
        fight.damage_by_ability[ability_name] += amount
        fight.total_damage += amount

    elif event_type in ("SPELL_DAMAGE", "SPELL_PERIODIC_DAMAGE"):
        if source_name != character_name:
            return
        # SPELL_DAMAGE layout (TBC advanced combat log):
        #   [0-8] common prefix
        #   [9] spellId, [10] spellName, [11] spellSchool
        #   [12] advanced unitName, [13] advanced unitFlags
        #   [14] amount, [15] overkill, [16] school, ...
        spell_id = int(fields[9]) if len(fields) > 9 else 0
        ability_name = SPELL_ID_TO_ABILITY.get(spell_id, f"unknown_{spell_id}")
        amount = _extract_damage_amount(fields, 14)
        fight.damage_by_ability[ability_name] += amount
        fight.total_damage += amount

    elif event_type == "SPELL_CAST_SUCCESS":
        if source_name != character_name:
            return
        spell_id = int(fields[9]) if len(fields) > 9 else 0
        ability_name = SPELL_ID_TO_ABILITY.get(spell_id, f"unknown_{spell_id}")
        fight.casts_by_ability[ability_name] += 1

    elif event_type == "SPELL_AURA_APPLIED":
        dest_name = fields[6].strip('"') if len(fields) > 6 else ""
        if dest_name != character_name:
            return
        spell_id = int(fields[9]) if len(fields) > 9 else 0
        ability_name = SPELL_ID_TO_ABILITY.get(spell_id, f"buff_{spell_id}")
        tracker = fight.buff_trackers.setdefault(ability_name, _BuffTracker())
        tracker.applied_at = timestamp_ms
        tracker.is_active = True

    elif event_type == "SPELL_AURA_REMOVED":
        dest_name = fields[6].strip('"') if len(fields) > 6 else ""
        if dest_name != character_name:
            return
        spell_id = int(fields[9]) if len(fields) > 9 else 0
        ability_name = SPELL_ID_TO_ABILITY.get(spell_id, f"buff_{spell_id}")
        existing = fight.buff_trackers.get(ability_name)
        if existing is not None and existing.is_active:
            existing.total_uptime_ms += timestamp_ms - existing.applied_at
            existing.is_active = False


def _extract_damage_amount(fields: list[str], index: int) -> int:
    """Safely extract an integer damage amount from a field index.

    Args:
        fields: Parsed CLEU fields.
        index: Index of the damage amount field.

    Returns:
        Parsed integer amount, or 0 if the field is missing or non-numeric.
    """
    if len(fields) <= index:
        return 0
    try:
        return int(fields[index])
    except (ValueError, TypeError):
        return 0


def _build_metrics(fight: _FightAccumulator) -> WCLFightMetrics:
    """Convert accumulated fight data to WCLFightMetrics.

    Args:
        fight: Completed fight accumulator.

    Returns:
        WCLFightMetrics ready for the comparison pipeline.
    """
    duration_ms = int(fight.end_time_ms - fight.start_time_ms)
    active_dps = fight.total_damage / (duration_ms / 1000) if duration_ms > 0 else 0.0

    ability_breakdown: dict[str, AbilityMetrics] = {}
    for name, damage in fight.damage_by_ability.items():
        ability_breakdown[name] = AbilityMetrics(
            damage_total=damage,
            damage_pct=(damage / fight.total_damage * 100) if fight.total_damage > 0 else 0.0,
            cast_count=fight.casts_by_ability.get(name, 0),
        )

    buff_uptimes: dict[str, float] = {}
    for name, bt in fight.buff_trackers.items():
        if duration_ms > 0:
            buff_uptimes[name] = bt.total_uptime_ms / duration_ms

    return WCLFightMetrics(
        total_damage=fight.total_damage,
        active_dps=active_dps,
        fight_duration_ms=duration_ms,
        ability_breakdown=ability_breakdown,
        buff_uptimes=buff_uptimes,
        proc_counts={},
    )
