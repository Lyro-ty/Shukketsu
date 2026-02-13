"""Raid buff, debuff, and consumable registry with stacking rules.

Provides static definitions for all TBC raid buffs, boss debuffs, and
consumables relevant to Rogue DPS. Handles category-based stacking
(same category = highest wins) and flask/elixir exclusivity.
"""

import logging

from pydantic import BaseModel

from code.shukketsu.sim.models import BuffCategory, ProcEffect, ProcTrigger

logger = logging.getLogger(__name__)


# --- Models ---


class BuffDef(BaseModel):
    """Definition of a raid buff, debuff, or consumable."""

    name: str
    buff_id: str
    category: BuffCategory
    stats: dict[str, float] = {}
    proc: ProcEffect | None = None
    blocks_poison: bool = False
    description: str = ""


class ResolvedBuffs(BaseModel):
    """Aggregated buff effects after stacking rules applied."""

    flat_stats: dict[str, float] = {}
    stat_multipliers: dict[str, float] = {}
    active_procs: list[ProcEffect] = []
    boss_armor_reduction: int = 0
    active_buff_ids: set[str] = set()


# --- Static Registries ---

RAID_BUFFS: dict[str, BuffDef] = {
    "kings": BuffDef(
        name="Blessing of Kings",
        buff_id="kings",
        category=BuffCategory.STATS_PCT,
        stats={"stats_pct": 0.10},
        description="Blessing of Kings",
    ),
    "battle_shout": BuffDef(
        name="Improved Battle Shout",
        buff_id="battle_shout",
        category=BuffCategory.ATTACK_POWER,
        stats={"attack_power": 382},
        description="Improved Battle Shout",
    ),
    "grace_of_air": BuffDef(
        name="Improved Grace of Air Totem",
        buff_id="grace_of_air",
        category=BuffCategory.AGILITY_FLAT,
        stats={"agility": 88},
        description="Improved Grace of Air Totem",
    ),
    "strength_of_earth": BuffDef(
        name="Improved Strength of Earth",
        buff_id="strength_of_earth",
        category=BuffCategory.STRENGTH_FLAT,
        stats={"strength": 98},
        description="Improved Strength of Earth",
    ),
    "motw": BuffDef(
        name="Improved Mark of the Wild",
        buff_id="motw",
        category=BuffCategory.UNCATEGORIZED,
        stats={"agility": 14, "strength": 14, "stamina": 14},
        description="Improved Mark of the Wild",
    ),
    "lotp": BuffDef(
        name="Leader of the Pack",
        buff_id="lotp",
        category=BuffCategory.MELEE_CRIT,
        stats={"melee_crit_pct": 0.05},
        description="Leader of the Pack",
    ),
    "wf_totem": BuffDef(
        name="Windfury Totem",
        buff_id="wf_totem",
        category=BuffCategory.UNCATEGORIZED,
        proc=ProcEffect(
            trigger=ProcTrigger.ON_HIT,
            rate=0.20,
            icd=3.0,
            duration=0.0,
            effect={"extra_attacks": 1},
        ),
        description="Windfury Totem",
    ),
    "trueshot_aura": BuffDef(
        name="Trueshot Aura",
        buff_id="trueshot_aura",
        category=BuffCategory.UNCATEGORIZED,
        stats={"attack_power": 125},
        description="Trueshot Aura",
    ),
    "heroism": BuffDef(
        name="Heroism / Bloodlust",
        buff_id="heroism",
        category=BuffCategory.UNCATEGORIZED,
        proc=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            duration=40.0,
            effect={"haste_pct": 0.30},
        ),
        description="Heroism / Bloodlust",
    ),
    "drums_of_battle": BuffDef(
        name="Drums of Battle",
        buff_id="drums_of_battle",
        category=BuffCategory.UNCATEGORIZED,
        stats={"haste_rating": 80},
        description="Drums of Battle",
    ),
}

BOSS_DEBUFFS: dict[str, BuffDef] = {
    "sunder_armor": BuffDef(
        name="Sunder Armor (5 stacks)",
        buff_id="sunder_armor",
        category=BuffCategory.UNCATEGORIZED,
        stats={"boss_armor_reduction": 2600},
        description="Sunder Armor (5 stacks)",
    ),
    "faerie_fire": BuffDef(
        name="Faerie Fire",
        buff_id="faerie_fire",
        category=BuffCategory.UNCATEGORIZED,
        stats={"boss_armor_reduction": 610},
        description="Faerie Fire",
    ),
    "curse_of_recklessness": BuffDef(
        name="Curse of Recklessness",
        buff_id="curse_of_recklessness",
        category=BuffCategory.UNCATEGORIZED,
        stats={"boss_armor_reduction": 800},
        description="Curse of Recklessness",
    ),
}

CONSUMABLES: dict[str, BuffDef] = {
    "flask_relentless_assault": BuffDef(
        name="Flask of Relentless Assault",
        buff_id="flask_relentless_assault",
        category=BuffCategory.FLASK,
        stats={"attack_power": 120},
        description="Flask of Relentless Assault",
    ),
    "elixir_major_agility": BuffDef(
        name="Elixir of Major Agility",
        buff_id="elixir_major_agility",
        category=BuffCategory.BATTLE_ELIXIR,
        stats={"agility": 35, "crit_rating": 20},
        description="Elixir of Major Agility",
    ),
    "elixir_draenic_wisdom": BuffDef(
        name="Elixir of Draenic Wisdom",
        buff_id="elixir_draenic_wisdom",
        category=BuffCategory.GUARDIAN_ELIXIR,
        stats={},
        description="Elixir of Draenic Wisdom",
    ),
    "food_warp_burger": BuffDef(
        name="Warp Burger",
        buff_id="food_warp_burger",
        category=BuffCategory.FOOD,
        stats={"agility": 20},
        description="Warp Burger",
    ),
    "food_clefthoof": BuffDef(
        name="Roasted Clefthoof",
        buff_id="food_clefthoof",
        category=BuffCategory.FOOD,
        stats={"strength": 20},
        description="Roasted Clefthoof",
    ),
    "haste_potion": BuffDef(
        name="Haste Potion",
        buff_id="haste_potion",
        category=BuffCategory.UNCATEGORIZED,
        proc=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            duration=15.0,
            effect={"haste_rating": 400},
        ),
        description="Haste Potion",
    ),
}

RAID_PRESETS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "full_25man": (
        [
            "kings",
            "battle_shout",
            "grace_of_air",
            "strength_of_earth",
            "motw",
            "lotp",
            "wf_totem",
            "trueshot_aura",
        ],
        ["sunder_armor", "faerie_fire", "curse_of_recklessness"],
        ["flask_relentless_assault", "food_warp_burger"],
    ),
    "karazhan_10man": (
        ["kings", "battle_shout", "grace_of_air", "lotp"],
        ["sunder_armor", "faerie_fire"],
        ["flask_relentless_assault", "food_warp_burger"],
    ),
    "solo": ([], [], []),
    "custom": ([], [], []),
}


# --- Functions ---


def _stat_total(buff: BuffDef) -> float:
    """Sum all stat values on a buff for comparison purposes."""
    return sum(buff.stats.values())


def _is_pct_stat(key: str) -> bool:
    """Return True if the stat key is a percentage multiplier."""
    return key.endswith("_pct")


def resolve_buffs(
    buff_ids: list[str],
    debuff_ids: list[str],
    consumable_ids: list[str],
) -> ResolvedBuffs:
    """Apply stacking rules and aggregate all buff effects.

    Same BuffCategory (non-UNCATEGORIZED) -> highest total stat value wins.
    Different categories or UNCATEGORIZED -> all stack additively.
    Flask present -> elixirs (BATTLE_ELIXIR, GUARDIAN_ELIXIR) are excluded.
    """
    # Collect all valid BuffDefs, skipping unknown IDs
    all_defs: list[BuffDef] = []

    for bid in buff_ids:
        if bid in RAID_BUFFS:
            all_defs.append(RAID_BUFFS[bid])
        else:
            logger.warning("Unknown raid buff ID: %s (skipped)", bid)

    for did in debuff_ids:
        if did in BOSS_DEBUFFS:
            all_defs.append(BOSS_DEBUFFS[did])
        else:
            logger.warning("Unknown boss debuff ID: %s (skipped)", did)

    # Handle flask/elixir exclusivity for consumables
    has_flask = any(cid in CONSUMABLES and CONSUMABLES[cid].category == BuffCategory.FLASK for cid in consumable_ids)
    for cid in consumable_ids:
        if cid not in CONSUMABLES:
            logger.warning("Unknown consumable ID: %s (skipped)", cid)
            continue
        c_def = CONSUMABLES[cid]
        if has_flask and c_def.category in (BuffCategory.BATTLE_ELIXIR, BuffCategory.GUARDIAN_ELIXIR):
            logger.debug("Skipping elixir %s (flask active)", cid)
            continue
        all_defs.append(c_def)

    # Apply category stacking: same category -> highest wins
    category_best: dict[BuffCategory, BuffDef] = {}
    uncategorized: list[BuffDef] = []

    for buff in all_defs:
        if buff.category == BuffCategory.UNCATEGORIZED:
            uncategorized.append(buff)
        else:
            existing = category_best.get(buff.category)
            if existing is None or _stat_total(buff) > _stat_total(existing):
                category_best[buff.category] = buff

    # Merge winners + all uncategorized
    kept = list(category_best.values()) + uncategorized

    # Aggregate into ResolvedBuffs
    flat_stats: dict[str, float] = {}
    stat_multipliers: dict[str, float] = {}
    active_procs: list[ProcEffect] = []
    boss_armor_reduction = 0
    active_buff_ids: set[str] = set()

    for buff in kept:
        active_buff_ids.add(buff.buff_id)

        for key, val in buff.stats.items():
            if key == "boss_armor_reduction":
                boss_armor_reduction += int(val)
            elif _is_pct_stat(key):
                stat_multipliers[key] = stat_multipliers.get(key, 0.0) + val
            else:
                flat_stats[key] = flat_stats.get(key, 0.0) + val

        if buff.proc is not None:
            active_procs.append(buff.proc)

    return ResolvedBuffs(
        flat_stats=flat_stats,
        stat_multipliers=stat_multipliers,
        active_procs=active_procs,
        boss_armor_reduction=boss_armor_reduction,
        active_buff_ids=active_buff_ids,
    )


def get_preset(name: str) -> tuple[list[str], list[str], list[str]]:
    """Return (buffs, debuffs, consumables) for a preset name.

    Args:
        name: Preset name (e.g. "full_25man", "solo").

    Returns:
        Tuple of (buff_ids, debuff_ids, consumable_ids).

    Raises:
        KeyError: If preset name not found.
    """
    return RAID_PRESETS[name]
