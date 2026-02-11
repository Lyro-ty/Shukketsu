"""WoW TBC entity types, relationship types, and extraction models.

Defines the domain schema for the knowledge graph. Entity extraction
uses these types to produce structured output from Llama 70B, which
is stored in the graph tables during ingest.
"""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    """WoW TBC entity categories for the knowledge graph."""

    ITEM = "item"
    SPELL = "spell"
    TALENT = "talent"
    TALENT_TREE = "talent_tree"
    SPEC = "spec"
    BOSS = "boss"
    INSTANCE = "instance"
    PHASE = "phase"
    STAT = "stat"
    CONSUMABLE = "consumable"
    ENCHANT = "enchant"
    GEM = "gem"
    PROFESSION = "profession"
    BUFF = "buff"
    DEBUFF = "debuff"
    MECHANIC = "mechanic"
    SLOT = "slot"


class RelationType(StrEnum):
    """Typed edges between entities in the knowledge graph."""

    # Item relationships
    DROPS_FROM = "drops_from"
    AVAILABLE_IN = "available_in"
    EQUIPS_IN = "equips_in"
    HAS_STAT = "has_stat"
    CRAFTED_BY = "crafted_by"
    BEST_IN_SLOT = "best_in_slot"

    # Spec/talent relationships
    BELONGS_TO = "belongs_to"
    SPEC_USES = "spec_uses"
    BENEFITS_FROM = "benefits_from"
    SYNERGIZES_WITH = "synergizes_with"

    # Combat mechanics
    AFFECTED_BY = "affected_by"
    THRESHOLD_AT = "threshold_at"
    COUNTERS = "counters"
    APPLIES = "applies"

    # Instance/boss
    CONTAINS = "contains"
    HAS_MECHANIC = "has_mechanic"
    REQUIRES = "requires"


class ExtractedEntity(BaseModel):
    """An entity extracted from a text chunk by the LLM."""

    name: str
    entity_type: EntityType
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    """A relationship extracted from a text chunk by the LLM."""

    source: str
    target: str
    relation_type: RelationType
    properties: dict[str, Any] = Field(default_factory=dict)


class ChunkExtraction(BaseModel):
    """Complete extraction result from a single text chunk."""

    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


def normalize_name(name: str) -> str:
    """Normalize an entity name to canonical form.

    Lowercases, strips whitespace and leading articles (the/a/an),
    and collapses internal whitespace.
    """
    name = name.lower().strip()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


# Common WoW TBC abbreviations and alternate names.
# Keys MUST be in normalized form (lowercase, no leading articles).
KNOWN_ALIASES: dict[str, str] = {
    # Items
    "dst": "dragonspine trophy",
    "wg": "warglaive of azzinoth",
    "t4": "tier 4",
    "t5": "tier 5",
    "t6": "tier 6",
    # Spells / abilities
    "snd": "slice and dice",
    "ss": "sinister strike",
    "ar": "adrenaline rush",
    "bf": "blade flurry",
    "ks": "killing spree",
    "evis": "eviscerate",
    "mut": "mutilate",
    "hemo": "hemorrhage",
    "ea": "expose armor",
    "rupture": "rupture",
    # Talents
    "cqc": "close quarters combat",
    "cp": "combat potency",
    # Instances
    "bt": "black temple",
    "ssc": "serpentshrine cavern",
    "tk": "tempest keep",
    "kara": "karazhan",
    "gruuls": "gruul's lair",
    "mh": "mount hyjal",
    "za": "zul'aman",
    "mag": "magtheridon's lair",
    "sp": "shadow labyrinth",
    # Stats
    "ap": "attack power",
    "agi": "agility",
    "str": "strength",
    "sta": "stamina",
    "crit": "critical strike rating",
    "hit": "hit rating",
    "exp": "expertise rating",
    "haste": "haste rating",
    "arp": "armor penetration rating",
    # Specs
    "combat swords": "combat swords",
    "combat daggers": "combat daggers",
    "combat fists": "combat fists",
    # Consumables
    "thistle tea": "thistle tea",
}


def resolve_canonical(name: str) -> str:
    """Resolve a name to its canonical form, checking aliases.

    First normalizes the name, then checks the alias table.
    Returns the alias target if found, otherwise the normalized name.
    """
    normalized = normalize_name(name)
    return KNOWN_ALIASES.get(normalized, normalized)
