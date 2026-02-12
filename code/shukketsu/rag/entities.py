"""WoW TBC entity types, relationship types, and extraction models.

Defines the domain schema for the knowledge graph. Entity extraction
uses these types to produce structured output from Llama 70B, which
is stored in the graph tables during ingest.
"""

import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from code.shukketsu.llm.structured import ModelBackend, get_structured_output
from code.shukketsu.resilience.errors import EntityExtractionError, LLMUnavailableError, StructuredOutputError

logger = logging.getLogger(__name__)


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


EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction system for World of Warcraft: The Burning Crusade (TBC) Rogue content.

Given a text chunk, extract all named entities and relationships between them.

## Entity Types
- item: Equipment, weapons, trinkets (e.g., Dragonspine Trophy, Warglaive of Azzinoth)
- spell: Abilities and spells (e.g., Sinister Strike, Slice and Dice)
- talent: Individual talent points (e.g., Combat Potency, Surprise Attacks)
- talent_tree: Talent tree names (e.g., Combat, Assassination, Subtlety)
- spec: Specific builds (e.g., Combat Swords, Mutilate, Combat Fists)
- boss: Raid/dungeon bosses (e.g., Gruul the Dragonkiller, Illidan)
- instance: Dungeons and raids (e.g., Karazhan, Black Temple)
- phase: Content phases (e.g., Phase 1, Phase 2)
- stat: Character stats (e.g., Hit Rating, Crit, Attack Power, Haste)
- consumable: Potions, food, flasks (e.g., Haste Potion, Scorpid Surprise)
- enchant: Weapon/armor enchants (e.g., Mongoose, Executioner)
- gem: Socketed gems (e.g., Delicate Living Ruby)
- profession: Crafting professions (e.g., Leatherworking, Engineering)
- buff: Party/raid buffs (e.g., Windfury Totem, Blessing of Might)
- debuff: Debuffs applied to targets (e.g., Expose Armor, Sunder Armor)
- mechanic: Game mechanics (e.g., Hit table, Dual wield penalty)
- slot: Equipment slots (e.g., Main Hand, Off Hand, Trinket 1)

## Relationship Types
- drops_from: item -> boss
- available_in: item/instance -> phase
- equips_in: item -> slot
- has_stat: item/enchant/gem -> stat
- crafted_by: item -> profession
- best_in_slot: item -> spec (contextual BiS)
- belongs_to: talent -> talent_tree
- spec_uses: spec -> talent_tree
- benefits_from: spec -> stat/buff/item
- synergizes_with: talent <-> talent, spell <-> spell
- affected_by: spell -> stat/mechanic
- threshold_at: stat -> value (e.g., hit cap)
- counters: debuff -> boss mechanic
- applies: spell -> buff/debuff
- contains: instance -> boss
- has_mechanic: boss -> mechanic
- requires: instance -> attunement/gear level

## Rules
- Only extract entities explicitly mentioned in the text.
- Do NOT invent entities or relationships not supported by the text.
- Use the most specific entity type possible.
- Use the full proper name when possible (e.g., "Dragonspine Trophy" not "DST").
- If a relationship is implied but not stated, skip it.
- Return empty lists if no entities or relationships are found.
"""


def build_extraction_messages(chunk_text: str) -> list[dict[str, str]]:
    """Build chat messages for entity extraction from a chunk.

    Args:
        chunk_text: The text content to extract entities from.

    Returns:
        A list of system + user messages ready for the LLM.
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{chunk_text}"},
    ]


async def extract_entities_from_chunk(chunk_text: str) -> ChunkExtraction:
    """Extract entities and relationships from a text chunk using Llama 70B.

    Args:
        chunk_text: The text content to extract entities from.

    Returns:
        A ChunkExtraction with extracted entities and relationships.

    Raises:
        EntityExtractionError: If the LLM fails to produce valid output.
    """
    messages = build_extraction_messages(chunk_text)
    try:
        result: ChunkExtraction = await get_structured_output(
            ChunkExtraction,
            messages,
            backend=ModelBackend.REASONING,
            temperature=0.1,
            max_tokens=2048,
        )
        return result
    except (StructuredOutputError, LLMUnavailableError) as exc:
        raise EntityExtractionError(f"Failed to extract entities: {exc}") from exc
