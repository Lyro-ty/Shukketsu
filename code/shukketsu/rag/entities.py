"""WoW TBC entity types, relationship types, and extraction models.

Defines the domain schema for the knowledge graph. Entity extraction
uses these types to produce structured output from Llama 70B, which
is stored in the graph tables during ingest.
"""

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
