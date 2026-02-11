"""Tests for WoW TBC entity types and extraction models."""

import pytest

from code.shukketsu.rag.entities import (
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
)


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_types_defined(self) -> None:
        """EntityType should have all WoW TBC entity types."""
        expected = {
            "item", "spell", "talent", "talent_tree", "spec", "boss",
            "instance", "phase", "stat", "consumable", "enchant", "gem",
            "profession", "buff", "debuff", "mechanic", "slot",
        }
        assert {e.value for e in EntityType} == expected

    def test_is_str_enum(self) -> None:
        """EntityType values should be strings for JSON serialization."""
        assert EntityType.ITEM == "item"
        assert isinstance(EntityType.ITEM, str)


class TestRelationType:
    """Tests for RelationType enum."""

    def test_all_types_defined(self) -> None:
        """RelationType should have all WoW TBC relationship types."""
        expected = {
            "drops_from", "available_in", "equips_in", "has_stat",
            "crafted_by", "best_in_slot", "belongs_to", "spec_uses",
            "benefits_from", "synergizes_with", "affected_by",
            "threshold_at", "counters", "applies", "contains",
            "has_mechanic", "requires",
        }
        assert {r.value for r in RelationType} == expected

    def test_is_str_enum(self) -> None:
        """RelationType values should be strings for JSON serialization."""
        assert RelationType.DROPS_FROM == "drops_from"
        assert isinstance(RelationType.DROPS_FROM, str)


class TestExtractedEntity:
    """Tests for ExtractedEntity Pydantic model."""

    def test_valid_entity(self) -> None:
        entity = ExtractedEntity(
            name="Dragonspine Trophy",
            entity_type=EntityType.ITEM,
        )
        assert entity.name == "Dragonspine Trophy"
        assert entity.entity_type == EntityType.ITEM
        assert entity.properties == {}

    def test_entity_with_properties(self) -> None:
        entity = ExtractedEntity(
            name="Dragonspine Trophy",
            entity_type=EntityType.ITEM,
            properties={"ilvl": 141, "slot": "trinket"},
        )
        assert entity.properties["ilvl"] == 141

    def test_entity_rejects_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            ExtractedEntity(name="Ghost", entity_type="nonexistent")


class TestExtractedRelationship:
    """Tests for ExtractedRelationship Pydantic model."""

    def test_valid_relationship(self) -> None:
        rel = ExtractedRelationship(
            source="Dragonspine Trophy",
            target="Gruul the Dragonkiller",
            relation_type=RelationType.DROPS_FROM,
        )
        assert rel.source == "Dragonspine Trophy"
        assert rel.target == "Gruul the Dragonkiller"
        assert rel.relation_type == RelationType.DROPS_FROM

    def test_relationship_with_properties(self) -> None:
        rel = ExtractedRelationship(
            source="Dragonspine Trophy",
            target="Phase 1",
            relation_type=RelationType.AVAILABLE_IN,
            properties={"note": "from launch"},
        )
        assert rel.properties["note"] == "from launch"

    def test_relationship_rejects_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            ExtractedRelationship(
                source="A", target="B", relation_type="invalid"
            )


class TestChunkExtraction:
    """Tests for ChunkExtraction model."""

    def test_empty_extraction(self) -> None:
        extraction = ChunkExtraction(entities=[], relationships=[])
        assert extraction.entities == []
        assert extraction.relationships == []

    def test_full_extraction(self) -> None:
        extraction = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Dragonspine Trophy", entity_type=EntityType.ITEM),
                ExtractedEntity(name="Gruul the Dragonkiller", entity_type=EntityType.BOSS),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Dragonspine Trophy",
                    target="Gruul the Dragonkiller",
                    relation_type=RelationType.DROPS_FROM,
                ),
            ],
        )
        assert len(extraction.entities) == 2
        assert len(extraction.relationships) == 1

    def test_serialization_round_trip(self) -> None:
        extraction = ChunkExtraction(
            entities=[
                ExtractedEntity(name="DST", entity_type=EntityType.ITEM),
            ],
            relationships=[],
        )
        data = extraction.model_dump()
        restored = ChunkExtraction.model_validate(data)
        assert restored.entities[0].name == "DST"

    def test_json_round_trip(self) -> None:
        extraction = ChunkExtraction(
            entities=[
                ExtractedEntity(
                    name="Sinister Strike",
                    entity_type=EntityType.SPELL,
                    properties={"energy_cost": 40},
                ),
            ],
            relationships=[],
        )
        json_str = extraction.model_dump_json()
        restored = ChunkExtraction.model_validate_json(json_str)
        assert restored.entities[0].properties["energy_cost"] == 40
