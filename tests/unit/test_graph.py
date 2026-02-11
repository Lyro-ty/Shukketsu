"""Tests for the knowledge graph storage layer."""

import sqlite3

import pytest

from code.shukketsu.rag.entities import EntityType, RelationType
from code.shukketsu.rag.graph import GraphStore


@pytest.fixture
def graph(test_db: sqlite3.Connection) -> GraphStore:
    """Create a GraphStore with seeded entity types."""
    store = GraphStore(test_db)
    store.seed_entity_types()
    return store


class TestSeedEntityTypes:
    """Tests for entity type seeding."""

    def test_seeds_all_types(self, test_db: sqlite3.Connection) -> None:
        store = GraphStore(test_db)
        store.seed_entity_types()
        count = test_db.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0]
        assert count == len(EntityType)

    def test_idempotent(self, test_db: sqlite3.Connection) -> None:
        store = GraphStore(test_db)
        store.seed_entity_types()
        store.seed_entity_types()  # second call should not fail
        count = test_db.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0]
        assert count == len(EntityType)

    def test_display_names(self, test_db: sqlite3.Connection) -> None:
        store = GraphStore(test_db)
        store.seed_entity_types()
        row = test_db.execute("SELECT display_name FROM entity_types WHERE name = 'talent_tree'").fetchone()
        assert row["display_name"] == "Talent Tree"


class TestGetEntityTypeId:
    """Tests for entity type ID lookup."""

    def test_returns_id(self, graph: GraphStore) -> None:
        type_id = graph.get_entity_type_id(EntityType.ITEM)
        assert isinstance(type_id, int)
        assert type_id > 0

    def test_raises_on_unseeded(self, test_db: sqlite3.Connection) -> None:
        store = GraphStore(test_db)
        # Don't seed — should raise
        with pytest.raises(ValueError, match="not seeded"):
            store.get_entity_type_id(EntityType.ITEM)


class TestUpsertEntity:
    """Tests for entity upsert."""

    def test_insert_new_entity(self, graph: GraphStore) -> None:
        entity_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        assert entity_id > 0
        row = graph._conn.execute("SELECT name, canonical_name FROM entities WHERE id = ?", (entity_id,)).fetchone()
        assert row["name"] == "Dragonspine Trophy"
        assert row["canonical_name"] == "dragonspine trophy"

    def test_dedup_by_canonical_name(self, graph: GraphStore) -> None:
        id1 = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        id2 = graph.upsert_entity("DST", EntityType.ITEM)  # alias resolves to same canonical
        assert id1 == id2

    def test_different_types_not_deduped(self, graph: GraphStore) -> None:
        id1 = graph.upsert_entity("Combat", EntityType.SPEC)
        id2 = graph.upsert_entity("Combat", EntityType.TALENT_TREE)
        assert id1 != id2

    def test_higher_confidence_updates(self, graph: GraphStore) -> None:
        graph.upsert_entity("DST", EntityType.ITEM, confidence=0.5)
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM, confidence=0.9)
        row = graph._conn.execute(
            "SELECT confidence FROM entities WHERE canonical_name = 'dragonspine trophy'"
        ).fetchone()
        assert row["confidence"] == 0.9

    def test_lower_confidence_does_not_downgrade(self, graph: GraphStore) -> None:
        graph.upsert_entity("DST", EntityType.ITEM, confidence=0.9)
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM, confidence=0.3)
        row = graph._conn.execute(
            "SELECT confidence FROM entities WHERE canonical_name = 'dragonspine trophy'"
        ).fetchone()
        assert row["confidence"] == 0.9

    def test_stores_properties(self, graph: GraphStore) -> None:
        graph.upsert_entity(
            "Dragonspine Trophy",
            EntityType.ITEM,
            properties={"ilvl": 141},
        )
        row = graph._conn.execute(
            "SELECT properties_json FROM entities WHERE canonical_name = 'dragonspine trophy'"
        ).fetchone()
        assert '"ilvl": 141' in row["properties_json"]

    def test_stores_source_chunk_id(self, graph: GraphStore) -> None:
        # Insert a source + chunk first
        graph._conn.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        graph._conn.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'test', 0)")
        graph._conn.commit()
        graph.upsert_entity("DST", EntityType.ITEM, source_chunk_id=1)
        row = graph._conn.execute(
            "SELECT source_chunk_id FROM entities WHERE canonical_name = 'dragonspine trophy'"
        ).fetchone()
        assert row["source_chunk_id"] == 1


class TestUpsertRelationship:
    """Tests for relationship upsert."""

    def test_insert_new_relationship(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul the Dragonkiller", EntityType.BOSS)
        rel_id = graph.upsert_relationship(
            item_id,
            boss_id,
            RelationType.DROPS_FROM,
        )
        assert rel_id > 0

    def test_dedup_relationship(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        id1 = graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        id2 = graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        assert id1 == id2

    def test_different_relation_types_not_deduped(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        phase_id = graph.upsert_entity("Phase 1", EntityType.PHASE)
        id1 = graph.upsert_relationship(item_id, phase_id, RelationType.AVAILABLE_IN)
        id2 = graph.upsert_relationship(item_id, phase_id, RelationType.BEST_IN_SLOT)
        assert id1 != id2

    def test_stores_properties(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("DST", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        graph.upsert_relationship(
            item_id,
            boss_id,
            RelationType.DROPS_FROM,
            properties={"drop_rate": 0.15},
        )
        row = graph._conn.execute("SELECT properties_json FROM relationships WHERE id = 1").fetchone()
        assert '"drop_rate": 0.15' in row["properties_json"]

    def test_higher_confidence_updates(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("DST", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM, confidence=0.5)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM, confidence=0.9)
        row = graph._conn.execute("SELECT confidence FROM relationships WHERE id = 1").fetchone()
        assert row["confidence"] == 0.9


class TestGetEntityByName:
    """Tests for entity lookup by name."""

    def test_finds_by_canonical(self, graph: GraphStore) -> None:
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        result = graph.get_entity_by_name("dragonspine trophy")
        assert result is not None
        assert result["name"] == "Dragonspine Trophy"

    def test_finds_by_alias(self, graph: GraphStore) -> None:
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        result = graph.get_entity_by_name("DST")
        assert result is not None
        assert result["canonical_name"] == "dragonspine trophy"

    def test_filters_by_type(self, graph: GraphStore) -> None:
        graph.upsert_entity("Combat", EntityType.SPEC)
        graph.upsert_entity("Combat", EntityType.TALENT_TREE)
        result = graph.get_entity_by_name("Combat", entity_type=EntityType.SPEC)
        assert result is not None
        type_id = graph.get_entity_type_id(EntityType.SPEC)
        assert result["entity_type_id"] == type_id

    def test_returns_none_for_missing(self, graph: GraphStore) -> None:
        result = graph.get_entity_by_name("nonexistent item")
        assert result is None


class TestGetRelationships:
    """Tests for relationship queries."""

    def test_outgoing_relationships(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        phase_id = graph.upsert_entity("Phase 1", EntityType.PHASE)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        graph.upsert_relationship(item_id, phase_id, RelationType.AVAILABLE_IN)
        rels = graph.get_relationships(item_id)
        assert len(rels) == 2

    def test_incoming_relationships(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        rels = graph.get_relationships(boss_id, direction="incoming")
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "drops_from"

    def test_filter_by_relation_type(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("DST", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        phase_id = graph.upsert_entity("Phase 1", EntityType.PHASE)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        graph.upsert_relationship(item_id, phase_id, RelationType.AVAILABLE_IN)
        rels = graph.get_relationships(
            item_id,
            relation_types=[RelationType.DROPS_FROM],
        )
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "drops_from"

    def test_empty_graph_returns_empty(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Lonely Item", EntityType.ITEM)
        rels = graph.get_relationships(item_id)
        assert rels == []


class TestFindEntitiesByName:
    """Tests for multi-match entity lookup."""

    def test_finds_single_match(self, graph: GraphStore) -> None:
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        results = graph.find_entities_by_name("Dragonspine Trophy")
        assert len(results) == 1
        assert results[0]["name"] == "Dragonspine Trophy"
        assert results[0]["entity_type_name"] == "item"

    def test_finds_multiple_types(self, graph: GraphStore) -> None:
        graph.upsert_entity("Combat", EntityType.SPEC)
        graph.upsert_entity("Combat", EntityType.TALENT_TREE)
        results = graph.find_entities_by_name("Combat")
        assert len(results) == 2
        type_names = {r["entity_type_name"] for r in results}
        assert type_names == {"spec", "talent_tree"}

    def test_resolves_aliases(self, graph: GraphStore) -> None:
        graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        results = graph.find_entities_by_name("DST")
        assert len(results) == 1
        assert results[0]["canonical_name"] == "dragonspine trophy"

    def test_filters_by_entity_type(self, graph: GraphStore) -> None:
        graph.upsert_entity("Combat", EntityType.SPEC)
        graph.upsert_entity("Combat", EntityType.TALENT_TREE)
        results = graph.find_entities_by_name("Combat", entity_type=EntityType.SPEC)
        assert len(results) == 1
        assert results[0]["entity_type_name"] == "spec"

    def test_returns_empty_for_missing(self, graph: GraphStore) -> None:
        results = graph.find_entities_by_name("nonexistent item")
        assert results == []

    def test_includes_entity_type_name(self, graph: GraphStore) -> None:
        graph.upsert_entity("Gruul", EntityType.BOSS)
        results = graph.find_entities_by_name("Gruul")
        assert results[0]["entity_type_name"] == "boss"


class TestGetRelationshipsWithType:
    """Tests that get_relationships includes entity type name."""

    def test_outgoing_includes_related_type(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        rels = graph.get_relationships(item_id)
        assert len(rels) == 1
        assert rels[0]["related_type"] == "boss"

    def test_incoming_includes_related_type(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Dragonspine Trophy", EntityType.ITEM)
        boss_id = graph.upsert_entity("Gruul", EntityType.BOSS)
        graph.upsert_relationship(item_id, boss_id, RelationType.DROPS_FROM)
        rels = graph.get_relationships(boss_id, direction="incoming")
        assert len(rels) == 1
        assert rels[0]["related_type"] == "item"
