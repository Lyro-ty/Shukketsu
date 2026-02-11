"""Tests for the graph_search agent tool."""

import sqlite3

import pytest

from code.shukketsu.rag.entities import EntityType, RelationType
from code.shukketsu.rag.graph import GraphStore
from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool


@pytest.fixture
def graph(test_db: sqlite3.Connection) -> GraphStore:
    """GraphStore with seeded entity types."""
    store = GraphStore(test_db)
    store.seed_entity_types()
    return store


@pytest.fixture
def populated_graph(graph: GraphStore, test_db: sqlite3.Connection) -> GraphStore:
    """GraphStore with a realistic set of test entities and relationships."""
    # Insert source + chunks first (FK constraints)
    test_db.execute("INSERT INTO sources (id, url, title) VALUES (1, 'https://example.com', 'Test Source')")
    test_db.execute("INSERT INTO chunks (id, source_id, content, chunk_index) VALUES (42, 1, 'DST drops from Gruul', 0)")
    test_db.execute("INSERT INTO chunks (id, source_id, content, chunk_index) VALUES (78, 1, 'Gruul encounter', 1)")
    test_db.commit()

    # Entities
    dst_id = graph.upsert_entity(
        "Dragonspine Trophy", EntityType.ITEM, source_chunk_id=42, confidence=0.9
    )
    gruul_id = graph.upsert_entity(
        "Gruul the Dragonkiller", EntityType.BOSS, source_chunk_id=78, confidence=0.9
    )
    phase1_id = graph.upsert_entity("Phase 1", EntityType.PHASE, confidence=0.95)
    haste_id = graph.upsert_entity("haste proc", EntityType.STAT, confidence=0.85)
    combat_spec_id = graph.upsert_entity("Combat Swords", EntityType.SPEC, confidence=0.7)
    test_db.commit()

    # Relationships
    graph.upsert_relationship(
        dst_id, gruul_id, RelationType.DROPS_FROM, source_chunk_id=42, confidence=0.9
    )
    graph.upsert_relationship(
        dst_id, phase1_id, RelationType.AVAILABLE_IN, confidence=0.95
    )
    graph.upsert_relationship(
        dst_id, haste_id, RelationType.HAS_STAT, confidence=0.85
    )
    graph.upsert_relationship(
        combat_spec_id, dst_id, RelationType.BEST_IN_SLOT, confidence=0.7
    )
    test_db.commit()

    return graph


@pytest.fixture
def tool(test_db: sqlite3.Connection) -> GraphSearchTool:
    """GraphSearchTool backed by the test database."""
    return GraphSearchTool(test_db)


class TestGraphSearchToolMeta:
    """Tests for tool metadata."""

    def test_has_correct_name(self, tool: GraphSearchTool) -> None:
        assert tool.name == "graph_search"

    def test_has_description(self, tool: GraphSearchTool) -> None:
        assert "knowledge graph" in tool.description.lower()

    def test_has_parameters_schema(self, tool: GraphSearchTool) -> None:
        assert "entity" in tool.parameters_schema
        assert "relation_types" in tool.parameters_schema
        assert "target_type" in tool.parameters_schema


class TestGraphSearchEntityLookup:
    """Tests for entity lookup."""

    async def test_search_existing_entity_returns_relationships(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "Dragonspine Trophy" in result
        assert "drops_from" in result
        assert "Gruul" in result

    async def test_search_with_alias_resolves_canonical(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "DST"})
        assert "Dragonspine Trophy" in result
        assert "drops_from" in result

    async def test_search_nonexistent_entity_returns_helpful_message(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Thunderfury"})
        assert "no entity found" in result.lower()
        assert "rag_search" in result.lower()

    async def test_search_ambiguous_entity_returns_disambiguation(
        self, graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        graph.upsert_entity("Combat", EntityType.SPEC)
        graph.upsert_entity("Combat", EntityType.TALENT_TREE)
        test_db.commit()
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Combat"})
        assert "multiple entities" in result.lower()
        assert "spec" in result.lower()
        assert "talent_tree" in result.lower()


class TestGraphSearchFiltering:
    """Tests for relationship filtering."""

    async def test_filter_by_relation_type(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute(
            {"entity": "Dragonspine Trophy", "relation_types": ["drops_from"]}
        )
        assert "drops_from" in result
        assert "available_in" not in result

    async def test_filter_by_target_type(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute(
            {"entity": "Dragonspine Trophy", "target_type": "boss"}
        )
        assert "Gruul" in result
        assert "Phase 1" not in result

    async def test_both_filters_combined(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute(
            {"entity": "Dragonspine Trophy", "relation_types": ["drops_from"], "target_type": "boss"}
        )
        assert "Gruul" in result
        assert "Phase 1" not in result
        assert "haste" not in result.lower()


class TestGraphSearchBidirectional:
    """Tests for bidirectional relationship display."""

    async def test_outgoing_shown_with_arrow(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "\u2192" in result
        assert "Outgoing" in result

    async def test_incoming_shown_with_reverse_arrow(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "\u2190" in result
        assert "Incoming" in result

    async def test_entity_with_both_directions(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "Outgoing" in result
        assert "Incoming" in result


class TestGraphSearchOutputFormat:
    """Tests for output formatting."""

    async def test_related_chunks_collected(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "Related chunks:" in result
        assert "42" in result

    async def test_null_source_chunk_ids_filtered(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        # "None" should not appear in the chunks section
        chunks_section = result.split("Related chunks:")[-1] if "Related chunks:" in result else ""
        assert "None" not in chunks_section

    async def test_results_sorted_by_confidence(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        # available_in (0.95) should appear before drops_from (0.9) in outgoing
        avail_pos = result.find("available_in")
        drops_pos = result.find("drops_from")
        assert avail_pos < drops_pos

    async def test_entity_header_includes_type_and_confidence(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "item" in result.lower()
        assert "0.9" in result
