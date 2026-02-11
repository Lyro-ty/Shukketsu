# Phase 2 Step 3: Graph Search Tool + Qwen 4B Reranker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a graph_search agent tool for knowledge graph traversal, and a Qwen 4B reranker that improves rag_search result quality.

**Architecture:** Two independent components. GraphSearchTool queries entities/relationships from SQLite, formats them for the LLM agent. The reranker is a shared module (`rag/reranker.py`) that rag_search calls internally — over-retrieve 3x, Qwen 4B scores HIGH/MEDIUM/LOW, return top_k. Circuit breaker wraps the reranker with fallback to unreranked results.

**Tech Stack:** SQLite (knowledge graph), Pydantic v2 (schemas), Instructor + Ollama Qwen 4B (reranker), pytest (TDD)

**Design doc:** `docs/plans/2026-02-10-phase2-step3-graph-search-reranker.md`

**Starting state:** 381 unit tests passing. Phase 2 Steps 1-2 complete.

**Test command:** `python3 -m pytest tests/unit/ -v` (MUST use `python3 -m pytest`, not bare `pytest`)

---

## Task 1: Config + Error Classes + Circuit Breaker

Foundation constants and error types that everything else depends on.

**Files:**
- Modify: `code/shukketsu/config.py:89` (after CB_BRAVE lines)
- Modify: `code/shukketsu/resilience/errors.py:127` (append after CircuitOpenError)
- Modify: `code/shukketsu/resilience/circuit_breaker.py:134-144` (after brave_breaker, update reset_all_breakers)
- Test: `tests/unit/test_step3_foundation.py` (new file)

**Step 1: Write the failing tests**

```python
"""Tests for Phase 2 Step 3 foundation: config, errors, circuit breaker."""

import pytest

from code.shukketsu import config
from code.shukketsu.resilience.errors import (
    FailureMode,
    GraphTraversalError,
    RerankerError,
    ShukketsuError,
)


class TestNewErrorClasses:
    """Tests for RerankerError and GraphTraversalError."""

    def test_reranker_error_inherits_shukketsu_error(self) -> None:
        err = RerankerError("reranker failed")
        assert isinstance(err, ShukketsuError)
        assert err.failure_mode == FailureMode.MODEL_UNAVAILABLE
        assert "reranker failed" in str(err)

    def test_graph_traversal_error_inherits_shukketsu_error(self) -> None:
        err = GraphTraversalError("graph query failed")
        assert isinstance(err, ShukketsuError)
        assert err.failure_mode == FailureMode.DB_ERROR
        assert "graph query failed" in str(err)


class TestNewConfig:
    """Tests for new config constants."""

    def test_graph_search_default_top_k_exists(self) -> None:
        assert hasattr(config, "GRAPH_SEARCH_DEFAULT_TOP_K")
        assert isinstance(config.GRAPH_SEARCH_DEFAULT_TOP_K, int)
        assert config.GRAPH_SEARCH_DEFAULT_TOP_K == 20

    def test_reranker_fetch_multiplier_exists(self) -> None:
        assert hasattr(config, "RERANKER_FETCH_MULTIPLIER")
        assert isinstance(config.RERANKER_FETCH_MULTIPLIER, int)
        assert config.RERANKER_FETCH_MULTIPLIER == 3

    def test_reranker_top_k_exists(self) -> None:
        assert hasattr(config, "RERANKER_TOP_K")
        assert isinstance(config.RERANKER_TOP_K, int)
        assert config.RERANKER_TOP_K == 5

    def test_cb_qwen_reranker_constants_exist(self) -> None:
        assert config.CB_QWEN_RERANKER_FAILURE_THRESHOLD == 3
        assert config.CB_QWEN_RERANKER_RECOVERY_TIMEOUT == 30.0


class TestQwenRerankerBreaker:
    """Tests for the qwen_reranker circuit breaker instance."""

    def test_breaker_exists(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker

        assert qwen_reranker_breaker.name == "qwen_reranker"

    def test_breaker_has_correct_threshold(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker

        assert qwen_reranker_breaker._failure_threshold == 3

    def test_reset_all_breakers_includes_new_breaker(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import (
            qwen_reranker_breaker,
            reset_all_breakers,
        )

        # Force breaker into failure state
        qwen_reranker_breaker._failure_count = 10
        reset_all_breakers()
        assert qwen_reranker_breaker._failure_count == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_step3_foundation.py -v`
Expected: FAIL — `RerankerError` and `GraphTraversalError` not defined, config attrs missing, `qwen_reranker_breaker` not defined.

**Step 3: Implement**

Add to `code/shukketsu/config.py` after line 89 (after `CB_BRAVE_RECOVERY_TIMEOUT`):

```python
CB_QWEN_RERANKER_FAILURE_THRESHOLD = int(os.getenv("CB_QWEN_RERANKER_FAILURE_THRESHOLD", "3"))
CB_QWEN_RERANKER_RECOVERY_TIMEOUT = float(os.getenv("CB_QWEN_RERANKER_RECOVERY_TIMEOUT", "30"))
```

Add after line 105 (after Phase 2 agent limits section):

```python
# Graph search
GRAPH_SEARCH_DEFAULT_TOP_K = int(os.getenv("GRAPH_SEARCH_DEFAULT_TOP_K", "20"))

# Reranker
RERANKER_FETCH_MULTIPLIER = int(os.getenv("RERANKER_FETCH_MULTIPLIER", "3"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
```

Add to `code/shukketsu/resilience/errors.py` after `CircuitOpenError` class (line 127):

```python
class RerankerError(ShukketsuError):
    """Raised when the reranker LLM call fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.MODEL_UNAVAILABLE)


class GraphTraversalError(ShukketsuError):
    """Raised when a knowledge graph query fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.DB_ERROR)
```

Add to `code/shukketsu/resilience/circuit_breaker.py` after `brave_breaker` (line 138):

```python
qwen_reranker_breaker = CircuitBreaker(
    "qwen_reranker",
    failure_threshold=config.CB_QWEN_RERANKER_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_QWEN_RERANKER_RECOVERY_TIMEOUT,
)
```

Update `reset_all_breakers()` (line 141-144) to include the new breaker:

```python
def reset_all_breakers() -> None:
    """Reset all named circuit breakers to CLOSED. Used by test fixtures."""
    for breaker in (
        reasoning_breaker,
        ollama_router_breaker,
        ollama_embed_breaker,
        brave_breaker,
        qwen_reranker_breaker,
    ):
        breaker.reset()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_step3_foundation.py -v`
Expected: 8 PASSED

Run: `python3 -m pytest tests/unit/ -v`
Expected: 389 passed (381 + 8 new). No regressions — existing tests still pass since `reset_all_breakers` now also resets the new breaker.

**Step 5: Commit**

```bash
git add code/shukketsu/config.py code/shukketsu/resilience/errors.py code/shukketsu/resilience/circuit_breaker.py tests/unit/test_step3_foundation.py
git commit -m "feat(resilience): add reranker/graph errors, circuit breaker, config constants"
```

---

## Task 2: GraphStore Extensions

Add `find_entities_by_name()` and extend `get_relationships()` to include entity type names.

**Files:**
- Modify: `code/shukketsu/rag/graph.py:136-194` (add method, modify query)
- Test: `tests/unit/test_graph.py` (add tests to existing file)

**Step 1: Write the failing tests**

Add to `tests/unit/test_graph.py` at the bottom:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_graph.py::TestFindEntitiesByName -v`
Expected: FAIL — `find_entities_by_name` not defined.

Run: `python3 -m pytest tests/unit/test_graph.py::TestGetRelationshipsWithType -v`
Expected: FAIL — `related_type` key not in rows.

**Step 3: Implement**

Add to `code/shukketsu/rag/graph.py` after `get_entity_by_name` (after line 155):

```python
    def find_entities_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> list[sqlite3.Row]:
        """Look up all entities matching a name (resolves aliases).

        Unlike get_entity_by_name() which returns one row, this returns all
        matches — needed for ambiguity detection in the graph_search tool.
        """
        canonical = resolve_canonical(name)
        if entity_type is not None:
            type_id = self.get_entity_type_id(entity_type)
            return self._conn.execute(
                "SELECT e.*, et.name AS entity_type_name "
                "FROM entities e JOIN entity_types et ON et.id = e.entity_type_id "
                "WHERE e.canonical_name = ? AND e.entity_type_id = ?",
                (canonical, type_id),
            ).fetchall()
        return self._conn.execute(
            "SELECT e.*, et.name AS entity_type_name "
            "FROM entities e JOIN entity_types et ON et.id = e.entity_type_id "
            "WHERE e.canonical_name = ?",
            (canonical,),
        ).fetchall()
```

Modify `get_relationships()` query (line 181-185) — add `entity_types` JOIN and `related_type`:

```python
        query = (
            f"SELECT r.*, e.name AS related_name, e.canonical_name AS related_canonical, "
            f"et.name AS related_type "
            f"FROM relationships r "
            f"JOIN entities e ON e.id = r.{join_col} "
            f"JOIN entity_types et ON et.id = e.entity_type_id "
            f"WHERE r.{col} = ?"
        )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_graph.py -v`
Expected: All graph tests pass (existing 25 + 8 new = 33). Existing tests pass since `related_type` is additive — no existing assertions break.

Run: `python3 -m pytest tests/unit/ -v`
Expected: 397 passed (389 + 8 new).

**Step 5: Commit**

```bash
git add code/shukketsu/rag/graph.py tests/unit/test_graph.py
git commit -m "feat(rag): add find_entities_by_name and entity type to relationships"
```

---

## Task 3: GraphSearchTool — Core

The graph search agent tool: entity lookup, bidirectional relationships, output formatting.

**Files:**
- Create: `code/shukketsu/tools/knowledge/graph_search.py`
- Modify: `code/shukketsu/tools/knowledge/__init__.py`
- Test: `tests/unit/test_graph_search_tool.py` (new file)

**Step 1: Write the failing tests**

Create `tests/unit/test_graph_search_tool.py`:

```python
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
        assert "→" in result
        assert "Outgoing" in result

    async def test_incoming_shown_with_reverse_arrow(
        self, populated_graph: GraphStore, test_db: sqlite3.Connection
    ) -> None:
        tool = GraphSearchTool(test_db)
        result = await tool.execute({"entity": "Dragonspine Trophy"})
        assert "←" in result
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_graph_search_tool.py -v`
Expected: FAIL — `GraphSearchTool` module does not exist.

**Step 3: Implement**

Create `code/shukketsu/tools/knowledge/graph_search.py`:

```python
"""Graph traversal tool for agent knowledge graph queries."""

import logging
import sqlite3
from typing import Any

from code.shukketsu import config
from code.shukketsu.rag.entities import EntityType, RelationType, resolve_canonical
from code.shukketsu.rag.graph import GraphStore
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


def _parse_relation_types(raw: list[str] | None) -> list[RelationType] | None:
    """Convert raw strings to RelationType enums, skipping invalid ones."""
    if not raw:
        return None
    valid = []
    for rt_str in raw:
        try:
            valid.append(RelationType(rt_str))
        except ValueError:
            logger.warning("Unknown relation type '%s', skipping", rt_str)
    return valid or None


class GraphSearchTool(Tool):
    """Search the WoW TBC knowledge graph for entity relationships.

    Returns entity-centric results: the queried entity's outgoing and
    incoming relationships, sorted by confidence, with linked chunk IDs.
    """

    name = "graph_search"
    description = (
        "Search the WoW TBC knowledge graph for entity relationships. "
        "Use when you need connections between game concepts: "
        '"What items drop from [boss]?", "What stats does [item] have?", '
        '"What\'s BiS for [spec] in [phase]?", "What talents synergize with [spell]?" '
        "Returns entity relationships and linked chunk IDs for follow-up with rag_search."
    )
    parameters_schema: dict[str, Any] = {
        "entity": {
            "type": "str",
            "description": "Entity name to search for (e.g., 'Dragonspine Trophy', 'Gruul', 'DST')",
        },
        "relation_types": {
            "type": "list[str]",
            "description": "Filter by relationship type (e.g., ['drops_from', 'has_stat'])",
            "optional": True,
        },
        "target_type": {
            "type": "str",
            "description": "Filter target entity type (e.g., 'boss', 'stat')",
            "optional": True,
        },
    }

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._graph = GraphStore(conn)

    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute a graph search for entity relationships."""
        entity_name = tool_input.get("entity", "")
        if not entity_name:
            return "Error: 'entity' parameter is required."

        raw_relation_types = tool_input.get("relation_types")
        target_type = tool_input.get("target_type")
        relation_types = _parse_relation_types(raw_relation_types)

        try:
            return self._search(entity_name, relation_types, target_type)
        except Exception as exc:
            logger.exception("Graph search failed for entity '%s'", entity_name)
            return f"Graph search failed: {exc}. Try rag_search instead."

    def _search(
        self,
        entity_name: str,
        relation_types: list[RelationType] | None,
        target_type: str | None,
    ) -> str:
        """Core search logic, separated for testability."""
        # Find all matching entities
        matches = self._graph.find_entities_by_name(entity_name)

        if not matches:
            return (
                f"No entity found matching '{entity_name}'. "
                "Try a different name or use rag_search for text-based search."
            )

        if len(matches) > 1:
            return self._format_disambiguation(entity_name, matches)

        entity = matches[0]
        entity_id = entity["id"]

        # Fetch both directions
        outgoing = self._graph.get_relationships(
            entity_id, relation_types=relation_types, direction="outgoing"
        )
        incoming = self._graph.get_relationships(
            entity_id, relation_types=relation_types, direction="incoming"
        )

        # Apply target_type filter
        if target_type:
            outgoing = [r for r in outgoing if r["related_type"] == target_type]
            incoming = [r for r in incoming if r["related_type"] == target_type]

        # Sort by confidence descending
        outgoing = sorted(outgoing, key=lambda r: r["confidence"], reverse=True)
        incoming = sorted(incoming, key=lambda r: r["confidence"], reverse=True)

        # Truncate
        top_k = config.GRAPH_SEARCH_DEFAULT_TOP_K
        outgoing = outgoing[:top_k]
        incoming = incoming[:top_k]

        # Collect chunk IDs
        chunk_ids = self._collect_chunk_ids(entity, outgoing, incoming)

        return self._format_output(entity, outgoing, incoming, chunk_ids)

    def _format_disambiguation(self, name: str, matches: list[sqlite3.Row]) -> str:
        """Format a disambiguation message for ambiguous entity lookups."""
        lines = [f'Multiple entities match "{name}":']
        for m in matches:
            lines.append(f"- {m['name']} ({m['entity_type_name']})")
        lines.append("")
        lines.append("Please specify with the target_type parameter or use a more specific name.")
        return "\n".join(lines)

    def _collect_chunk_ids(
        self,
        entity: sqlite3.Row,
        outgoing: list[sqlite3.Row],
        incoming: list[sqlite3.Row],
    ) -> list[int]:
        """Collect unique source_chunk_ids from entity and relationships."""
        ids: set[int] = set()
        if entity["source_chunk_id"] is not None:
            ids.add(entity["source_chunk_id"])
        for rel in (*outgoing, *incoming):
            if rel["source_chunk_id"] is not None:
                ids.add(rel["source_chunk_id"])
        return sorted(ids)

    def _format_output(
        self,
        entity: sqlite3.Row,
        outgoing: list[sqlite3.Row],
        incoming: list[sqlite3.Row],
        chunk_ids: list[int],
    ) -> str:
        """Format the observation string for the agent."""
        lines: list[str] = []

        # Header
        lines.append(
            f"Entity: {entity['name']} ({entity['entity_type_name']}, "
            f"confidence: {entity['confidence']:.2f})"
        )
        lines.append("")

        # Outgoing
        if outgoing:
            lines.append(f"Outgoing relationships ({len(outgoing)}):")
            for rel in outgoing:
                lines.append(
                    f"- {rel['relation_type']} → {rel['related_name']} "
                    f"({rel['related_type']}, confidence: {rel['confidence']:.2f})"
                )
            lines.append("")

        # Incoming
        if incoming:
            lines.append(f"Incoming relationships ({len(incoming)}):")
            for rel in incoming:
                lines.append(
                    f"- {rel['relation_type']} ← {rel['related_name']} "
                    f"({rel['related_type']}, confidence: {rel['confidence']:.2f})"
                )
            lines.append("")

        # Chunks
        if chunk_ids:
            lines.append(f"Related chunks: {', '.join(str(c) for c in chunk_ids)}")
        else:
            lines.append("Related chunks: none")

        return "\n".join(lines)
```

Update `code/shukketsu/tools/knowledge/__init__.py`:

```python
"""Knowledge search tools for the agent."""
```

(The `__init__.py` is nearly empty — `GraphSearchTool` is imported directly by consumers. No re-export needed since the existing `RagSearchTool` is also imported directly.)

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_graph_search_tool.py -v`
Expected: 18 PASSED

Run: `python3 -m pytest tests/unit/ -v`
Expected: 415 passed (397 + 18 new).

**Step 5: Commit**

```bash
git add code/shukketsu/tools/knowledge/graph_search.py tests/unit/test_graph_search_tool.py
git commit -m "feat(tools): add graph_search tool for knowledge graph traversal"
```

---

## Task 4: Reranker Module — Schemas + Core Logic

The Qwen 4B reranker: Pydantic schemas, prompt building, index validation, categorical filtering.

**Files:**
- Create: `code/shukketsu/rag/reranker.py`
- Test: `tests/unit/test_reranker.py` (new file)

**Step 1: Write the failing tests**

Create `tests/unit/test_reranker.py`:

```python
"""Tests for the Qwen 4B categorical reranker."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.rag.reranker import (
    RankedItem,
    RerankerResponse,
    _apply_rankings,
    _build_rerank_messages,
    rerank,
)
from code.shukketsu.rag.search import SearchResult


def _make_result(chunk_id: int, content: str = "test", rrf_score: float = 0.5) -> SearchResult:
    """Helper to create SearchResult instances."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        source_title=f"Source {chunk_id}",
        source_url=f"https://example.com/{chunk_id}",
        trust_score=0.7,
        rrf_score=rrf_score,
    )


class TestRerankerSchemas:
    """Tests for Pydantic schemas."""

    def test_ranked_item_validates(self) -> None:
        item = RankedItem(index=0, relevance="HIGH", reason="directly relevant")
        assert item.relevance == "HIGH"

    def test_ranked_item_rejects_invalid_relevance(self) -> None:
        with pytest.raises(Exception):
            RankedItem(index=0, relevance="SUPER_HIGH", reason="nope")

    def test_reranker_response_validates(self) -> None:
        resp = RerankerResponse(
            rankings=[RankedItem(index=0, relevance="HIGH", reason="good")]
        )
        assert len(resp.rankings) == 1


class TestBuildMessages:
    """Tests for prompt construction."""

    def test_includes_query(self) -> None:
        results = [_make_result(1, "Hit cap is 142")]
        messages = _build_rerank_messages("hit cap", results)
        user_msg = messages[-1]["content"]
        assert "hit cap" in user_msg

    def test_includes_results_indexed(self) -> None:
        results = [_make_result(1, "Content A"), _make_result(2, "Content B")]
        messages = _build_rerank_messages("query", results)
        user_msg = messages[-1]["content"]
        assert "[0]" in user_msg
        assert "[1]" in user_msg

    def test_truncates_long_content(self) -> None:
        long_content = "x" * 500
        results = [_make_result(1, long_content)]
        messages = _build_rerank_messages("query", results)
        user_msg = messages[-1]["content"]
        # Should not contain the full 500 chars
        assert len(user_msg) < 400


class TestApplyRankings:
    """Tests for the index validation + categorical filtering logic."""

    def test_filters_low_keeps_high(self) -> None:
        results = [_make_result(i, rrf_score=0.5 - i * 0.1) for i in range(5)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="HIGH", reason=""),
            RankedItem(index=2, relevance="LOW", reason=""),
            RankedItem(index=3, relevance="LOW", reason=""),
            RankedItem(index=4, relevance="LOW", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 2  # only 2 HIGHs, no MEDIUMs to fill
        assert filtered[0].chunk_id == 0
        assert filtered[1].chunk_id == 1

    def test_fills_medium_up_to_top_k(self) -> None:
        results = [_make_result(i, rrf_score=0.5 - i * 0.1) for i in range(5)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="MEDIUM", reason=""),
            RankedItem(index=2, relevance="MEDIUM", reason=""),
            RankedItem(index=3, relevance="MEDIUM", reason=""),
            RankedItem(index=4, relevance="LOW", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 3
        assert filtered[0].chunk_id == 0  # HIGH first
        assert filtered[1].chunk_id == 1  # then MEDIUMs
        assert filtered[2].chunk_id == 2

    def test_preserves_original_order_within_category(self) -> None:
        results = [_make_result(i, rrf_score=1.0 - i * 0.1) for i in range(4)]
        rankings = [
            RankedItem(index=2, relevance="HIGH", reason=""),
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=3, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="HIGH", reason=""),
        ]
        # All HIGH — order should follow original rrf_score order (0, 1, 2, 3)
        filtered = _apply_rankings(results, rankings, top_k=4)
        assert [r.chunk_id for r in filtered] == [0, 1, 2, 3]

    def test_all_high_returns_top_k(self) -> None:
        results = [_make_result(i) for i in range(6)]
        rankings = [RankedItem(index=i, relevance="HIGH", reason="") for i in range(6)]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 3

    def test_ignores_out_of_range_indices(self) -> None:
        results = [_make_result(0), _make_result(1)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=99, relevance="HIGH", reason=""),  # out of range
            RankedItem(index=-1, relevance="HIGH", reason=""),  # negative
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == 0

    def test_deduplicates_indices(self) -> None:
        results = [_make_result(0), _make_result(1)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=0, relevance="HIGH", reason=""),  # duplicate
            RankedItem(index=1, relevance="MEDIUM", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 2

    def test_unranked_treated_as_low(self) -> None:
        results = [_make_result(0), _make_result(1), _make_result(2)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            # indices 1 and 2 not ranked — should be treated as LOW
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == 0


class TestRerank:
    """Tests for the top-level rerank() function."""

    async def test_skips_when_results_lte_top_k(self) -> None:
        results = [_make_result(0), _make_result(1)]
        filtered = await rerank("query", results, top_k=5)
        assert filtered == results  # unchanged, no LLM call

    async def test_empty_results_returns_empty(self) -> None:
        filtered = await rerank("query", [], top_k=5)
        assert filtered == []

    @patch("code.shukketsu.rag.reranker._rerank_impl")
    async def test_falls_back_on_circuit_open(self, mock_impl: AsyncMock) -> None:
        from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker
        from code.shukketsu.resilience.errors import CircuitOpenError

        # Force breaker open
        qwen_reranker_breaker._state = qwen_reranker_breaker._state.__class__("open")
        qwen_reranker_breaker._failure_count = 100
        import time

        qwen_reranker_breaker._last_failure_time = time.monotonic()

        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)
        assert len(filtered) == 3
        assert filtered == results[:3]
        mock_impl.assert_not_called()

    @patch("code.shukketsu.rag.reranker._rerank_impl")
    async def test_falls_back_on_llm_error(self, mock_impl: AsyncMock) -> None:
        from code.shukketsu.resilience.errors import LLMUnavailableError

        mock_impl.side_effect = LLMUnavailableError("Qwen down")
        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)
        # After enough failures, breaker opens — but first call should still fallback
        assert len(filtered) == 3
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_reranker.py -v`
Expected: FAIL — module `code.shukketsu.rag.reranker` does not exist.

**Step 3: Implement**

Create `code/shukketsu/rag/reranker.py`:

```python
"""Qwen 4B categorical relevance reranker for search results.

Scores search results as HIGH/MEDIUM/LOW and filters to top_k.
Circuit breaker wraps the LLM call — falls back to unreranked on failure.
"""

import logging
from typing import Literal

from pydantic import BaseModel

from code.shukketsu.llm.structured import ModelBackend, get_structured_output
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker
from code.shukketsu.resilience.errors import CircuitOpenError

logger = logging.getLogger(__name__)

_CONTENT_TRUNCATE = 200

_SYSTEM_PROMPT = (
    "You are a relevance scorer for WoW TBC Rogue search results.\n"
    "Given a query and numbered search results, rate each result's "
    "relevance as HIGH, MEDIUM, or LOW.\n\n"
    "HIGH: Directly answers or is essential to answering the query.\n"
    "MEDIUM: Related and potentially useful but not a direct answer.\n"
    "LOW: Irrelevant or only tangentially related.\n\n"
    "Be strict — only mark HIGH if the result clearly helps answer the query."
)


class RankedItem(BaseModel):
    """A single result's relevance assessment."""

    index: int
    relevance: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str


class RerankerResponse(BaseModel):
    """Batch relevance assessment from Qwen 4B."""

    rankings: list[RankedItem]


def _build_rerank_messages(
    query: str, results: list[SearchResult]
) -> list[dict[str, str]]:
    """Build the prompt messages for the reranker."""
    result_lines = []
    for i, r in enumerate(results):
        content = r.content[:_CONTENT_TRUNCATE]
        result_lines.append(f"[{i}] {r.source_title} — {content}")

    user_content = f"Query: {query}\n\nResults:\n" + "\n".join(result_lines)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _apply_rankings(
    results: list[SearchResult],
    rankings: list[RankedItem],
    top_k: int,
) -> list[SearchResult]:
    """Apply categorical rankings with index validation.

    - Ignores out-of-range indices
    - Deduplicates indices (keeps first occurrence)
    - Unranked results treated as LOW (dropped)
    - Within each category, preserves original list order (by rrf_score)
    """
    n = len(results)
    seen: set[int] = set()
    buckets: dict[str, set[int]] = {"HIGH": set(), "MEDIUM": set(), "LOW": set()}

    for item in rankings:
        if item.index < 0 or item.index >= n:
            continue
        if item.index in seen:
            continue
        seen.add(item.index)
        buckets[item.relevance].add(item.index)

    # Collect in original order within each bucket
    high = [results[i] for i in range(n) if i in buckets["HIGH"]]
    medium = [results[i] for i in range(n) if i in buckets["MEDIUM"]]

    # All HIGHs, then fill with MEDIUMs up to top_k
    merged = high[:top_k]
    remaining = top_k - len(merged)
    if remaining > 0:
        merged.extend(medium[:remaining])

    return merged


async def _rerank_impl(
    query: str, results: list[SearchResult], top_k: int
) -> list[SearchResult]:
    """Call Qwen 4B for relevance scoring."""
    messages = _build_rerank_messages(query, results)
    response = await get_structured_output(
        RerankerResponse,
        messages,
        backend=ModelBackend.ROUTER,
        max_tokens=2048,
    )

    if not response.rankings:
        return results[:top_k]

    return _apply_rankings(results, response.rankings, top_k)


async def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank search results using Qwen 4B categorical scoring.

    Short-circuits if results <= top_k. Falls back to truncated unreranked
    results if the circuit breaker is open or the LLM call fails.

    Args:
        query: The original search query.
        results: Search results to rerank.
        top_k: Number of results to return.

    Returns:
        Filtered and reranked list of SearchResult.
    """
    if len(results) <= top_k:
        return results

    try:
        return await qwen_reranker_breaker.call(_rerank_impl, query, results, top_k)
    except CircuitOpenError:
        logger.warning("Reranker circuit breaker open — returning unreranked results")
        return results[:top_k]
    except Exception:
        logger.warning("Reranker failed — returning unreranked results", exc_info=True)
        return results[:top_k]
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_reranker.py -v`
Expected: 16 PASSED

Run: `python3 -m pytest tests/unit/ -v`
Expected: 431 passed (415 + 16 new).

**Step 5: Commit**

```bash
git add code/shukketsu/rag/reranker.py tests/unit/test_reranker.py
git commit -m "feat(rag): add Qwen 4B categorical reranker with circuit breaker"
```

---

## Task 5: Wire Reranker into RagSearchTool

Modify the existing `rag_search` tool to over-retrieve and rerank.

**Files:**
- Modify: `code/shukketsu/tools/knowledge/search.py:38-63`
- Test: `tests/unit/test_search_reranking.py` (new file)

**Step 1: Write the failing tests**

Create `tests/unit/test_search_reranking.py`:

```python
"""Tests for reranker integration with RagSearchTool."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu import config
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.tools.knowledge.search import RagSearchTool


def _make_result(chunk_id: int, content: str = "test") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        source_title=f"Source {chunk_id}",
        source_url=f"https://example.com/{chunk_id}",
        trust_score=0.7,
        rrf_score=0.5,
    )


async def _mock_embed(text: str) -> list[float]:
    return [0.1] * 768


class TestRagSearchReranking:
    """Tests for reranker integration in RagSearchTool."""

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_over_retrieves_with_multiplier(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap", "top_k": 5})

        # hybrid_search should be called with top_k * multiplier
        call_kwargs = mock_hybrid.call_args
        assert call_kwargs.kwargs["top_k"] == 5 * config.RERANKER_FETCH_MULTIPLIER

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_passes_results_through_reranker(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap"})

        mock_rerank.assert_called_once()
        args = mock_rerank.call_args
        assert args[0][0] == "hit cap"  # query
        assert len(args[0][1]) == 15  # all raw results passed in

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_fetch_k_scales_with_multiplier(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        mock_hybrid.return_value = []
        mock_rerank.return_value = []

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap", "top_k": 10})

        call_kwargs = mock_hybrid.call_args
        expected_fetch = 10 * config.RERANKER_FETCH_MULTIPLIER
        assert call_kwargs.kwargs["fetch_k"] >= expected_fetch

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_top_k_respected_after_reranking(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        # Reranker returns exactly top_k
        mock_rerank.return_value = raw_results[:3]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap", "top_k": 3})

        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "[4]" not in result

    async def test_fts_fallback_skips_reranking(self, test_db) -> None:
        """When embedding fails, FTS fallback should not call reranker."""
        from code.shukketsu.resilience.errors import EmbeddingError

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("Ollama embed down")

        with patch("code.shukketsu.tools.knowledge.search.rerank") as mock_rerank:
            tool = RagSearchTool(conn=test_db, embed_fn=failing_embed)
            await tool.execute({"query": "hit cap"})
            mock_rerank.assert_not_called()

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_reranker_failure_returns_unreranked(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        """If reranker itself fails, we still get results (unreranked fallback)."""
        raw_results = [_make_result(i, f"Content {i}") for i in range(6)]
        mock_hybrid.return_value = raw_results
        # rerank already has internal fallback — simulate it returning truncated
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "Content" in result
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_search_reranking.py -v`
Expected: FAIL — `rerank` not imported in `search.py`, hybrid_search not called with multiplied top_k.

**Step 3: Implement**

Modify `code/shukketsu/tools/knowledge/search.py`. Replace the `execute` method (lines 38-63):

```python
    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute a hybrid search (vector + FTS5 + RRF) with reranking."""
        query = tool_input.get("query", "")
        top_k = tool_input.get("top_k", config.RAG_SEARCH_TOP_K)

        if not query:
            return "Error: 'query' parameter is required."

        try:
            embedding = await ollama_embed_breaker.call(self._embed_fn, query)
        except Exception as exc:
            logger.warning("Embedding failed, falling back to keyword-only search: %s", exc)
            return self._fts_only_search(query, top_k)

        # Over-retrieve for reranking
        rerank_fetch = top_k * config.RERANKER_FETCH_MULTIPLIER
        effective_fetch_k = max(config.RAG_SEARCH_FETCH_K, rerank_fetch)
        raw_results = await hybrid_search(
            self._conn, query, embedding, top_k=rerank_fetch, fetch_k=effective_fetch_k
        )

        # Rerank (falls back to truncated on failure internally)
        results = await rerank(query, raw_results, top_k)

        if not results:
            return "No relevant documents found for this query."

        parts = [f"Found {len(results)} result{'s' if len(results) != 1 else ''}:\n"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] Source: {r.source_title} ({r.source_url})\nTrust: {r.trust_score}\nContent: {r.content}\n"
            )

        return "\n".join(parts)
```

Add the import at the top of the file (after the existing imports):

```python
from code.shukketsu.rag.reranker import rerank
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_search_reranking.py -v`
Expected: 6 PASSED

Run: `python3 -m pytest tests/unit/test_rag_search.py -v`
Expected: All 8 existing rag_search tests still pass (no regressions). The mock embed tests bypass the reranker path because the embedding fails first.

Run: `python3 -m pytest tests/unit/ -v`
Expected: 437 passed (431 + 6 new).

**Step 5: Commit**

```bash
git add code/shukketsu/tools/knowledge/search.py tests/unit/test_search_reranking.py
git commit -m "feat(tools): wire Qwen 4B reranker into rag_search tool"
```

---

## Task 6: Full Regression + Formatting

Run the complete test suite, fix any regressions, apply ruff formatting.

**Files:** No new files — cleanup only.

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: ~437 passed, 0 failed.

**Step 2: Run linting**

Run: `ruff check code/ tests/`
Run: `ruff format code/ tests/`

Fix any issues flagged by ruff (import order, unused imports, line length).

**Step 3: Run tests again after formatting**

Run: `python3 -m pytest tests/unit/ -v`
Expected: Same count, all passing.

**Step 4: Final commit**

```bash
git add -A
git commit -m "style: apply ruff formatting to Phase 2 Step 3 files"
```

---

## Task 7: Update Documentation

Update CLAUDE.md and memory to reflect Step 3 completion.

**Files:**
- Modify: `CLAUDE.md` (update step status)

**Step 1: Update CLAUDE.md**

In the Phase 2 step list, change step 3 from:
```
3. Graph Traversal Tool + Qwen 4B Reranking (graph_search tool, reranker, unified search) — **NEXT**
```
to:
```
3. ~~Graph Traversal Tool + Qwen 4B Reranking~~ — COMPLETE (graph_search tool, reranker, rag_search reranking)
4. Researcher Agent (specialized prompts, ResearchResult, multi-strategy retrieval) — **NEXT**
```

Also update the "Key files with real code" list to include `tools/knowledge/graph_search.py` and `rag/reranker.py`.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 3 completion"
```

---

## Summary

| Task | Tests Added | Running Total | What It Builds |
|------|------------|---------------|----------------|
| 1 | 8 | 389 | Config, errors, circuit breaker |
| 2 | 8 | 397 | GraphStore extensions |
| 3 | 18 | 415 | GraphSearchTool (core) |
| 4 | 16 | 431 | Reranker module |
| 5 | 6 | 437 | Wire reranker into rag_search |
| 6 | 0 | 437 | Regression + formatting |
| 7 | 0 | 437 | Documentation |

**Total: ~56 new tests** (381 → ~437). Higher than the design doc estimate of ~35 because we included GraphStore extension tests and more thorough schema/message tests.
