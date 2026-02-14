"""Graph traversal tool for agent knowledge graph queries."""

import logging
import sqlite3
from typing import Any

from code.shukketsu import config
from code.shukketsu.rag.entities import RelationType
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
            "type": "string",
            "description": "Entity name to search for (e.g., 'Dragonspine Trophy', 'Gruul', 'DST')",
        },
        "relation_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by relationship type (e.g., ['drops_from', 'has_stat'])",
            "optional": True,
        },
        "target_type": {
            "type": "string",
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
        outgoing = self._graph.get_relationships(entity_id, relation_types=relation_types, direction="outgoing")
        incoming = self._graph.get_relationships(entity_id, relation_types=relation_types, direction="incoming")

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
        lines.append(f"Entity: {entity['name']} ({entity['entity_type_name']}, confidence: {entity['confidence']:.2f})")
        lines.append("")

        # Outgoing
        if outgoing:
            lines.append(f"Outgoing relationships ({len(outgoing)}):")
            for rel in outgoing:
                lines.append(
                    f"- {rel['relation_type']} \u2192 {rel['related_name']} "
                    f"({rel['related_type']}, confidence: {rel['confidence']:.2f})"
                )
            lines.append("")

        # Incoming
        if incoming:
            lines.append(f"Incoming relationships ({len(incoming)}):")
            for rel in incoming:
                lines.append(
                    f"- {rel['relation_type']} \u2190 {rel['related_name']} "
                    f"({rel['related_type']}, confidence: {rel['confidence']:.2f})"
                )
            lines.append("")

        # Chunks
        if chunk_ids:
            lines.append(f"Related chunks: {', '.join(str(c) for c in chunk_ids)}")
        else:
            lines.append("Related chunks: none")

        return "\n".join(lines)
