"""Knowledge graph storage layer for entity and relationship CRUD.

Handles insertion, deduplication, and querying of the knowledge graph
stored in SQLite. Entity names are normalized via canonical name
resolution before storage.
"""

import json
import logging
import sqlite3

from code.shukketsu.rag.entities import EntityType, RelationType, resolve_canonical

logger = logging.getLogger(__name__)


class GraphStore:
    """SQLite-backed knowledge graph storage.

    Manages entity_types, entities, and relationships tables.
    Entity names are normalized to canonical form for deduplication.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def seed_entity_types(self) -> None:
        """Populate the entity_types table from the EntityType enum.

        Idempotent — uses INSERT OR IGNORE.
        """
        for et in EntityType:
            display = et.value.replace("_", " ").title()
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_types (name, display_name) VALUES (?, ?)",
                (et.value, display),
            )
        self._conn.commit()

    def get_entity_type_id(self, entity_type: EntityType) -> int:
        """Look up the DB ID for an entity type.

        Raises:
            ValueError: If the entity type has not been seeded.
        """
        row = self._conn.execute("SELECT id FROM entity_types WHERE name = ?", (entity_type.value,)).fetchone()
        if row is None:
            raise ValueError(f"Entity type '{entity_type.value}' not seeded. Call seed_entity_types() first.")
        return int(row["id"])

    def upsert_entity(
        self,
        name: str,
        entity_type: EntityType,
        *,
        properties: dict | None = None,
        source_chunk_id: int | None = None,
        confidence: float = 0.5,
    ) -> int:
        """Insert or update an entity, deduplicating by canonical name.

        If an entity with the same canonical_name + entity_type exists,
        updates it (keeps higher confidence). Otherwise inserts a new row.

        Returns:
            The entity ID (existing or newly created).
        """
        canonical = resolve_canonical(name)
        type_id = self.get_entity_type_id(entity_type)
        props_json = json.dumps(properties) if properties else None

        cursor = self._conn.execute(
            "INSERT INTO entities "
            "(name, entity_type_id, canonical_name, properties_json, source_chunk_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(canonical_name, entity_type_id) DO UPDATE SET "
            "properties_json = COALESCE(excluded.properties_json, entities.properties_json), "
            "confidence = MAX(entities.confidence, excluded.confidence) "
            "RETURNING id",
            (name, type_id, canonical, props_json, source_chunk_id, confidence),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("INSERT/UPDATE entities failed to return id")
        return int(row[0])

    def upsert_relationship(
        self,
        source_entity_id: int,
        target_entity_id: int,
        relation_type: RelationType,
        *,
        properties: dict | None = None,
        source_chunk_id: int | None = None,
        confidence: float = 0.5,
    ) -> int:
        """Insert or update a relationship, deduplicating by (src, tgt, type).

        Returns:
            The relationship ID (existing or newly created).
        """
        props_json = json.dumps(properties) if properties else None

        cursor = self._conn.execute(
            "INSERT INTO relationships "
            "(source_entity_id, target_entity_id, relation_type, "
            "properties_json, source_chunk_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_entity_id, target_entity_id, relation_type) DO UPDATE SET "
            "properties_json = COALESCE(excluded.properties_json, relationships.properties_json), "
            "confidence = MAX(relationships.confidence, excluded.confidence) "
            "RETURNING id",
            (source_entity_id, target_entity_id, relation_type.value, props_json, source_chunk_id, confidence),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("INSERT/UPDATE relationships failed to return id")
        return int(row[0])

    def get_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> sqlite3.Row | None:
        """Look up an entity by name (resolves aliases).

        Returns:
            A Row dict with entity columns, or None if not found.
        """
        canonical = resolve_canonical(name)

        if entity_type is not None:
            type_id = self.get_entity_type_id(entity_type)
            row: sqlite3.Row | None = self._conn.execute(
                "SELECT * FROM entities WHERE canonical_name = ? AND entity_type_id = ?",
                (canonical, type_id),
            ).fetchone()
            return row

        result: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM entities WHERE canonical_name = ?", (canonical,)
        ).fetchone()
        return result

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

    def get_relationships(
        self,
        entity_id: int,
        *,
        relation_types: list[RelationType] | None = None,
        direction: str = "outgoing",
    ) -> list[sqlite3.Row]:
        """Query relationships for an entity.

        Args:
            entity_id: The entity to query relationships for.
            relation_types: Optional filter by relationship type(s).
            direction: "outgoing" (entity is source) or "incoming" (entity is target).

        Returns:
            List of relationship rows with joined entity names.
        """
        if direction == "outgoing":
            col = "source_entity_id"
            join_col = "target_entity_id"
        elif direction == "incoming":
            col = "target_entity_id"
            join_col = "source_entity_id"
        else:
            raise ValueError(f"Invalid direction '{direction}': must be 'outgoing' or 'incoming'")

        query = (
            f"SELECT r.*, e.name AS related_name, e.canonical_name AS related_canonical, "
            f"et.name AS related_type "
            f"FROM relationships r "
            f"JOIN entities e ON e.id = r.{join_col} "
            f"JOIN entity_types et ON et.id = e.entity_type_id "
            f"WHERE r.{col} = ?"
        )
        params: list = [entity_id]

        if relation_types:
            placeholders = ", ".join("?" for _ in relation_types)
            query += f" AND r.relation_type IN ({placeholders})"
            params.extend(rt.value for rt in relation_types)

        return self._conn.execute(query, params).fetchall()
