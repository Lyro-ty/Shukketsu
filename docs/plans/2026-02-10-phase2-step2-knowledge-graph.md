# Phase 2 Step 2: Knowledge Graph Schema + Entity Extraction

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a knowledge graph to SQLite (entities + relationships tables), define WoW TBC entity types, extract entities from ingested text using structured LLM output, and store them alongside the existing chunk/embedding pipeline.

**Architecture:** Three new DB tables (`entity_types`, `entities`, `relationships`) store a typed knowledge graph extracted during ingest. Entity extraction uses Llama 70B via the existing `get_structured_output()` structured client. A `GraphStore` class handles all graph CRUD. The `IngestPipeline` is extended with an optional entity extraction step after chunking. Canonical name normalization + alias resolution handles WoW abbreviations (DST, SnD, etc.).

**Tech Stack:** SQLite, Pydantic v2, instructor (existing structured output client), pytest

**Starting point:** 302 tests passing (Phase 2 Step 1 complete).

---

## Task 1: Graph Database Schema

**Files:**
- Modify: `code/shukketsu/db/schema.sql` — add graph tables
- Modify: `code/shukketsu/db/connection.py` — add v1→v2 migration
- Test: `tests/unit/test_db.py` — add graph table tests

### Step 1: Write the failing tests

Add to `tests/unit/test_db.py`, inside `TestInitDb`:

```python
def test_creates_graph_tables(self, db: sqlite3.Connection) -> None:
    """init_db should create entity_types, entities, relationships tables."""
    tables = _get_tables(db)
    for name in ("entity_types", "entities", "relationships"):
        assert name in tables, f"Missing table: {name}"

def test_schema_version_is_2(self, db: sqlite3.Connection) -> None:
    """Schema version should be 2 after fresh initialization."""
    version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert version == 2

def test_entity_type_unique_name(self, db: sqlite3.Connection) -> None:
    """entity_types.name should enforce uniqueness."""
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item Duplicate')")

def test_entity_unique_canonical_per_type(self, db: sqlite3.Connection) -> None:
    """entities should enforce UNIQUE(canonical_name, entity_type_id)."""
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
    type_id = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
    db.execute(
        "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
        ("Dragonspine Trophy", type_id, "dragonspine trophy"),
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("DST", type_id, "dragonspine trophy"),
        )

def test_relationship_unique_triple(self, db: sqlite3.Connection) -> None:
    """relationships should enforce UNIQUE(source_entity_id, target_entity_id, relation_type)."""
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('boss', 'Boss')")
    type_item = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
    type_boss = db.execute("SELECT id FROM entity_types WHERE name = 'boss'").fetchone()["id"]
    db.execute(
        "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
        ("DST", type_item, "dragonspine trophy"),
    )
    db.execute(
        "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
        ("Gruul", type_boss, "gruul"),
    )
    db.commit()
    item_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'dragonspine trophy'").fetchone()["id"]
    boss_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'gruul'").fetchone()["id"]
    db.execute(
        "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
        (item_id, boss_id, "drops_from"),
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
            (item_id, boss_id, "drops_from"),
        )

def test_entity_fk_to_entity_types(self, db: sqlite3.Connection) -> None:
    """entities.entity_type_id should reference entity_types(id)."""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("Ghost Item", 999, "ghost item"),
        )

def test_relationship_fk_cascade_on_entity_delete(self, db: sqlite3.Connection) -> None:
    """Deleting an entity should cascade-delete its relationships."""
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
    db.execute("INSERT INTO entity_types (name, display_name) VALUES ('boss', 'Boss')")
    type_item = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
    type_boss = db.execute("SELECT id FROM entity_types WHERE name = 'boss'").fetchone()["id"]
    db.execute(
        "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
        ("DST", type_item, "dragonspine trophy"),
    )
    db.execute(
        "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
        ("Gruul", type_boss, "gruul"),
    )
    db.commit()
    item_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'dragonspine trophy'").fetchone()["id"]
    boss_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'gruul'").fetchone()["id"]
    db.execute(
        "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
        (item_id, boss_id, "drops_from"),
    )
    db.commit()
    db.execute("DELETE FROM entities WHERE id = ?", (item_id,))
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    assert count == 0
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_db.py -v -k "graph or version_is_2 or entity_type_unique or entity_unique_canonical or relationship_unique or entity_fk or relationship_fk"`
Expected: FAIL (tables don't exist yet)

### Step 3: Update schema.sql

Add graph tables to `code/shukketsu/db/schema.sql`. Insert **after** the articles table, **before** the closing of the file:

```sql
-- ============================================================
-- Knowledge Graph (Phase 2)
-- ============================================================

-- Entity type catalog (seeded from code on first use)
CREATE TABLE entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

-- Named entities extracted from source content
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,                                -- original surface form
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id),
    canonical_name TEXT NOT NULL,                      -- normalized for dedup
    properties_json TEXT,                              -- {"ilvl": 141, ...}
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(canonical_name, entity_type_id)
);

CREATE INDEX idx_entities_type ON entities(entity_type_id);
CREATE INDEX idx_entities_canonical ON entities(canonical_name);

-- Typed edges between entities
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,                       -- from RelationType enum
    properties_json TEXT,                              -- {"phase": 1, ...}
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relation_type);
```

Also update the schema_version insert from `VALUES (1)` to `VALUES (2)`.

### Step 4: Update connection.py for migration

Add a `_migrate_v1_to_v2()` function to `code/shukketsu/db/connection.py` and update `init_db()`:

```python
_GRAPH_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id),
    canonical_name TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(canonical_name, entity_type_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema from schema.sql. Idempotent."""
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            if version < 2:
                _migrate_v1_to_v2(conn)
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet -- need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    logger.info("Database schema initialized (version 2)")


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate v1 schema to v2: add knowledge graph tables."""
    conn.executescript(_GRAPH_TABLES_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.commit()
    logger.info("Database migrated from v1 to v2 (knowledge graph tables)")
```

### Step 5: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: ALL PASS (old + new tests)

Also update the existing `test_sets_schema_version` test — it currently asserts version == 1. Change assertion to `assert version == 2`.

### Step 6: Run full suite to verify no regressions

Run: `python3 -m pytest tests/unit/ -v`
Expected: 302+ tests PASS (all existing tests still pass)

### Step 7: Commit

```bash
git add code/shukketsu/db/schema.sql code/shukketsu/db/connection.py tests/unit/test_db.py
git commit -m "feat(db): add knowledge graph tables (entity_types, entities, relationships)"
```

---

## Task 2: Entity Type + Relationship Type Enums and Extraction Models

**Files:**
- Create: `code/shukketsu/rag/entities.py`
- Test: `tests/unit/test_entities.py`

### Step 1: Write the failing tests

Create `tests/unit/test_entities.py`:

```python
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
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_entities.py -v`
Expected: FAIL (module doesn't exist)

### Step 3: Write the implementation

Create `code/shukketsu/rag/entities.py`:

```python
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
```

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_entities.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add code/shukketsu/rag/entities.py tests/unit/test_entities.py
git commit -m "feat(rag): add WoW TBC entity/relationship types and extraction models"
```

---

## Task 3: Canonical Name Normalization + Alias Resolution

**Files:**
- Modify: `code/shukketsu/rag/entities.py` — add normalization functions
- Test: `tests/unit/test_entities.py` — add normalization tests

### Step 1: Write the failing tests

Add to `tests/unit/test_entities.py`:

```python
from code.shukketsu.rag.entities import (
    KNOWN_ALIASES,
    normalize_name,
    resolve_canonical,
)


class TestNormalizeName:
    """Tests for entity name normalization."""

    def test_lowercase(self) -> None:
        assert normalize_name("Dragonspine Trophy") == "dragonspine trophy"

    def test_strips_whitespace(self) -> None:
        assert normalize_name("  Dragonspine Trophy  ") == "dragonspine trophy"

    def test_strips_leading_articles(self) -> None:
        assert normalize_name("The Black Temple") == "black temple"
        assert normalize_name("A Random Item") == "random item"
        assert normalize_name("An Enchant") == "enchant"

    def test_collapses_whitespace(self) -> None:
        assert normalize_name("Gruul   the   Dragonkiller") == "gruul the dragonkiller"

    def test_empty_string(self) -> None:
        assert normalize_name("") == ""

    def test_already_normalized(self) -> None:
        assert normalize_name("sinister strike") == "sinister strike"

    def test_preserves_apostrophes(self) -> None:
        assert normalize_name("Gruul's Lair") == "gruul's lair"

    def test_preserves_hyphens(self) -> None:
        assert normalize_name("Best-in-Slot") == "best-in-slot"


class TestResolveCanonical:
    """Tests for canonical name resolution with alias support."""

    def test_known_alias(self) -> None:
        assert resolve_canonical("DST") == "dragonspine trophy"

    def test_known_alias_case_insensitive(self) -> None:
        assert resolve_canonical("dst") == "dragonspine trophy"

    def test_unknown_name_normalized(self) -> None:
        assert resolve_canonical("Some New Item") == "some new item"

    def test_already_canonical(self) -> None:
        assert resolve_canonical("dragonspine trophy") == "dragonspine trophy"

    def test_alias_with_article(self) -> None:
        """Alias lookup should work after article stripping."""
        assert resolve_canonical("the BT") == "black temple"

    def test_kara_alias(self) -> None:
        assert resolve_canonical("Kara") == "karazhan"

    def test_snd_alias(self) -> None:
        assert resolve_canonical("SnD") == "slice and dice"


class TestKnownAliases:
    """Tests for the alias table."""

    def test_aliases_are_normalized(self) -> None:
        """All alias keys should be in normalized form (lowercase, no articles)."""
        for key in KNOWN_ALIASES:
            assert key == key.lower(), f"Alias key '{key}' is not lowercase"
            assert not key.startswith(("the ", "a ", "an ")), f"Alias key '{key}' has leading article"

    def test_alias_values_are_normalized(self) -> None:
        """All alias values should be in normalized form."""
        for value in KNOWN_ALIASES.values():
            assert value == normalize_name(value), f"Alias value '{value}' is not normalized"
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_entities.py -v -k "Normalize or Canonical or Aliases"`
Expected: FAIL (functions not defined)

### Step 3: Write the implementation

Add to `code/shukketsu/rag/entities.py` (after the models, before EOF):

```python
import re


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
```

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_entities.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add code/shukketsu/rag/entities.py tests/unit/test_entities.py
git commit -m "feat(rag): add canonical name normalization and alias resolution"
```

---

## Task 4: Graph Storage Layer

**Files:**
- Create: `code/shukketsu/rag/graph.py`
- Test: `tests/unit/test_graph.py`

### Step 1: Write the failing tests

Create `tests/unit/test_graph.py`:

```python
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
        row = test_db.execute(
            "SELECT display_name FROM entity_types WHERE name = 'talent_tree'"
        ).fetchone()
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
        row = graph._conn.execute(
            "SELECT name, canonical_name FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
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
            "Dragonspine Trophy", EntityType.ITEM,
            properties={"ilvl": 141},
        )
        row = graph._conn.execute(
            "SELECT properties_json FROM entities WHERE canonical_name = 'dragonspine trophy'"
        ).fetchone()
        assert '"ilvl": 141' in row["properties_json"]

    def test_stores_source_chunk_id(self, graph: GraphStore) -> None:
        # Insert a source + chunk first
        graph._conn.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        graph._conn.execute(
            "INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'test', 0)"
        )
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
            item_id, boss_id, RelationType.DROPS_FROM,
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
            item_id, boss_id, RelationType.DROPS_FROM,
            properties={"drop_rate": 0.15},
        )
        row = graph._conn.execute(
            "SELECT properties_json FROM relationships WHERE id = 1"
        ).fetchone()
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
            item_id, relation_types=[RelationType.DROPS_FROM],
        )
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "drops_from"

    def test_empty_graph_returns_empty(self, graph: GraphStore) -> None:
        item_id = graph.upsert_entity("Lonely Item", EntityType.ITEM)
        rels = graph.get_relationships(item_id)
        assert rels == []
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_graph.py -v`
Expected: FAIL (module doesn't exist)

### Step 3: Write the implementation

Create `code/shukketsu/rag/graph.py`:

```python
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
        row = self._conn.execute(
            "SELECT id FROM entity_types WHERE name = ?", (entity_type.value,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Entity type '{entity_type.value}' not seeded. "
                "Call seed_entity_types() first."
            )
        return row["id"]

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

        existing = self._conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ? AND entity_type_id = ?",
            (canonical, type_id),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE entities SET "
                "properties_json = COALESCE(?, properties_json), "
                "confidence = MAX(confidence, ?) "
                "WHERE id = ?",
                (props_json, confidence, existing["id"]),
            )
            return existing["id"]

        cursor = self._conn.execute(
            "INSERT INTO entities "
            "(name, entity_type_id, canonical_name, properties_json, source_chunk_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, type_id, canonical, props_json, source_chunk_id, confidence),
        )
        return cursor.lastrowid  # type: ignore[return-value]

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

        existing = self._conn.execute(
            "SELECT id FROM relationships "
            "WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?",
            (source_entity_id, target_entity_id, relation_type.value),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE relationships SET "
                "properties_json = COALESCE(?, properties_json), "
                "confidence = MAX(confidence, ?) "
                "WHERE id = ?",
                (props_json, confidence, existing["id"]),
            )
            return existing["id"]

        cursor = self._conn.execute(
            "INSERT INTO relationships "
            "(source_entity_id, target_entity_id, relation_type, "
            "properties_json, source_chunk_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_entity_id, target_entity_id, relation_type.value,
             props_json, source_chunk_id, confidence),
        )
        return cursor.lastrowid  # type: ignore[return-value]

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
            return self._conn.execute(
                "SELECT * FROM entities WHERE canonical_name = ? AND entity_type_id = ?",
                (canonical, type_id),
            ).fetchone()

        return self._conn.execute(
            "SELECT * FROM entities WHERE canonical_name = ?", (canonical,)
        ).fetchone()

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
        else:
            col = "target_entity_id"
            join_col = "source_entity_id"

        query = (
            f"SELECT r.*, e.name AS related_name, e.canonical_name AS related_canonical "
            f"FROM relationships r "
            f"JOIN entities e ON e.id = r.{join_col} "
            f"WHERE r.{col} = ?"
        )
        params: list = [entity_id]

        if relation_types:
            placeholders = ", ".join("?" for _ in relation_types)
            query += f" AND r.relation_type IN ({placeholders})"
            params.extend(rt.value for rt in relation_types)

        return self._conn.execute(query, params).fetchall()
```

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_graph.py -v`
Expected: ALL PASS

### Step 5: Run full suite

Run: `python3 -m pytest tests/unit/ -v`
Expected: ALL PASS (no regressions)

### Step 6: Commit

```bash
git add code/shukketsu/rag/graph.py tests/unit/test_graph.py
git commit -m "feat(rag): add GraphStore for entity/relationship CRUD with dedup"
```

---

## Task 5: Entity Extraction via Structured LLM Output

**Files:**
- Modify: `code/shukketsu/rag/entities.py` — add extraction prompt + function
- Modify: `code/shukketsu/resilience/errors.py` — add EntityExtractionError
- Test: `tests/unit/test_entity_extraction.py`

### Step 1: Write the failing tests

Create `tests/unit/test_entity_extraction.py`:

```python
"""Tests for LLM-powered entity extraction."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.rag.entities import (
    EXTRACTION_SYSTEM_PROMPT,
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
    build_extraction_messages,
    extract_entities_from_chunk,
)
from code.shukketsu.resilience.errors import EntityExtractionError


class TestExtractionPrompt:
    """Tests for the extraction prompt and message builder."""

    def test_system_prompt_exists(self) -> None:
        assert len(EXTRACTION_SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_entity_types(self) -> None:
        for et in ["item", "spell", "talent", "boss", "instance", "stat"]:
            assert et in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_relation_types(self) -> None:
        for rt in ["drops_from", "has_stat", "available_in"]:
            assert rt in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_build_messages_structure(self) -> None:
        messages = build_extraction_messages("Some WoW text about rogues.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Some WoW text about rogues." in messages[1]["content"]

    def test_build_messages_includes_system_prompt(self) -> None:
        messages = build_extraction_messages("text")
        assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT


class TestExtractEntities:
    """Tests for extract_entities_from_chunk()."""

    async def test_returns_chunk_extraction(self) -> None:
        mock_result = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Dragonspine Trophy", entity_type=EntityType.ITEM),
                ExtractedEntity(name="Gruul", entity_type=EntityType.BOSS),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Dragonspine Trophy",
                    target="Gruul",
                    relation_type=RelationType.DROPS_FROM,
                ),
            ],
        )

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await extract_entities_from_chunk("DST drops from Gruul.")

        assert isinstance(result, ChunkExtraction)
        assert len(result.entities) == 2
        assert len(result.relationships) == 1

    async def test_empty_extraction(self) -> None:
        mock_result = ChunkExtraction(entities=[], relationships=[])

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await extract_entities_from_chunk("No entities here.")

        assert result.entities == []
        assert result.relationships == []

    async def test_wraps_llm_errors(self) -> None:
        """LLM failures should be wrapped in EntityExtractionError."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            side_effect=StructuredOutputError("Failed after retries"),
        ):
            with pytest.raises(EntityExtractionError, match="Failed"):
                await extract_entities_from_chunk("text")

    async def test_passes_correct_model(self) -> None:
        mock_result = ChunkExtraction(entities=[], relationships=[])

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await extract_entities_from_chunk("text")

        call_kwargs = mock_fn.call_args
        assert call_kwargs[0][0] is ChunkExtraction  # response_model
        assert call_kwargs[1]["backend"].value == "reasoning"
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_entity_extraction.py -v`
Expected: FAIL (imports don't exist)

### Step 3: Add EntityExtractionError

Add to `code/shukketsu/resilience/errors.py`:

```python
class EntityExtractionError(ShukketsuError):
    """Raised when entity extraction from a chunk fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.LLM_MALFORMED_OUTPUT)
```

### Step 4: Add extraction prompt and function to entities.py

Add to `code/shukketsu/rag/entities.py` (add imports at top, functions at bottom):

```python
# Additional imports at top:
import logging

from code.shukketsu.llm.structured import ModelBackend, get_structured_output
from code.shukketsu.resilience.errors import EntityExtractionError, LLMUnavailableError, StructuredOutputError

logger = logging.getLogger(__name__)


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
        return await get_structured_output(
            ChunkExtraction,
            messages,
            backend=ModelBackend.REASONING,
            temperature=0.1,
            max_tokens=2048,
        )
    except (StructuredOutputError, LLMUnavailableError) as exc:
        raise EntityExtractionError(f"Failed to extract entities: {exc}") from exc
```

### Step 5: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_entity_extraction.py -v`
Expected: ALL PASS

### Step 6: Commit

```bash
git add code/shukketsu/rag/entities.py code/shukketsu/resilience/errors.py tests/unit/test_entity_extraction.py
git commit -m "feat(rag): add LLM-powered entity extraction with structured output"
```

---

## Task 6: Pipeline Integration

**Files:**
- Modify: `code/shukketsu/ingest/pipeline.py` — add optional entity extraction
- Test: `tests/unit/test_pipeline.py` — add extraction integration tests

### Step 1: Write the failing tests

Add to `tests/unit/test_pipeline.py`:

```python
from code.shukketsu.rag.entities import (
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
)
from code.shukketsu.rag.graph import GraphStore


def _mock_extractor(extraction: ChunkExtraction | None = None) -> AsyncMock:
    """Create a mock entity extractor."""
    if extraction is None:
        extraction = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Sinister Strike", entity_type=EntityType.SPELL),
                ExtractedEntity(name="Hit Rating", entity_type=EntityType.STAT),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Sinister Strike",
                    target="Hit Rating",
                    relation_type=RelationType.AFFECTED_BY,
                ),
            ],
        )

    async def fake_extract(chunk_text: str) -> ChunkExtraction:
        return extraction

    return AsyncMock(side_effect=fake_extract)


class TestIngestPipelineWithExtraction:
    """Tests for entity extraction during ingest."""

    async def test_extraction_stores_entities(self, test_db: sqlite3.Connection) -> None:
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        await pipeline.ingest(
            text="The hit cap for Sinister Strike is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        entity = graph.get_entity_by_name("Sinister Strike")
        assert entity is not None

    async def test_extraction_stores_relationships(self, test_db: sqlite3.Connection) -> None:
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        await pipeline.ingest(
            text="Sinister Strike is affected by Hit Rating.",
            url="https://example.com/guide",
            title="Guide",
        )
        entity = graph.get_entity_by_name("Sinister Strike")
        rels = graph.get_relationships(entity["id"])
        assert len(rels) >= 1

    async def test_no_extraction_without_graph_store(self, test_db: sqlite3.Connection) -> None:
        """Pipeline without graph_store should not attempt extraction."""
        extractor = _mock_extractor()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            extract_fn=extractor,
        )
        await pipeline.ingest(
            text="Some content.",
            url="https://example.com/guide",
            title="Guide",
        )
        extractor.assert_not_called()

    async def test_extraction_failure_does_not_abort_ingest(
        self, test_db: sqlite3.Connection
    ) -> None:
        """If extraction fails for a chunk, ingest should still succeed."""
        from code.shukketsu.resilience.errors import EntityExtractionError

        graph = GraphStore(test_db)
        graph.seed_entity_types()

        failing_extractor = AsyncMock(
            side_effect=EntityExtractionError("LLM down")
        )
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=failing_extractor,
        )
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.chunk_count >= 1  # ingest succeeded despite extraction failure

    async def test_extraction_called_per_chunk(self, test_db: sqlite3.Connection) -> None:
        """Extractor should be called once per chunk."""
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        extractor = _mock_extractor(ChunkExtraction(entities=[], relationships=[]))
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=extractor,
        )
        text = "The hit cap for combat rogues is 9%. " * 50  # enough to split into multiple chunks
        result = await pipeline.ingest(text=text, url="https://example.com/guide", title="Guide")
        assert extractor.call_count == result.chunk_count

    async def test_ingest_result_includes_entity_count(self, test_db: sqlite3.Connection) -> None:
        """IngestResult should report how many entities were extracted."""
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        result = await pipeline.ingest(
            text="Sinister Strike and Hit Rating.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.entity_count >= 0  # new field

    async def test_existing_pipeline_still_works(self, test_db: sqlite3.Connection) -> None:
        """Pipeline without new params should work exactly as before."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="Basic content.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.chunk_count >= 1
        assert result.already_existed is False
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/unit/test_pipeline.py -v -k "Extraction or existing_pipeline"`
Expected: FAIL (new params don't exist)

### Step 3: Write the implementation

Modify `code/shukketsu/ingest/pipeline.py`:

```python
"""Ingest pipeline: chunk text, embed it, extract entities, store in SQLite."""

import hashlib
import logging
import sqlite3
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from code.shukketsu.ingest.chunker import chunk_text
from code.shukketsu.ingest.embedder import Embedder
from code.shukketsu.rag.entities import ChunkExtraction
from code.shukketsu.rag.graph import GraphStore
from code.shukketsu.resilience.errors import EntityExtractionError

logger = logging.getLogger(__name__)

# Type alias for the extraction function
ExtractFn = Callable[[str], Awaitable[ChunkExtraction]]


@dataclass(frozen=True)
class IngestResult:
    """Result of an ingest operation."""

    source_id: int
    chunk_count: int
    already_existed: bool
    entity_count: int = 0
    relationship_count: int = 0


class IngestPipeline:
    """Orchestrates chunking, embedding, entity extraction, and storage.

    Handles deduplication via content hashing and stores chunks, vectors,
    entities, and source metadata in a single SQLite transaction.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        *,
        graph_store: GraphStore | None = None,
        extract_fn: ExtractFn | None = None,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._graph_store = graph_store
        self._extract_fn = extract_fn

    async def ingest(
        self,
        text: str,
        url: str,
        title: str,
        source_type: str = "guide",
    ) -> IngestResult:
        """Ingest text into the knowledge base.

        Args:
            text: The full text content to ingest.
            url: Source URL (used as unique key for dedup).
            title: Human-readable title for the source.
            source_type: Category of source (guide, forum, wiki, etc.).

        Returns:
            IngestResult with source_id, chunk_count, entity stats, and dedup status.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest() if text.strip() else ""

        # Check for existing source with the same URL
        existing = self._conn.execute("SELECT id, content_hash FROM sources WHERE url = ?", (url,)).fetchone()

        if existing:
            if existing["content_hash"] == content_hash:
                chunk_count = self._conn.execute(
                    "SELECT chunk_count FROM sources WHERE id = ?", (existing["id"],)
                ).fetchone()["chunk_count"]
                return IngestResult(
                    source_id=existing["id"],
                    chunk_count=chunk_count,
                    already_existed=True,
                )

        # Embed BEFORE touching the database (this is the most likely failure point)
        if text.strip():
            chunks = chunk_text(text)
            embeddings = await self._embedder.embed_texts([c.content for c in chunks])
        else:
            chunks = []
            embeddings = []

        # Now write everything in a single transaction
        entity_count = 0
        relationship_count = 0
        try:
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, content_hash = ?, "
                    "fetched_at = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, datetime.now(UTC).isoformat(), source_id),
                )
            else:
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, datetime.now(UTC).isoformat()),
                )
                source_id = cursor.lastrowid

            chunk_ids: list[int] = []
            for chunk, embedding in zip(chunks, embeddings):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (source_id, content, chunk_index, metadata_json) VALUES (?, ?, ?, NULL)",
                    (source_id, chunk.content, chunk.chunk_index),
                )
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)
                embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
                self._conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, embedding_blob),
                )

            self._conn.execute(
                "UPDATE sources SET chunk_count = ? WHERE id = ?",
                (len(chunks), source_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        # Entity extraction (after commit — non-critical, best-effort)
        if self._graph_store and self._extract_fn and chunks:
            e_count, r_count = await self._extract_and_store(chunks, chunk_ids)
            entity_count = e_count
            relationship_count = r_count

        logger.info(
            "Ingested %d chunks, %d entities from %s (%s)",
            len(chunks), entity_count, url, title,
        )

        return IngestResult(
            source_id=source_id,
            chunk_count=len(chunks),
            already_existed=False,
            entity_count=entity_count,
            relationship_count=relationship_count,
        )

    async def _extract_and_store(
        self,
        chunks: list,
        chunk_ids: list[int],
    ) -> tuple[int, int]:
        """Extract entities from chunks and store in graph. Best-effort."""
        total_entities = 0
        total_relationships = 0

        for chunk, chunk_id in zip(chunks, chunk_ids):
            try:
                extraction = await self._extract_fn(chunk.content)
            except (EntityExtractionError, Exception) as exc:
                logger.warning("Entity extraction failed for chunk %d: %s", chunk_id, exc)
                continue

            entity_id_map: dict[str, int] = {}
            for entity in extraction.entities:
                eid = self._graph_store.upsert_entity(
                    entity.name,
                    entity.entity_type,
                    properties=entity.properties or None,
                    source_chunk_id=chunk_id,
                )
                entity_id_map[entity.name] = eid
                total_entities += 1

            for rel in extraction.relationships:
                src_id = entity_id_map.get(rel.source)
                tgt_id = entity_id_map.get(rel.target)
                if src_id is None or tgt_id is None:
                    logger.debug(
                        "Skipping relationship %s->%s: entity not in this chunk's extraction",
                        rel.source, rel.target,
                    )
                    continue
                self._graph_store.upsert_relationship(
                    src_id, tgt_id, rel.relation_type,
                    properties=rel.properties or None,
                    source_chunk_id=chunk_id,
                )
                total_relationships += 1

        self._conn.commit()
        return total_entities, total_relationships

    def _delete_chunks_and_vectors(self, source_id: int) -> None:
        """Delete all chunks and their vectors for a source."""
        self._conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE source_id = ?)",
            (source_id,),
        )
        self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
```

**Key design decisions:**
- `graph_store` and `extract_fn` are optional kwargs — backward compatible
- Entity extraction happens AFTER the chunk/vector commit — non-critical path
- Extraction failures are caught per-chunk (logged, not re-raised)
- `IngestResult` gains `entity_count` and `relationship_count` fields (default 0)
- Relationship insertion skips if source/target entity wasn't in the same chunk's extraction (entities may come from different chunks — the graph dedup handles cross-chunk references via canonical names)

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/unit/test_pipeline.py -v`
Expected: ALL PASS (old + new)

### Step 5: Run full suite

Run: `python3 -m pytest tests/unit/ -v`
Expected: ALL PASS

### Step 6: Commit

```bash
git add code/shukketsu/ingest/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(ingest): integrate entity extraction into ingest pipeline"
```

---

## Task 7: Final Wiring + Verification

**Files:**
- Modify: `code/shukketsu/rag/__init__.py` — export new modules
- Run: Full test suite verification

### Step 1: Update rag/__init__.py exports

This is optional but helps discoverability. Update `code/shukketsu/rag/__init__.py` if needed (currently empty). No test needed.

### Step 2: Run full test suite

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: ALL tests pass. Confirm count is 302 + new tests (target: ~340-350 tests).

### Step 3: Run linter

Run: `ruff check code/shukketsu/rag/ code/shukketsu/db/ code/shukketsu/ingest/ tests/unit/`
Run: `ruff format --check code/shukketsu/rag/ code/shukketsu/db/ code/shukketsu/ingest/ tests/unit/`
Expected: No issues

### Step 4: Final commit (if any formatting fixes needed)

```bash
git add -A
git commit -m "chore: lint fixes for Phase 2 Step 2"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `entity_types`, `entities`, `relationships` tables exist in schema
- [ ] `EntityType` enum has 17 entity types
- [ ] `RelationType` enum has 17 relationship types
- [ ] `normalize_name()` strips articles, lowercases, collapses whitespace
- [ ] `resolve_canonical()` maps "DST" -> "dragonspine trophy"
- [ ] `GraphStore.upsert_entity()` deduplicates by canonical name + type
- [ ] `GraphStore.upsert_relationship()` deduplicates by (src, tgt, type)
- [ ] `extract_entities_from_chunk()` returns `ChunkExtraction` via Llama 70B
- [ ] `IngestPipeline` with `graph_store` extracts and stores entities
- [ ] `IngestPipeline` without `graph_store` works exactly as before
- [ ] Extraction failure per-chunk doesn't abort the ingest
- [ ] All existing 302 tests still pass (no regressions)
- [ ] New tests bring total to ~340-350

## Test Count Target

| Test file | New tests | Running total |
|-----------|-----------|---------------|
| test_db.py | +7 schema | ~17 |
| test_entities.py | ~14 models + ~10 normalization | ~24 |
| test_graph.py | ~22 | ~22 |
| test_entity_extraction.py | ~8 | ~8 |
| test_pipeline.py | ~7 | ~16 |
| **New total** | **~58** | **~360** |
