# Phase 2, Step 6: Editor Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Editor agent that verifies article claims against the knowledge base and knowledge graph, computes confidence scores, and conditionally promotes articles to review status.

**Architecture:** The Editor overrides `execute()` with a deterministic Python loop (no ReAct). For each claim, it gathers evidence from `rag_search` and `graph_search` tools, then calls `get_structured_output()` with `ClaimJudgment` schema for per-claim LLM judgment. After all claims, it computes article confidence, updates frontmatter, and optionally transitions status from draft to review.

**Tech Stack:** Python 3.12, Pydantic v2, pytest (async), instructor/OpenAI-compat structured output

**Design doc:** `docs/plans/2026-02-11-phase2-step6-editor-agent.md`

**Starting test count:** 527

---

## Errata: Bugs Found in Design Doc

The following issues were identified during codebase audit and are corrected in this plan:

| # | Design Doc Says | Actual Codebase | Fix |
|---|----------------|-----------------|-----|
| 1 | Constructor: `max_iterations: int = config.AGENT_MAX_ITERATIONS` | Writer uses `int \| None = None` with lazy `from code.shukketsu import config` inside `__init__` to avoid circular imports | Use Writer's pattern: `int \| None = None`, resolve in body |
| 2 | `self._tool_registry` (private) | `BaseAgent` sets `self.tool_registry` (public, no underscore) | Use `self.tool_registry` |
| 3 | Non-EditTask → "fall back to `super().execute()`" | Writer returns `AgentResult(status=FAILED)` for non-WriteTask. Falling back to `super()` runs the ReAct loop, which is nonsensical for the Editor (no ReAct tools). | Return FAILED `AgentResult` like Writer does |
| 4 | Step 7: "Update `meta.updated_at` to now" before `update_draft()` | `KnowledgeManager.update_draft()` already updates `updated_at` internally (line 177-178) | Remove redundant timestamp update from Editor |
| 5 | "Add Editor to `_ROLE_PROMPTS` dict" (implied) | `_ROLE_PROMPTS` already contains `AgentRole.EDITOR: config.EDITOR_SYSTEM_PROMPT` | No change needed to `_ROLE_PROMPTS` |
| 6 | Missing `ToolNotFoundError` import | `ToolRegistry.get()` raises `ToolNotFoundError` from `resilience.errors` | Import and catch `ToolNotFoundError` explicitly |
| 7 | Design doc omits `await` on `tool.execute()` calls | `RagSearchTool.execute()` and `GraphSearchTool.execute()` are both `async def` | All `tool.execute()` calls must be `await`ed |
| 8 | Design says "Add Editor to `_ROLE_CLASSES` dict" | `_ROLE_CLASSES` currently has Researcher + Writer only | Correct — must add `AgentRole.EDITOR: Editor` |

---

## Task 1: New Models in `tasks.py`

**Files:**
- Modify: `code/shukketsu/agents/tasks.py` (after `EditTask` class, ~line 142)
- Test: `tests/unit/test_tasks.py` (after `TestEditTask` class, ~line 269)

### Step 1: Write the failing tests

Append to `tests/unit/test_tasks.py`, after the `TestEditTask` class (before `TestSubTask`):

```python
# --- Add these imports at the top of the file ---
# (merge into existing imports from code.shukketsu.agents.tasks)
# Add: ClaimJudgment, ClaimVerification, EditResult, VerificationStatus

class TestVerificationStatus:
    def test_values(self) -> None:
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.UNCERTAIN == "uncertain"
        assert VerificationStatus.CONTRADICTED == "contradicted"
        assert VerificationStatus.UNSUPPORTED == "unsupported"

    def test_is_str_enum(self) -> None:
        assert isinstance(VerificationStatus.VERIFIED, str)


class TestClaimVerification:
    def test_valid(self) -> None:
        cv = ClaimVerification(
            claim="Hit cap is 142",
            status=VerificationStatus.VERIFIED,
            supporting_evidence=["chunk:15"],
            confidence=0.85,
            note="Confirmed by two sources",
        )
        assert cv.claim == "Hit cap is 142"
        assert cv.confidence == 0.85

    def test_confidence_bounds_low(self) -> None:
        with pytest.raises(ValidationError):
            ClaimVerification(
                claim="x", status=VerificationStatus.VERIFIED, confidence=-0.1
            )

    def test_confidence_bounds_high(self) -> None:
        with pytest.raises(ValidationError):
            ClaimVerification(
                claim="x", status=VerificationStatus.VERIFIED, confidence=1.1
            )

    def test_defaults(self) -> None:
        cv = ClaimVerification(
            claim="x", status=VerificationStatus.UNCERTAIN, confidence=0.5
        )
        assert cv.supporting_evidence == []
        assert cv.contradicting_evidence == []
        assert cv.note == ""


class TestClaimJudgment:
    def test_valid(self) -> None:
        cj = ClaimJudgment(
            status=VerificationStatus.CONTRADICTED,
            confidence=0.1,
            supporting=[],
            contradicting=["Source says 9%, not 8%"],
            note="Numeric mismatch",
        )
        assert cj.status == VerificationStatus.CONTRADICTED
        assert len(cj.contradicting) == 1


class TestEditResult:
    def test_valid(self) -> None:
        result = EditResult(
            task_id="t1",
            agent_role=AgentRole.EDITOR,
            status=TaskStatus.SUCCESS,
            output="Verified 3 claims",
            article_path="combat/gear/trinkets.md",
            claim_results=[],
            overall_confidence=0.75,
            approved_for_review=True,
        )
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.approved_for_review is True

    def test_defaults(self) -> None:
        result = EditResult(
            task_id="t1",
            agent_role=AgentRole.EDITOR,
            status=TaskStatus.SUCCESS,
            output="ok",
            article_path="p",
        )
        assert result.claim_results == []
        assert result.overall_confidence == 0.0
        assert result.approved_for_review is False
        assert result.internal_consistency is True
        assert result.corrections == []
        assert result.needs_more_research == []

    def test_inherits_agent_result(self) -> None:
        result = EditResult(
            task_id="t1",
            agent_role=AgentRole.EDITOR,
            status=TaskStatus.FAILED,
            output="err",
            article_path="p",
        )
        assert isinstance(result, AgentResult)
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_tasks.py -v -k "TestVerificationStatus or TestClaimVerification or TestClaimJudgment or TestEditResult" 2>&1 | tail -20
```

Expected: FAIL — `ImportError: cannot import name 'VerificationStatus'`

### Step 3: Write minimal implementation

Add to `code/shukketsu/agents/tasks.py`, after the `EditTask` class (~line 142) and before `SubTask`:

```python
class VerificationStatus(StrEnum):
    """Outcome of verifying a single claim against the knowledge base."""

    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str
    status: VerificationStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = ""


class ClaimJudgment(BaseModel):
    """LLM's assessment of a single claim against gathered evidence."""

    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)
    note: str


class EditResult(AgentResult):
    """Structured output from the Editor agent."""

    article_path: str
    claim_results: list[ClaimVerification] = Field(default_factory=list)
    overall_confidence: float = 0.0
    approved_for_review: bool = False
    internal_consistency: bool = True
    corrections: list[str] = Field(default_factory=list)
    needs_more_research: list[str] = Field(default_factory=list)
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_tasks.py -v -k "TestVerificationStatus or TestClaimVerification or TestClaimJudgment or TestEditResult" 2>&1 | tail -20
```

Expected: 10 PASSED

### Step 5: Commit

```bash
git add code/shukketsu/agents/tasks.py tests/unit/test_tasks.py
git commit -m "feat(agents): add VerificationStatus, ClaimVerification, ClaimJudgment, EditResult models"
```

---

## Task 2: Editor Prompts

**Files:**
- Create: `code/shukketsu/llm/prompts/editor.py`
- Modify: `code/shukketsu/config.py` (~line 124-127, replace stub)

### Step 1: Create the prompt module

Create `code/shukketsu/llm/prompts/editor.py`:

```python
"""Editor agent system prompt.

This module is a pure string constant with zero imports.
config.py imports from here — never the reverse (prevents circular imports).
"""

EDITOR_SYSTEM_PROMPT = """\
You are a Fact-Checking Editor for WoW: The Burning Crusade (TBC) Rogue content.
Your job is to verify claims in draft articles against evidence from the knowledge \
base and knowledge graph.

## Verification Rules

For each claim, you will receive:
- The claim text
- Evidence from the knowledge base (text search results)
- Evidence from the knowledge graph (entity relationships)

Assign a verification status:
- VERIFIED: 2+ independent pieces of evidence agree with the claim
- UNCERTAIN: Some evidence supports the claim but not conclusive (1 source, or weak match)
- CONTRADICTED: Evidence directly conflicts with the claim — flag the contradiction clearly
- UNSUPPORTED: No relevant evidence found in the knowledge base

## What Counts as Evidence

NUMBERS: Stat caps, DPS values, proc rates, item stats — must match exactly
RELATIONSHIPS: "X drops from Y", "X is BiS for Z" — verify against graph data
RANKINGS: "A is better than B" — look for comparative evidence
MECHANICS: Combat formulas, hit tables, proc behavior — verify against mechanic descriptions
PHASE ACCURACY: "Available in Phase X" — verify against phase data in graph

## Confidence Scoring

- VERIFIED with strong evidence: 0.8-1.0
- VERIFIED with moderate evidence: 0.6-0.8
- UNCERTAIN: 0.3-0.6
- CONTRADICTED: 0.0-0.2 (low confidence that the claim is correct)
- UNSUPPORTED: 0.2-0.4 (absence of evidence is not evidence of absence)

## You Do NOT

- Rewrite the article (that's the Writer's job)
- Research new topics (that's the Researcher's job)
- Make judgment calls on contradictions (that's the human's job)
- Invent evidence you weren't given\
"""

VERIFICATION_PROMPT = """\
Verify the following claim against the provided evidence.

## Claim
{claim}

## Knowledge Base Evidence
{rag_evidence}

## Knowledge Graph Evidence
{graph_evidence}

Assess whether the evidence supports, contradicts, or is silent on this claim.\
"""
```

### Step 2: Update `config.py` to import from the new module

Replace the existing stub in `code/shukketsu/config.py` (~lines 124-127). Find:

```python
EDITOR_SYSTEM_PROMPT = (
    "You are a Fact-Checking Editor for WoW TBC Rogue content. "
    "Your job is to verify claims in draft articles against the knowledge base."
)
```

Replace with:

```python
from code.shukketsu.llm.prompts.editor import (  # noqa: E402
    EDITOR_SYSTEM_PROMPT as EDITOR_SYSTEM_PROMPT,
)
```

Also add the confidence threshold constant. Find a suitable location after the Editor import (or with other agent config). Add:

```python
EDITOR_CONFIDENCE_THRESHOLD = float(os.getenv("EDITOR_CONFIDENCE_THRESHOLD", "0.6"))
```

### Step 3: Run existing factory test to verify prompt import works

```bash
python3 -m pytest tests/unit/test_agent_factory.py::TestAgentFactoryCreate::test_editor_prompt_mentions_verify -v 2>&1 | tail -10
```

Expected: PASSED (the existing test checks that `"verify"` is in the Editor prompt — our new prompt contains it)

### Step 4: Commit

```bash
git add code/shukketsu/llm/prompts/editor.py code/shukketsu/config.py
git commit -m "feat(llm): add Editor system prompt and verification prompt template"
```

---

## Task 3: Confidence Scoring + Entity Matching + Frontmatter Update

These are three module-level pure functions in `editor.py`, tested independently before the full agent.

**Files:**
- Create: `code/shukketsu/agents/editor.py` (functions only, no class yet)
- Create: `tests/unit/test_editor.py`

### Step 1: Write failing tests for all three functions

Create `tests/unit/test_editor.py`:

```python
"""Tests for the Editor agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.agents.editor import (
    Editor,
    _match_entities,
    _update_claims,
    compute_article_confidence,
)
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    TaskStatus,
    VerificationStatus,
)
from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestComputeArticleConfidence:
    def test_basic_average(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=0.8),
            ClaimVerification(claim="b", status=VerificationStatus.VERIFIED, confidence=0.6),
        ]
        assert compute_article_confidence(claims) == pytest.approx(0.7)

    def test_contradiction_penalty(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=0.8),
            ClaimVerification(claim="b", status=VerificationStatus.CONTRADICTED, confidence=0.1),
        ]
        # average = 0.45, penalty = 1 * 0.15 = 0.15, result = 0.30
        assert compute_article_confidence(claims) == pytest.approx(0.30)

    def test_empty_claims(self) -> None:
        assert compute_article_confidence([]) == 0.0

    def test_clamps_to_zero(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.CONTRADICTED, confidence=0.1),
            ClaimVerification(claim="b", status=VerificationStatus.CONTRADICTED, confidence=0.1),
            ClaimVerification(claim="c", status=VerificationStatus.CONTRADICTED, confidence=0.1),
        ]
        # average = 0.1, penalty = 3 * 0.15 = 0.45, result = max(0.0, -0.35) = 0.0
        assert compute_article_confidence(claims) == 0.0

    def test_clamps_to_one(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=1.0),
        ]
        assert compute_article_confidence(claims) == pytest.approx(1.0)

    def test_single_claim(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.UNCERTAIN, confidence=0.5),
        ]
        assert compute_article_confidence(claims) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


class TestMatchEntities:
    def test_exact_match(self) -> None:
        result = _match_entities("Dragonspine Trophy drops from Gruul", ["Dragonspine Trophy"])
        assert result == ["Dragonspine Trophy"]

    def test_case_insensitive(self) -> None:
        result = _match_entities("dragonspine trophy is BiS", ["Dragonspine Trophy"])
        assert result == ["Dragonspine Trophy"]

    def test_no_match(self) -> None:
        result = _match_entities("Hit cap is 142 rating", ["Dragonspine Trophy", "Gruul"])
        assert result == []

    def test_multiple_matches(self) -> None:
        result = _match_entities(
            "Dragonspine Trophy drops from Gruul",
            ["Dragonspine Trophy", "Gruul", "Magtheridon"],
        )
        assert set(result) == {"Dragonspine Trophy", "Gruul"}


# ---------------------------------------------------------------------------
# Frontmatter update
# ---------------------------------------------------------------------------


def _article_meta(**overrides: Any) -> ArticleMeta:
    """Build an ArticleMeta with sensible defaults."""
    from datetime import UTC, datetime

    defaults: dict[str, Any] = dict(
        title="Test Article",
        spec=Spec.COMBAT,
        category="gear",
        status=ArticleStatus.DRAFT,
        confidence=0.0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        sources=[],
        claims=[
            ClaimRef(text="Hit cap is 142", verified=False, evidence=[]),
            ClaimRef(text="DST is BiS", verified=False, evidence=["chunk:42"]),
        ],
        entity_refs=["Hit Rating", "Dragonspine Trophy"],
        tags=["combat", "gear"],
    )
    defaults.update(overrides)
    return ArticleMeta(**defaults)


class TestUpdateClaims:
    def test_marks_verified(self) -> None:
        meta = _article_meta()
        results = [
            ClaimVerification(
                claim="Hit cap is 142",
                status=VerificationStatus.VERIFIED,
                supporting_evidence=["chunk:15"],
                confidence=0.9,
            ),
        ]
        _update_claims(meta, results)
        hit_claim = next(c for c in meta.claims if c.text == "Hit cap is 142")
        assert hit_claim.verified is True

    def test_marks_unverified(self) -> None:
        meta = _article_meta()
        results = [
            ClaimVerification(
                claim="DST is BiS",
                status=VerificationStatus.CONTRADICTED,
                contradicting_evidence=["Source says Tsunami Talisman is BiS"],
                confidence=0.1,
            ),
        ]
        _update_claims(meta, results)
        dst_claim = next(c for c in meta.claims if c.text == "DST is BiS")
        assert dst_claim.verified is False

    def test_merges_evidence_deduped(self) -> None:
        meta = _article_meta()
        results = [
            ClaimVerification(
                claim="DST is BiS",
                status=VerificationStatus.VERIFIED,
                supporting_evidence=["chunk:42", "chunk:99"],
                confidence=0.85,
            ),
        ]
        _update_claims(meta, results)
        dst_claim = next(c for c in meta.claims if c.text == "DST is BiS")
        # chunk:42 was already in evidence, should not be duplicated
        assert dst_claim.evidence == ["chunk:42", "chunk:99"]

    def test_unmatched_claims_ignored(self) -> None:
        meta = _article_meta()
        results = [
            ClaimVerification(
                claim="Nonexistent claim",
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
            ),
        ]
        _update_claims(meta, results)
        # Original claims unchanged
        assert meta.claims[0].verified is False
        assert meta.claims[1].verified is False
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_editor.py -v -k "TestComputeArticleConfidence or TestMatchEntities or TestUpdateClaims" 2>&1 | tail -20
```

Expected: FAIL — `ImportError: cannot import name 'compute_article_confidence' from 'code.shukketsu.agents.editor'`

### Step 3: Write minimal implementation

Create `code/shukketsu/agents/editor.py` (functions only for now — class added in Task 4):

```python
"""Editor specialist agent.

Overrides execute() with deterministic claim-by-claim verification.
No ReAct loop — the Editor gathers evidence from rag_search and
graph_search tools, then uses structured LLM output to judge each claim.
"""

import logging
from typing import Any

from code.shukketsu.agents.tasks import (
    ClaimVerification,
    VerificationStatus,
)
from code.shukketsu.knowledge.manager import ArticleMeta

logger = logging.getLogger(__name__)


def compute_article_confidence(claims: list[ClaimVerification]) -> float:
    """Compute overall article confidence from individual claim verifications.

    Returns the average claim confidence with a penalty for contradicted claims.
    Each contradicted claim reduces the score by 0.15.
    """
    if not claims:
        return 0.0
    base = sum(c.confidence for c in claims) / len(claims)
    contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
    penalty = contradicted * 0.15
    return max(0.0, min(1.0, base - penalty))


def _match_entities(claim: str, entity_refs: list[str]) -> list[str]:
    """Find entity_refs that appear in the claim text (case-insensitive)."""
    claim_lower = claim.lower()
    return [e for e in entity_refs if e.lower() in claim_lower]


def _update_claims(meta: ArticleMeta, results: list[ClaimVerification]) -> None:
    """Update article ClaimRefs based on verification results."""
    result_map = {r.claim: r for r in results}
    for claim_ref in meta.claims:
        if claim_ref.text in result_map:
            verification = result_map[claim_ref.text]
            claim_ref.verified = verification.status == VerificationStatus.VERIFIED
            new_evidence = verification.supporting_evidence + verification.contradicting_evidence
            claim_ref.evidence = list(dict.fromkeys(claim_ref.evidence + new_evidence))
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_editor.py -v -k "TestComputeArticleConfidence or TestMatchEntities or TestUpdateClaims" 2>&1 | tail -20
```

Expected: 14 PASSED

### Step 5: Commit

```bash
git add code/shukketsu/agents/editor.py tests/unit/test_editor.py
git commit -m "feat(agents): add confidence scoring, entity matching, frontmatter update functions"
```

---

## Task 4: Editor Agent Class — Validation + Empty Claims Guard

**Files:**
- Modify: `code/shukketsu/agents/editor.py` (add `Editor` class with validation only)
- Modify: `tests/unit/test_editor.py` (add validation tests)

### Step 1: Write failing tests

Append to `tests/unit/test_editor.py`:

```python
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def km(test_db: sqlite3.Connection, knowledge_dir: Path) -> KnowledgeManager:
    return KnowledgeManager(test_db, knowledge_dir)


@pytest.fixture
def editor(km: KnowledgeManager) -> Editor:
    return Editor(
        tool_registry=ToolRegistry(),
        role=AgentRole.EDITOR,
        knowledge_manager=km,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestEditorValidation:
    async def test_execute_requires_role(self, km: KnowledgeManager) -> None:
        editor = Editor(tool_registry=ToolRegistry(), role=None, knowledge_manager=km)
        task = EditTask(query="Verify", article_path="p", claims=["x"])
        with pytest.raises(ValueError, match="role"):
            await editor.execute(task)

    async def test_execute_requires_edit_task(self, editor: Editor) -> None:
        task = AgentTask(query="Not an edit task")
        result = await editor.execute(task)
        assert result.status == TaskStatus.FAILED
        assert isinstance(result, AgentResult)

    async def test_execute_empty_claims_returns_failed(self, editor: Editor, km: KnowledgeManager, knowledge_dir: Path) -> None:
        # Create a draft article so read_article works
        km.create_draft("test.md", _article_meta(), "# Test")
        task = EditTask(query="Verify", article_path="test.md", claims=[])
        result = await editor.execute(task)
        assert result.status == TaskStatus.FAILED
        assert isinstance(result, EditResult)
        assert result.overall_confidence == 0.0
        assert result.approved_for_review is False
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_editor.py::TestEditorValidation -v 2>&1 | tail -20
```

Expected: FAIL — `ImportError: cannot import name 'Editor'`

### Step 3: Write minimal implementation

Add to `code/shukketsu/agents/editor.py`. Update imports at top and add class:

```python
# Add to existing imports:
from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    TaskStatus,
    VerificationStatus,
)
from code.shukketsu.knowledge.manager import ArticleMeta, ArticleStatus, KnowledgeManager
from code.shukketsu.llm.prompts.editor import EDITOR_SYSTEM_PROMPT, VERIFICATION_PROMPT
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import ToolNotFoundError
from code.shukketsu.tools.registry import ToolRegistry


class Editor(BaseAgent):
    """Fact-checking editor that verifies article claims against the knowledge base."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        from code.shukketsu import config

        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations or config.AGENT_MAX_ITERATIONS,
            system_prompt=system_prompt or config.SYSTEM_PROMPT,
        )
        self._km = knowledge_manager

    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> EditResult | AgentResult:
        """Verify claims in a draft article against the knowledge base.

        Args:
            task: Must be an EditTask with article_path and claims.
            on_status: Optional async callback for progress updates.

        Returns:
            EditResult on success/partial, AgentResult with FAILED for bad input.
        """
        # Step 1: Validate input
        if self.role is None:
            raise ValueError("Editor requires a role. Use AgentFactory or set role in constructor.")

        if not isinstance(task, EditTask):
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="Editor requires an EditTask",
            )

        # Step 2: Guard — empty claims
        if not task.claims:
            return EditResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="No claims to verify",
                article_path=task.article_path,
                overall_confidence=0.0,
                approved_for_review=False,
            )

        # Full verification flow implemented in Task 5
        raise NotImplementedError("Verification flow not yet implemented")
```

**Important:** The full `execute()` body is a `NotImplementedError` stub — we'll fill it in Task 5. This keeps Task 4 focused on validation tests only.

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_editor.py::TestEditorValidation -v 2>&1 | tail -20
```

Expected: 3 PASSED

### Step 5: Commit

```bash
git add code/shukketsu/agents/editor.py tests/unit/test_editor.py
git commit -m "feat(agents): add Editor class with input validation and empty claims guard"
```

---

## Task 5: Editor Agent — Full Verification Flow

This is the core of the Editor. The execute() method loops over claims, gathers evidence, calls the LLM, and produces the EditResult.

**Files:**
- Modify: `code/shukketsu/agents/editor.py` (replace `NotImplementedError` with full flow)
- Modify: `tests/unit/test_editor.py` (add verification flow tests)

### Step 1: Write failing tests

Append to `tests/unit/test_editor.py`. First add test helpers, then the test class:

```python
# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubRagSearch(Tool):
    """Stub rag_search tool that returns canned results."""

    name = "rag_search"
    description = "Search the knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Search query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        q = tool_input.get("query", "")
        return f"Found 1 result:\n[1] Source: Test Guide\nTrust: 0.8\nContent: Info about {q}\n"


class _StubGraphSearch(Tool):
    """Stub graph_search tool that returns canned results."""

    name = "graph_search"
    description = "Search the knowledge graph."
    parameters_schema = {"entity": {"type": "string", "description": "Entity name"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        e = tool_input.get("entity", "")
        return f"Entity: {e} (type: item, confidence: 0.9)\nOutgoing: drops_from -> Gruul\n"


class _EmptyRagSearch(Tool):
    """Stub rag_search that returns no results."""

    name = "rag_search"
    description = "Search the knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Search query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "No relevant documents found for this query."


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _edit_task(**overrides: Any) -> EditTask:
    defaults: dict[str, Any] = dict(
        query="Verify combat trinkets article",
        article_path="test.md",
        claims=["Hit cap is 142 rating", "DST drops from Gruul"],
    )
    defaults.update(overrides)
    return EditTask(**defaults)


def _mock_judgment(
    status: VerificationStatus = VerificationStatus.VERIFIED,
    confidence: float = 0.85,
    **kwargs: Any,
) -> ClaimJudgment:
    defaults: dict[str, Any] = dict(
        status=status,
        confidence=confidence,
        supporting=["Evidence supports this"],
        contradicting=[],
        note="Confirmed",
    )
    defaults.update(kwargs)
    return ClaimJudgment(**defaults)


def _write_test_article(km: KnowledgeManager, path: str = "test.md", **meta_overrides: Any) -> None:
    """Write a draft article to disk+DB for testing."""
    meta = _article_meta(**meta_overrides)
    km.create_draft(path, meta, "# Test Article\n\nSome content about combat trinkets.")


# ---------------------------------------------------------------------------
# Verification flow
# ---------------------------------------------------------------------------


class TestEditorVerification:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_calls_rag_search_per_claim(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment()
        _write_test_article(km)
        rag = _StubRagSearch()
        reg = _registry(rag)
        editor = Editor(tool_registry=reg, role=AgentRole.EDITOR, knowledge_manager=km)
        task = _edit_task()
        await editor.execute(task)
        # LLM called once per claim (2 claims)
        assert mock_llm.call_count == 2

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_calls_graph_search_for_matched_entities(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment()
        _write_test_article(km)
        rag = _StubRagSearch()
        graph = _StubGraphSearch()
        reg = _registry(rag, graph)
        editor = Editor(tool_registry=reg, role=AgentRole.EDITOR, knowledge_manager=km)
        # "DST drops from Gruul" should match entity_ref "Dragonspine Trophy" — NO (DST != Dragonspine Trophy)
        # But "Hit Rating" should match "Hit cap is 142 rating" — YES (case-insensitive substring)
        # Let's use a claim that contains an actual entity_ref
        task = _edit_task(claims=["Dragonspine Trophy drops from Gruul"])
        await editor.execute(task)
        # Should have called graph_search for "Dragonspine Trophy"
        assert mock_llm.call_count == 1
        call_kwargs = mock_llm.call_args.kwargs
        user_msg = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "user")
        assert "Gruul" in user_msg  # graph evidence included

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_skips_graph_search_no_entity_match(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment()
        _write_test_article(km, entity_refs=[])
        rag = _StubRagSearch()
        graph = _StubGraphSearch()
        reg = _registry(rag, graph)
        editor = Editor(tool_registry=reg, role=AgentRole.EDITOR, knowledge_manager=km)
        task = _edit_task(claims=["Some claim with no entities"])
        await editor.execute(task)
        call_kwargs = mock_llm.call_args.kwargs
        user_msg = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "user")
        # Graph evidence section should indicate no results
        assert "No graph evidence" in user_msg or "None" in user_msg

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_returns_edit_result(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment()
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task())
        assert isinstance(result, EditResult)
        assert result.article_path == "test.md"
        assert len(result.claim_results) == 2

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_verified_claim(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.9)
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Hit cap is 142 rating"]))
        assert result.claim_results[0].status == VerificationStatus.VERIFIED
        assert result.claim_results[0].confidence == 0.9

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_contradicted_claim_populates_corrections(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(
            status=VerificationStatus.CONTRADICTED,
            confidence=0.1,
            note="Evidence says 9%, not 8%",
        )
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Hit cap is 8%"]))
        assert len(result.corrections) == 1
        assert "contradicted" in result.corrections[0].lower()

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_unsupported_claim_populates_research_gaps(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(
            status=VerificationStatus.UNSUPPORTED,
            confidence=0.3,
            note="No evidence found",
        )
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Proc rate is 3.7 PPM"]))
        assert "Proc rate is 3.7 PPM" in result.needs_more_research

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_mixed_claims(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.side_effect = [
            _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.9),
            _mock_judgment(status=VerificationStatus.CONTRADICTED, confidence=0.1, note="Wrong"),
            _mock_judgment(status=VerificationStatus.UNSUPPORTED, confidence=0.3, note="No evidence"),
        ]
        _write_test_article(km, claims=[
            ClaimRef(text="c1"), ClaimRef(text="c2"), ClaimRef(text="c3"),
        ])
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["c1", "c2", "c3"]))
        assert result.status == TaskStatus.SUCCESS
        assert len(result.claim_results) == 3
        assert len(result.corrections) == 1
        assert len(result.needs_more_research) == 1
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_editor.py::TestEditorVerification -v 2>&1 | tail -20
```

Expected: FAIL — `NotImplementedError: Verification flow not yet implemented`

### Step 3: Write full implementation

Replace the `NotImplementedError` stub in `execute()` in `code/shukketsu/agents/editor.py` with the complete flow. The final file should look like this:

```python
"""Editor specialist agent.

Overrides execute() with deterministic claim-by-claim verification.
No ReAct loop — the Editor gathers evidence from rag_search and
graph_search tools, then uses structured LLM output to judge each claim.
"""

import logging
from typing import Any

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    TaskStatus,
    VerificationStatus,
)
from code.shukketsu.knowledge.manager import ArticleMeta, ArticleStatus, KnowledgeManager
from code.shukketsu.llm.prompts.editor import EDITOR_SYSTEM_PROMPT, VERIFICATION_PROMPT
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import ToolNotFoundError
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def compute_article_confidence(claims: list[ClaimVerification]) -> float:
    """Compute overall article confidence from individual claim verifications.

    Returns the average claim confidence with a penalty for contradicted claims.
    Each contradicted claim reduces the score by 0.15.
    """
    if not claims:
        return 0.0
    base = sum(c.confidence for c in claims) / len(claims)
    contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
    penalty = contradicted * 0.15
    return max(0.0, min(1.0, base - penalty))


def _match_entities(claim: str, entity_refs: list[str]) -> list[str]:
    """Find entity_refs that appear in the claim text (case-insensitive)."""
    claim_lower = claim.lower()
    return [e for e in entity_refs if e.lower() in claim_lower]


def _update_claims(meta: ArticleMeta, results: list[ClaimVerification]) -> None:
    """Update article ClaimRefs based on verification results."""
    result_map = {r.claim: r for r in results}
    for claim_ref in meta.claims:
        if claim_ref.text in result_map:
            verification = result_map[claim_ref.text]
            claim_ref.verified = verification.status == VerificationStatus.VERIFIED
            new_evidence = verification.supporting_evidence + verification.contradicting_evidence
            claim_ref.evidence = list(dict.fromkeys(claim_ref.evidence + new_evidence))


class Editor(BaseAgent):
    """Fact-checking editor that verifies article claims against the knowledge base."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        from code.shukketsu import config

        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations or config.AGENT_MAX_ITERATIONS,
            system_prompt=system_prompt or config.SYSTEM_PROMPT,
        )
        self._km = knowledge_manager

    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> EditResult | AgentResult:
        """Verify claims in a draft article against the knowledge base.

        Args:
            task: Must be an EditTask with article_path and claims.
            on_status: Optional async callback for progress updates.

        Returns:
            EditResult on success/partial, AgentResult with FAILED for bad input.
        """
        # Step 1: Validate input
        if self.role is None:
            raise ValueError("Editor requires a role. Use AgentFactory or set role in constructor.")

        if not isinstance(task, EditTask):
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="Editor requires an EditTask",
            )

        # Step 2: Guard — empty claims
        if not task.claims:
            return EditResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="No claims to verify",
                article_path=task.article_path,
                overall_confidence=0.0,
                approved_for_review=False,
            )

        # Step 3: Read article
        meta, content = self._km.read_article(task.article_path)
        entity_refs = meta.entity_refs

        if on_status:
            await on_status(f"verifying {len(task.claims)} claims...")

        # Step 4: Verify each claim
        claim_results: list[ClaimVerification] = []
        failures = 0

        for i, claim in enumerate(task.claims):
            if on_status:
                await on_status(f"checking claim {i + 1}/{len(task.claims)}...")

            try:
                verification = await self._verify_claim(claim, entity_refs)
            except Exception as exc:
                logger.warning("Claim verification failed for '%s': %s", claim[:50], exc)
                verification = ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNSUPPORTED,
                    confidence=0.0,
                    note=f"Verification failed: {exc}",
                )
                failures += 1

            claim_results.append(verification)

        # Step 5: Compute confidence
        overall_confidence = compute_article_confidence(claim_results)

        # Step 6: Build corrections and research gaps
        corrections: list[str] = []
        needs_more_research: list[str] = []
        for cv in claim_results:
            if cv.status == VerificationStatus.CONTRADICTED:
                corrections.append(f"Claim '{cv.claim}' contradicted: {cv.note}")
            elif cv.status == VerificationStatus.UNSUPPORTED:
                needs_more_research.append(cv.claim)

        # Step 7: Update article frontmatter (only if still in draft)
        approved_for_review = False
        try:
            _update_claims(meta, claim_results)
            meta_updated = meta.model_copy(update={"confidence": overall_confidence})
            self._km.update_draft(task.article_path, meta_updated, content)
        except ValueError as exc:
            logger.info("Skipping frontmatter update: %s", exc)

        # Step 8: Conditional status transition
        from code.shukketsu import config

        if overall_confidence >= config.EDITOR_CONFIDENCE_THRESHOLD:
            try:
                self._km.set_status(task.article_path, ArticleStatus.REVIEW)
                approved_for_review = True
            except ValueError as exc:
                logger.info("Status transition failed: %s", exc)

        # Step 9: Determine overall status
        if failures == len(task.claims):
            status = TaskStatus.PARTIAL
        else:
            status = TaskStatus.SUCCESS

        return EditResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=status,
            output=f"Verified {len(claim_results)} claims. Confidence: {overall_confidence:.2f}",
            article_path=task.article_path,
            claim_results=claim_results,
            overall_confidence=overall_confidence,
            approved_for_review=approved_for_review,
            corrections=corrections,
            needs_more_research=needs_more_research,
        )

    async def _verify_claim(self, claim: str, entity_refs: list[str]) -> ClaimVerification:
        """Gather evidence and judge a single claim."""
        # RAG search
        rag_text = await self._search_rag(claim)

        # Graph search for matched entities
        matched = _match_entities(claim, entity_refs)
        graph_parts: list[str] = []
        for entity in matched:
            result = await self._search_graph(entity)
            if result:
                graph_parts.append(result)
        graph_text = "\n".join(graph_parts) if graph_parts else "No graph evidence available."

        # LLM judgment
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VERIFICATION_PROMPT.format(
                    claim=claim,
                    rag_evidence=rag_text,
                    graph_evidence=graph_text,
                ),
            },
        ]
        judgment = await get_structured_output(
            response_model=ClaimJudgment,
            messages=messages,
        )

        return ClaimVerification(
            claim=claim,
            status=judgment.status,
            supporting_evidence=judgment.supporting,
            contradicting_evidence=judgment.contradicting,
            confidence=judgment.confidence,
            note=judgment.note,
        )

    async def _search_rag(self, query: str) -> str:
        """Search the knowledge base via rag_search tool."""
        try:
            tool = self.tool_registry.get("rag_search")
            return await tool.execute({"query": query})
        except ToolNotFoundError:
            logger.debug("rag_search tool not available, skipping")
            return "No RAG evidence available (tool not configured)."
        except Exception as exc:
            logger.warning("rag_search failed: %s", exc)
            return f"RAG search error: {exc}"

    async def _search_graph(self, entity: str) -> str:
        """Search the knowledge graph via graph_search tool."""
        try:
            tool = self.tool_registry.get("graph_search")
            return await tool.execute({"entity": entity})
        except ToolNotFoundError:
            logger.debug("graph_search tool not available, skipping")
            return ""
        except Exception as exc:
            logger.warning("graph_search failed for %s: %s", entity, exc)
            return ""
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_editor.py::TestEditorVerification -v 2>&1 | tail -20
```

Expected: 8 PASSED

### Step 5: Commit

```bash
git add code/shukketsu/agents/editor.py tests/unit/test_editor.py
git commit -m "feat(agents): implement Editor verification flow with evidence gathering and LLM judgment"
```

---

## Task 6: Editor Agent — Confidence Threshold + Status Transition + Error Handling

**Files:**
- Modify: `tests/unit/test_editor.py` (add threshold, transition, and error tests)

### Step 1: Write failing tests

Append to `tests/unit/test_editor.py`:

```python
class TestEditorConfidenceAndApproval:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_above_threshold_approves(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.9)
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Hit cap is 142 rating"]))
        assert result.approved_for_review is True

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_below_threshold_rejects(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.UNCERTAIN, confidence=0.3)
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Some uncertain claim"]))
        assert result.approved_for_review is False

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_updates_frontmatter_confidence(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.85)
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        await editor.execute(_edit_task(claims=["Hit cap is 142 rating"]))
        # Read back the article and check confidence was updated
        meta, _ = km.read_article("test.md")
        assert meta.confidence == pytest.approx(0.85)

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_transitions_to_review(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.9)
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        await editor.execute(_edit_task(claims=["Hit cap is 142 rating"]))
        # Article should now be in review status
        row = km._conn.execute("SELECT status FROM articles WHERE path = ?", ("test.md",)).fetchone()
        assert row["status"] == ArticleStatus.REVIEW


class TestEditorErrorHandling:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_llm_failure_marks_unsupported(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.side_effect = Exception("LLM unavailable")
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Test claim"]))
        assert result.claim_results[0].status == VerificationStatus.UNSUPPORTED
        assert "failed" in result.claim_results[0].note.lower()

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_all_failures_returns_partial(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.side_effect = Exception("LLM down")
        _write_test_article(km)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task())
        assert result.status == TaskStatus.PARTIAL

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_missing_rag_tool_still_verifies(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment()
        _write_test_article(km)
        # Empty registry — no tools at all
        editor = Editor(tool_registry=ToolRegistry(), role=AgentRole.EDITOR, knowledge_manager=km)
        result = await editor.execute(_edit_task(claims=["Test claim"]))
        # Should still succeed — tools are best-effort
        assert result.status == TaskStatus.SUCCESS
        assert len(result.claim_results) == 1

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_non_draft_skips_update(
        self, mock_llm: AsyncMock, km: KnowledgeManager, knowledge_dir: Path
    ) -> None:
        mock_llm.return_value = _mock_judgment(status=VerificationStatus.VERIFIED, confidence=0.9)
        _write_test_article(km)
        # Move article to review status first
        km.set_status("test.md", ArticleStatus.REVIEW)
        editor = Editor(tool_registry=_registry(_StubRagSearch()), role=AgentRole.EDITOR, knowledge_manager=km)
        # Should not crash — just skip frontmatter update
        result = await editor.execute(_edit_task(claims=["Hit cap is 142 rating"]))
        assert result.status == TaskStatus.SUCCESS
        # approved_for_review should be False since set_status(REVIEW) would fail
        # (article is already in review, can't transition to review again)
        assert result.approved_for_review is False
```

### Step 2: Run tests to verify they pass

These tests should pass against the implementation from Task 5. Run:

```bash
python3 -m pytest tests/unit/test_editor.py -v -k "TestEditorConfidenceAndApproval or TestEditorErrorHandling" 2>&1 | tail -30
```

Expected: 8 PASSED

If any fail, debug and fix the `execute()` implementation.

### Step 3: Commit

```bash
git add tests/unit/test_editor.py
git commit -m "test(agents): add Editor confidence threshold, status transition, and error handling tests"
```

---

## Task 7: Factory + Exports Integration

**Files:**
- Modify: `code/shukketsu/agents/factory.py` (add Editor to `_ROLE_CLASSES`)
- Modify: `code/shukketsu/agents/__init__.py` (add new exports)
- Modify: `tests/unit/test_agent_factory.py` (update `_factory_create` helper, add Editor tests)

### Step 1: Write failing tests

Add to `tests/unit/test_agent_factory.py`. First update the `_factory_create` helper, then add tests:

Update `_factory_create` (~line 28):

```python
def _factory_create(factory: AgentFactory, role: AgentRole, **kwargs: Any) -> BaseAgent:
    """Create an agent, passing knowledge_manager for roles that need it."""
    if role in (AgentRole.WRITER, AgentRole.EDITOR) and "knowledge_manager" not in kwargs:
        kwargs["knowledge_manager"] = _mock_km()
    return factory.create(role, **kwargs)
```

Add new test class at the end of the file:

```python
class TestAgentFactoryEditor:
    def test_factory_creates_editor(self) -> None:
        from code.shukketsu.agents.editor import Editor

        factory = AgentFactory()
        agent = _factory_create(factory, AgentRole.EDITOR)
        assert isinstance(agent, Editor)

    def test_factory_editor_requires_knowledge_manager(self) -> None:
        factory = AgentFactory()
        with pytest.raises(TypeError):
            factory.create(AgentRole.EDITOR)

    def test_factory_editor_in_role_classes(self) -> None:
        from code.shukketsu.agents.factory import _ROLE_CLASSES

        assert AgentRole.EDITOR in _ROLE_CLASSES
```

Also add `import pytest` to the existing imports if not already present.

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_agent_factory.py::TestAgentFactoryEditor -v 2>&1 | tail -15
```

Expected: FAIL — `test_factory_creates_editor` creates a `BaseAgent` (not `Editor`) because `_ROLE_CLASSES` doesn't have Editor yet.

### Step 3: Write minimal implementation

**`code/shukketsu/agents/factory.py`** — Add Editor to imports and `_ROLE_CLASSES`:

Add to imports (with existing Researcher/Writer imports):
```python
from code.shukketsu.agents.editor import Editor
```

Update `_ROLE_CLASSES`:
```python
_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
    AgentRole.EDITOR: Editor,
}
```

**`code/shukketsu/agents/__init__.py`** — Add new exports:

Add to the import from `tasks`:
```python
ClaimJudgment,
ClaimVerification,
EditResult,
VerificationStatus,
```

Add import for Editor:
```python
from code.shukketsu.agents.editor import Editor
```

Add to `__all__`:
```python
"ClaimJudgment",
"ClaimVerification",
"Editor",
"EditResult",
"VerificationStatus",
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_agent_factory.py -v 2>&1 | tail -20
```

Expected: All tests PASSED (including 3 new Editor tests + all existing tests unchanged)

### Step 5: Commit

```bash
git add code/shukketsu/agents/factory.py code/shukketsu/agents/__init__.py tests/unit/test_agent_factory.py
git commit -m "feat(agents): register Editor in factory and add exports"
```

---

## Task 8: Full Suite Verification + Lint + Type Check

**Files:** None (verification only)

### Step 1: Run full test suite

```bash
python3 -m pytest tests/unit/ -v 2>&1 | tail -30
```

Expected: All tests pass. Count should be ~565-575 (527 + ~40-45 new tests).

### Step 2: Run linter

```bash
ruff check code/shukketsu/agents/editor.py code/shukketsu/llm/prompts/editor.py tests/unit/test_editor.py && ruff format code/shukketsu/agents/editor.py code/shukketsu/llm/prompts/editor.py tests/unit/test_editor.py
```

Expected: No errors.

### Step 3: Run type checker

```bash
python3 -m mypy code/shukketsu/agents/editor.py code/shukketsu/llm/prompts/editor.py
```

Expected: No type errors.

### Step 4: Fix any issues found, then commit

```bash
git add -A && git commit -m "chore: fix lint and type issues from Editor implementation"
```

(Only if fixes were needed.)

### Step 5: Final commit — update CLAUDE.md

Update the Phase 2 step 6 line in `CLAUDE.md` from:
```
6. Editor Agent (claim verification, confidence scoring, fact-checking) — **DESIGN COMPLETE, NEXT**
```
to:
```
6. ~~Editor Agent (claim verification, confidence scoring, fact-checking)~~ — COMPLETE (N tests: ...)
```

And update the test count in the opening paragraph.

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 6 completion"
```

---

## Expected Final State

| Metric | Value |
|--------|-------|
| New files | `code/shukketsu/agents/editor.py`, `code/shukketsu/llm/prompts/editor.py`, `tests/unit/test_editor.py` |
| Modified files | `tasks.py` (+4 models), `factory.py` (+1 import, +1 dict entry), `__init__.py` (+5 exports), `config.py` (replace stub + add threshold), `test_tasks.py` (+10 tests), `test_agent_factory.py` (+3 tests, helper update) |
| New tests | ~40-45 |
| Total test count | ~567-572 |
| Commits | 7-8 |
