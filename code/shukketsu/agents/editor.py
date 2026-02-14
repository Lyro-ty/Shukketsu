"""Editor specialist agent.

Overrides execute() with direct verification (no ReAct loop).
For each claim, gathers evidence from rag_search and graph_search,
then uses a single LLM call to judge verification status.
"""

import logging
import sqlite3

from langfuse import observe

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
from code.shukketsu.resilience.circuit_breaker import reasoning_breaker
from code.shukketsu.resilience.errors import CircuitOpenError
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.trust.scoring import TRUST_DELTAS, record_trust_event

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

    def _get_db_conn(self) -> sqlite3.Connection | None:
        """Get the database connection from KnowledgeManager (for trust events)."""
        try:
            return self._km.get_db_connection()
        except AttributeError:
            return None

    @observe(as_type="agent")
    async def execute(
        self,
        task: AgentTask,
        *,
        on_status: StatusCallback | None = None,
        model_name: str | None = None,
    ) -> EditResult | AgentResult:
        """Execute an edit task: verify claims in an article against the KB.

        Args:
            task: Must be an EditTask with article path and claims.
            on_status: Optional async callback for progress updates.

        Returns:
            EditResult on success, AgentResult with FAILED status on error.
        """
        from code.shukketsu import config

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

        # Step 4: Verify each claim
        claim_results: list[ClaimVerification] = []
        all_failed = True

        for i, claim in enumerate(task.claims):
            if on_status:
                await on_status(f"verifying claim {i + 1}/{len(task.claims)}...")

            try:
                verification = await self._verify_claim(claim, entity_refs)
                all_failed = False
            except (CircuitOpenError, Exception) as exc:
                logger.warning("Verification failed for claim '%s': %s", claim[:50], exc)
                verification = ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNSUPPORTED,
                    confidence=0.3,
                    note=f"Verification failed: {exc}",
                )

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

        # Step 6b: Record trust events for contradicted claims
        # Match source URLs from article frontmatter (not LLM evidence text)
        conn = self._get_db_conn()
        if conn is not None:
            try:
                source_urls = [s.url for s in meta.sources] if meta.sources else []
                contradiction_count = sum(1 for cv in claim_results if cv.status == VerificationStatus.CONTRADICTED)
                if contradiction_count > 0 and source_urls:
                    contradiction_ratio = contradiction_count / max(1, len(claim_results))
                    scaled_delta = TRUST_DELTAS["contradiction"] * contradiction_ratio
                    for url in source_urls:
                        source_row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
                        if source_row:
                            contradicted_claims = [
                                cv.claim[:100] for cv in claim_results if cv.status == VerificationStatus.CONTRADICTED
                            ]
                            record_trust_event(
                                conn,
                                source_row["id"],
                                "contradiction",
                                scaled_delta,
                                details=f"Contradicted claims: {'; '.join(contradicted_claims)}",
                            )
            except Exception:
                logger.warning("Failed to record trust events", exc_info=True)

        # Step 7: Update article frontmatter
        try:
            _update_claims(meta, claim_results)
            meta.confidence = round(overall_confidence, 4)
            self._km.update_draft(task.article_path, meta, content)
        except ValueError as exc:
            logger.warning("Could not update article frontmatter: %s", exc)

        # Step 8: Conditional status transition
        approved_for_review = False
        if overall_confidence >= config.EDITOR_CONFIDENCE_THRESHOLD:
            try:
                self._km.set_status(task.article_path, ArticleStatus.REVIEW)
                approved_for_review = True
            except ValueError as exc:
                logger.warning("Could not transition article to review: %s", exc)

        # Determine final status
        status = TaskStatus.PARTIAL if all_failed else TaskStatus.SUCCESS

        # Step 9: Return EditResult
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
        """Verify a single claim by gathering evidence and asking the LLM to judge."""
        # RAG search
        rag_text = await self._search_rag(claim)

        # Entity matching + graph search
        matched_entities = _match_entities(claim, entity_refs)
        graph_text = await self._search_graph(matched_entities)

        # LLM judgment
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VERIFICATION_PROMPT.format(
                    claim=claim,
                    rag_evidence=rag_text or "(no evidence found)",
                    graph_evidence=graph_text or "(no evidence found)",
                ),
            },
        ]

        judgment: ClaimJudgment = await reasoning_breaker.call(
            get_structured_output, response_model=ClaimJudgment, messages=messages
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
        if "rag_search" not in self.tool_registry:
            return ""
        try:
            return await self.tool_registry.execute("rag_search", {"query": query})
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)
            return ""

    async def _search_graph(self, entities: list[str]) -> str:
        """Search the knowledge graph for each matched entity."""
        if "graph_search" not in self.tool_registry:
            return ""

        parts: list[str] = []
        for entity in entities:
            try:
                result = await self.tool_registry.execute("graph_search", {"entity": entity})
                parts.append(result)
            except Exception as exc:
                logger.warning("Graph search failed for '%s': %s", entity, exc)

        return "\n\n".join(parts)
