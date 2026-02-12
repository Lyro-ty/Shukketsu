"""Writer specialist agent.

Overrides execute() with direct LLM generation (no ReAct loop).
Two-pass: generate article from findings, then extract claims/entities.
"""

import logging
import statistics
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    TaskStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)
from code.shukketsu.llm.prompts.writer import EXTRACTION_PROMPT, WRITER_SYSTEM_PROMPT
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class GeneratedArticle(BaseModel):
    """Schema for the article generation pass."""

    title: str
    content: str


class ArticleExtraction(BaseModel):
    """Schema for the claims/entity extraction pass."""

    claims: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)


class Writer(BaseAgent):
    """Wiki article writer agent.

    Uses direct LLM generation (no ReAct loop). Receives a ResearchResult,
    generates an article via Llama 70B, extracts claims/entities in a
    second pass, and writes the result via KnowledgeManager.
    """

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

    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> WriteResult | AgentResult:
        """Execute a write task: generate article from research findings.

        Args:
            task: Must be a WriteTask with ResearchResult.
            on_status: Optional async callback for progress updates.

        Returns:
            WriteResult on success, AgentResult with FAILED status on error.
        """
        # Step 1: Validate input
        if self.role is None:
            raise ValueError("Writer requires a role. Use AgentFactory or set role in constructor.")

        if not isinstance(task, WriteTask):
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="Writer requires a WriteTask",
            )

        # Step 2: Guard — empty findings
        if not task.research.findings:
            return WriteResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="No findings to write about",
                article_path="",
                title="",
            )

        # Step 3: Generate article content (Pass 1)
        if on_status:
            await on_status("generating article...")

        try:
            generated = await self._generate_article(task)
        except Exception as exc:
            logger.warning("Article generation failed: %s", exc)
            return WriteResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output=f"Article generation failed: {exc}",
                article_path="",
                title="",
            )

        # Step 4: Check for existing article (using LLM-generated title)
        existing_path = self._km.exists(task.spec, task.category, generated.title)
        if existing_path is not None:
            db_status = self._km.get_article_status(existing_path)
            if db_status is not None and db_status != ArticleStatus.DRAFT:
                return WriteResult(
                    task_id=task.task_id,
                    agent_role=self.role,
                    status=TaskStatus.FAILED,
                    output=f"Cannot overwrite {db_status} article: {existing_path}",
                    article_path=existing_path,
                    title=generated.title,
                )

        # Step 5: Extract claims and entities (Pass 2)
        if on_status:
            await on_status("extracting claims...")

        try:
            extraction = await self._extract_claims(generated.content)
        except Exception as exc:
            logger.warning("Extraction pass failed, continuing with empty claims: %s", exc)
            extraction = ArticleExtraction()

        # Step 6: Build metadata and write
        now = datetime.now(tz=UTC)
        confidence = statistics.mean(f.confidence for f in task.research.findings)

        # Deduplicate source URLs from findings evidence
        seen_urls: set[str] = set()
        sources: list[SourceRef] = []
        for finding in task.research.findings:
            for ev in finding.evidence:
                if ev.startswith("http") and ev not in seen_urls:
                    seen_urls.add(ev)
                    sources.append(SourceRef(url=ev, trust=finding.confidence))

        claims = [ClaimRef(text=c, verified=False) for c in extraction.claims]

        article_meta = ArticleMeta(
            title=generated.title,
            spec=Spec(task.spec),
            category=task.category,
            status=ArticleStatus.DRAFT,
            confidence=round(confidence, 4),
            created_at=now,
            updated_at=now,
            sources=sources,
            claims=claims,
            entity_refs=extraction.entity_refs,
            tags=[task.spec, task.category],
        )

        if existing_path is not None:
            self._km.update_draft(existing_path, article_meta, generated.content)
            path = existing_path
        else:
            path = self._km.create_draft(article_meta, generated.content)

        # Step 7: Return WriteResult
        return WriteResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=TaskStatus.SUCCESS,
            output=f"Article written: {generated.title}",
            evidence=list(seen_urls),
            article_path=path,
            title=generated.title,
            claims=extraction.claims,
            research_gaps=task.research.gaps,
            word_count=len(generated.content.split()),
            entity_refs=extraction.entity_refs,
        )

    async def _generate_article(self, task: WriteTask) -> GeneratedArticle:
        """Pass 1: Generate article content from research findings."""
        findings_text = "\n".join(
            f"- {f.claim} (confidence: {f.confidence}, evidence: {f.evidence})" for f in task.research.findings
        )
        gaps_text = "\n".join(f"- {g}" for g in task.research.gaps) if task.research.gaps else "(none)"

        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Write a {task.article_type.value} article about: {task.query}\n\n"
                    f"Spec: {task.spec}\nCategory: {task.category}\n\n"
                    f"Research findings:\n{findings_text}\n\n"
                    f"Research gaps:\n{gaps_text}"
                ),
            },
        ]

        result: GeneratedArticle = await get_structured_output(response_model=GeneratedArticle, messages=messages)
        return result

    async def _extract_claims(self, content: str) -> ArticleExtraction:
        """Pass 2: Extract claims and entity references from generated article."""
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": content},
        ]

        result: ArticleExtraction = await get_structured_output(response_model=ArticleExtraction, messages=messages)
        return result
