"""Tests for Writer agent."""

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.agents.tasks import (
    AgentRole,
    AgentTask,
    ArticleType,
    Finding,
    ResearchResult,
    TaskStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.agents.writer import Writer
from code.shukketsu.knowledge.manager import KnowledgeManager
from code.shukketsu.tools.registry import ToolRegistry


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a temporary knowledge directory."""
    d = tmp_path / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def km(test_db: sqlite3.Connection, knowledge_dir: Path) -> KnowledgeManager:
    """Create a KnowledgeManager backed by test DB and tmp knowledge dir."""
    return KnowledgeManager(test_db, knowledge_dir)


@pytest.fixture
def writer(km: KnowledgeManager) -> Writer:
    """Create a Writer agent with a test KnowledgeManager."""
    return Writer(
        tool_registry=ToolRegistry(),
        role=AgentRole.WRITER,
        knowledge_manager=km,
    )


def _research_result(**overrides: object) -> ResearchResult:
    """Build a ResearchResult with sensible defaults."""
    defaults: dict[str, object] = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research complete",
        findings=[
            Finding(claim="Hit cap is 142 rating", confidence=0.9, evidence=["https://example.com/guide"]),
            Finding(claim="DST is BiS trinket", confidence=0.8, evidence=["chunk:42"]),
        ],
        gaps=["proc rate details"],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_task(**overrides: object) -> WriteTask:
    """Build a WriteTask with sensible defaults."""
    defaults: dict[str, object] = dict(
        query="Write about combat trinkets",
        research=_research_result(),
        article_type=ArticleType.GUIDE,
        spec="combat",
        category="gear",
    )
    defaults.update(overrides)
    return WriteTask(**defaults)


def _mock_structured_output(response_model: type, messages: list, **kwargs: object) -> object:
    """Return a mock structured response based on the model type."""
    if response_model.__name__ == "GeneratedArticle":
        return response_model(
            title="Combat Trinkets Guide",
            content=(
                "# Combat Trinkets Guide\n\n## Overview\n\nTrinkets are important.\n\n"
                "## BiS List\n\n**Dragonspine Trophy** is the best."
            ),
        )
    if response_model.__name__ == "ArticleExtraction":
        return response_model(
            claims=["Hit cap is 142 rating", "DST is BiS trinket"],
            entity_refs=["Dragonspine Trophy", "Hit Rating"],
        )
    raise ValueError(f"Unexpected model: {response_model.__name__}")


class TestWriterValidation:
    """Validation and guard-clause tests."""

    async def test_execute_requires_role(self, km: KnowledgeManager) -> None:
        """Writer without a role raises ValueError."""
        writer = Writer(tool_registry=ToolRegistry(), role=None, knowledge_manager=km)
        task = _write_task()
        with pytest.raises(ValueError, match="role"):
            await writer.execute(task)

    async def test_execute_requires_write_task(self, writer: Writer) -> None:
        """Non-WriteTask input returns FAILED AgentResult."""
        task = AgentTask(query="Not a write task")
        result = await writer.execute(task)
        assert result.status == TaskStatus.FAILED

    async def test_execute_empty_findings_returns_failed(self, writer: Writer) -> None:
        """WriteTask with empty findings returns FAILED WriteResult."""
        task = _write_task(research=_research_result(findings=[]))
        result = await writer.execute(task)
        assert result.status == TaskStatus.FAILED
        assert "No findings" in result.output


class TestWriterGeneration:
    """Article generation and metadata tests."""

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_produces_article_file(
        self, mock_llm: AsyncMock, writer: Writer, knowledge_dir: Path
    ) -> None:
        """Successful execution creates a .md file on disk."""
        task = _write_task()
        result = await writer.execute(task)
        assert result.status == TaskStatus.SUCCESS
        full_path = knowledge_dir / result.article_path
        assert full_path.exists()

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_returns_write_result(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """Successful execution returns a WriteResult with article path and title."""
        task = _write_task()
        result = await writer.execute(task)
        assert isinstance(result, WriteResult)
        assert result.article_path.endswith(".md")
        assert result.title == "Combat Trinkets Guide"

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_article_has_frontmatter(
        self, mock_llm: AsyncMock, writer: Writer, knowledge_dir: Path
    ) -> None:
        """Written article contains YAML frontmatter with correct metadata."""
        import frontmatter

        task = _write_task()
        result = await writer.execute(task)
        raw = (knowledge_dir / result.article_path).read_text()
        post = frontmatter.loads(raw)
        assert post.metadata["title"] == "Combat Trinkets Guide"
        assert post.metadata["spec"] == "combat"

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_claims_extracted(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """WriteResult contains claims from the extraction pass."""
        result = await writer.execute(_write_task())
        assert len(result.claims) == 2
        assert "Hit cap is 142 rating" in result.claims

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_entity_refs_extracted(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """WriteResult contains entity references from extraction pass."""
        result = await writer.execute(_write_task())
        assert "Dragonspine Trophy" in result.entity_refs

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_word_count_accurate(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """WriteResult word_count reflects the generated content."""
        result = await writer.execute(_write_task())
        assert result.word_count > 0

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_gaps_carried_through(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """Research gaps are propagated to the WriteResult."""
        result = await writer.execute(_write_task())
        assert result.research_gaps == ["proc rate details"]

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_confidence_is_average(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """Article confidence is the mean of finding confidences."""
        result = await writer.execute(_write_task())
        # Findings have confidence 0.9 and 0.8, average is 0.85
        meta, _ = writer._km.read_article(result.article_path)
        assert meta.confidence == pytest.approx(0.85)

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_uses_task_spec_category(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """Article path reflects the task's spec and category."""
        task = _write_task(spec="assassination", category="rotation")
        result = await writer.execute(task)
        assert result.article_path.startswith("assassination/rotation/")


class TestWriterExistingArticle:
    """Tests for overwriting drafts and refusing non-draft articles."""

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_overwrites_draft(self, mock_llm: AsyncMock, writer: Writer, knowledge_dir: Path) -> None:
        """Writing the same title twice overwrites the existing draft."""
        task = _write_task()
        await writer.execute(task)  # First write
        result = await writer.execute(task)  # Should overwrite draft
        assert result.status == TaskStatus.SUCCESS

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_refuses_published(
        self, mock_llm: AsyncMock, writer: Writer, test_db: sqlite3.Connection
    ) -> None:
        """Cannot overwrite a published article."""
        task = _write_task()
        result1 = await writer.execute(task)
        test_db.execute("UPDATE articles SET status = 'published' WHERE path = ?", (result1.article_path,))
        test_db.commit()
        result2 = await writer.execute(task)
        assert result2.status == TaskStatus.FAILED

    @patch(
        "code.shukketsu.agents.writer.get_structured_output",
        new_callable=AsyncMock,
        side_effect=_mock_structured_output,
    )
    async def test_execute_refuses_review(
        self, mock_llm: AsyncMock, writer: Writer, test_db: sqlite3.Connection
    ) -> None:
        """Cannot overwrite an article in review status."""
        task = _write_task()
        result1 = await writer.execute(task)
        test_db.execute("UPDATE articles SET status = 'review' WHERE path = ?", (result1.article_path,))
        test_db.commit()
        result2 = await writer.execute(task)
        assert result2.status == TaskStatus.FAILED


class TestWriterErrors:
    """Error handling: generation failures, extraction failures."""

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock)
    async def test_execute_generation_failure_returns_failed(self, mock_llm: AsyncMock, writer: Writer) -> None:
        """StructuredOutputError during generation returns FAILED."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        mock_llm.side_effect = StructuredOutputError("LLM failed")
        result = await writer.execute(_write_task())
        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock)
    async def test_execute_extraction_failure_still_writes(
        self, mock_llm: AsyncMock, writer: Writer, knowledge_dir: Path
    ) -> None:
        """Extraction failure still produces an article with empty claims."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        call_count = 0

        async def _side_effect(response_model: type, messages: list, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Generation pass succeeds
                return response_model(
                    title="Test Article",
                    content="# Test\n\nContent here.",
                )
            # Extraction pass fails
            raise StructuredOutputError("Extraction failed")

        mock_llm.side_effect = _side_effect
        result = await writer.execute(_write_task())
        assert result.status == TaskStatus.SUCCESS
        assert result.claims == []
        assert (knowledge_dir / result.article_path).exists()


class TestWriterFactory:
    """Factory integration tests."""

    def test_factory_creates_writer(self, km: KnowledgeManager) -> None:
        """AgentFactory creates a Writer instance for WRITER role."""
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.WRITER, knowledge_manager=km)
        assert isinstance(agent, Writer)

    def test_factory_writer_requires_knowledge_manager(self) -> None:
        """Writer via factory without knowledge_manager raises TypeError."""
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        with pytest.raises(TypeError):
            factory.create(AgentRole.WRITER)
