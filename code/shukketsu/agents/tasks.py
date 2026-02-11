"""Structured Task Protocol models for multi-agent communication.

Defines the typed Task/Result contracts that all agents use. Tasks describe
what a specialist should do; Results capture what they produced. All
inter-agent communication flows through these models.
"""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(StrEnum):
    """Specialist agent roles in the multi-agent system."""

    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(StrEnum):
    """Outcome status of an agent task execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentTask(BaseModel):
    """Base task that any agent can execute.

    All specialist task types inherit from this. The task_id and trace_id
    enable end-to-end tracing through Langfuse.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Base result returned by any agent after executing a task.

    Specialist result types (ResearchResult, WriteResult, etc.) inherit
    from this and add role-specific fields.
    """

    task_id: str
    agent_role: AgentRole
    status: TaskStatus
    output: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchStrategy(StrEnum):
    """Search strategy for the Researcher agent."""

    HYBRID = "hybrid"
    GRAPH = "graph"
    WEB = "web"
    AUTO = "auto"


class ArticleType(StrEnum):
    """Article types the Writer agent can produce."""

    GUIDE = "guide"
    REFERENCE = "reference"
    ANALYSIS = "analysis"


class ResearchTask(AgentTask):
    """Task for the Researcher agent.

    Optionally specifies a search strategy and max sources to consider.
    """

    search_strategy: SearchStrategy | None = None
    max_sources: int = 10


class WriteTask(AgentTask):
    """Task for the Writer agent.

    Requires research results and an article type.
    """

    research: AgentResult
    article_type: ArticleType


class EditTask(AgentTask):
    """Task for the Editor agent.

    Points to a draft article and lists claims to verify.
    """

    article_path: str
    claims: list[str]


class SubTask(BaseModel):
    """A single sub-task in an Orchestrator plan."""

    agent_role: AgentRole
    description: str
    depends_on: list[int] = Field(default_factory=list)
    task_params: dict[str, Any] = Field(default_factory=dict)


class OrchestratorPlan(BaseModel):
    """Execution plan produced by the Orchestrator.

    Contains an ordered list of sub-tasks with dependency information.
    The Orchestrator walks the dependency graph, executing tasks whose
    dependencies are satisfied, grouping independent tasks for concurrency.
    """

    reasoning: str
    subtasks: list[SubTask]
    can_answer_directly: bool = False
    direct_answer: str | None = None
