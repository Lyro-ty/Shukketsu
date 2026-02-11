"""Multi-agent system: BaseAgent, AgentFactory, and Structured Task Protocol."""

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.editor import Editor
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.researcher import Researcher
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    Finding,
    OrchestratorPlan,
    ResearchResult,
    ResearchTask,
    SearchStrategy,
    SubTask,
    TaskStatus,
    VerificationStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.agents.writer import Writer

__all__ = [
    "AgentFactory",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "ArticleType",
    "BaseAgent",
    "ClaimJudgment",
    "ClaimVerification",
    "EditResult",
    "EditTask",
    "Editor",
    "Finding",
    "OrchestratorPlan",
    "Researcher",
    "ResearchResult",
    "ResearchTask",
    "SearchStrategy",
    "SubTask",
    "TaskStatus",
    "VerificationStatus",
    "Writer",
    "WriteResult",
    "WriteTask",
]
