"""Multi-agent system: BaseAgent, AgentFactory, and Structured Task Protocol."""

from code.shukketsu.agents.analyst import Analyst
from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.editor import Editor
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.orchestrator import Orchestrator
from code.shukketsu.agents.researcher import Researcher
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    AnalysisResult,
    AnalysisTask,
    ArticleType,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
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
    "Analyst",
    "AnalysisResult",
    "AnalysisTask",
    "ArticleType",
    "BaseAgent",
    "ClaimJudgment",
    "ClaimVerification",
    "EditResult",
    "EditTask",
    "Editor",
    "Finding",
    "Orchestrator",
    "OrchestratorPlan",
    "OrchestratorResult",
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
