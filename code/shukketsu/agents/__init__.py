"""Multi-agent system: BaseAgent, AgentFactory, and Structured Task Protocol."""

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    OrchestratorPlan,
    ResearchTask,
    SearchStrategy,
    SubTask,
    TaskStatus,
    WriteTask,
)

__all__ = [
    "AgentFactory",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "ArticleType",
    "BaseAgent",
    "EditTask",
    "OrchestratorPlan",
    "ResearchTask",
    "SearchStrategy",
    "SubTask",
    "TaskStatus",
    "WriteTask",
]
