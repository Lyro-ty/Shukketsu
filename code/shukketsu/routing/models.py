"""Routing data models for multi-model query classification."""

from enum import StrEnum

from pydantic import BaseModel


class TaskComplexity(StrEnum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TaskCategory(StrEnum):
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    WRITING = "writing"
    CONVERSATION = "conversation"


class RoutingDecision(BaseModel):
    complexity: TaskComplexity
    category: TaskCategory
    needs_tools: bool
    suggested_agent: str
    direct_answer: str | None = None
