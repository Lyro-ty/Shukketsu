"""Pydantic models for the memory subsystem."""

from pydantic import BaseModel, Field


class MemoryExtraction(BaseModel):
    """LLM-extracted facts from a conversation turn.

    Used as the structured output schema for Qwen 4B extraction.
    """

    summary: str
    key_facts: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    """A recalled session memory with composite relevance score."""

    id: int
    query: str
    answer_summary: str
    key_facts: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    retrieval_quality: float
    created_at: str
    score: float = 0.0
