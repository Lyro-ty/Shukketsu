"""Pydantic response models for LLM structured output.

These schemas define the contract between the LLM and the agent loop.
Instructor validates LLM responses against these models and retries
on validation failure.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class ActionType(StrEnum):
    """Actions an agent can take in a ReAct step."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class ToolCall(BaseModel):
    """A tool invocation requested by the agent."""

    thought: str
    tool_name: str
    tool_input: dict[str, Any]


class AgentStep(BaseModel):
    """A single step in the agent's ReAct loop.

    The LLM returns one of these per iteration. Either it calls a tool
    (action=tool_call with tool_call populated) or gives a final answer
    (action=final_answer with answer populated).
    """

    reasoning: str
    action: ActionType
    tool_call: ToolCall | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def _check_action_fields(self) -> "AgentStep":
        if self.action == ActionType.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call required when action is tool_call")
        if self.action == ActionType.FINAL_ANSWER and self.answer is None:
            raise ValueError("answer required when action is final_answer")
        return self


class ReflectionResult(BaseModel):
    """Result of a reflection pass on a research answer."""

    supported: bool
    issues: list[str] = []
    revised_answer: str = ""
