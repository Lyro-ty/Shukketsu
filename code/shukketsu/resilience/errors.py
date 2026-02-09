"""Error taxonomy for Shukketsu failure modes."""

from enum import StrEnum


class FailureMode(StrEnum):
    # LLM failures
    LLM_TIMEOUT = "llm_timeout"
    LLM_LOOP = "llm_loop"
    LLM_MALFORMED_OUTPUT = "llm_malformed"
    LLM_REFUSAL = "llm_refusal"

    # Tool failures
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"

    # Network failures
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"

    # Data failures
    DB_ERROR = "db_error"
    EMBEDDING_ERROR = "embedding_error"

    # System failures
    OOM = "out_of_memory"
    MODEL_UNAVAILABLE = "model_unavailable"


class ShukketsuError(Exception):
    """Base exception for all Shukketsu errors."""

    def __init__(self, message: str, failure_mode: FailureMode):
        super().__init__(message)
        self.failure_mode = failure_mode


class StructuredOutputError(ShukketsuError):
    """Raised when structured output cannot be obtained after all retries."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.LLM_MALFORMED_OUTPUT)


class AgentLoopError(ShukketsuError):
    """Raised when an agent is detected to be stuck in a loop."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.LLM_LOOP)


class LLMUnavailableError(ShukketsuError):
    """Raised when the LLM server cannot be reached."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.MODEL_UNAVAILABLE)
