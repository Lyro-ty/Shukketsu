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
    ACCESS_DENIED = "access_denied"

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


class DatabaseError(ShukketsuError):
    """Raised when a database operation fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.DB_ERROR)


class ToolNotFoundError(ShukketsuError):
    """Raised when an agent requests a tool that is not registered."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.TOOL_NOT_FOUND)


class ToolExecutionError(ShukketsuError):
    """Raised when a tool execution fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.TOOL_EXECUTION_ERROR)


class EmbeddingError(ShukketsuError):
    """Raised when the embedding model fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.EMBEDDING_ERROR)


class ScrapingError(ShukketsuError):
    """Raised when a web page cannot be fetched."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.HTTP_ERROR)


class RobotsDisallowedError(ShukketsuError):
    """Raised when robots.txt disallows access to a URL."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.ACCESS_DENIED)


class BraveSearchError(ShukketsuError):
    """Raised when the Brave Search API call fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.TOOL_EXECUTION_ERROR)


class EntityExtractionError(ShukketsuError):
    """Raised when entity extraction from a chunk fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.LLM_MALFORMED_OUTPUT)


class CircuitOpenError(ShukketsuError):
    """Raised when a circuit breaker is open and rejecting requests."""

    def __init__(self, breaker_name: str):
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open — service unavailable",
            FailureMode.MODEL_UNAVAILABLE,
        )
        self.breaker_name = breaker_name


class RerankerError(ShukketsuError):
    """Raised when the reranker LLM call fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.MODEL_UNAVAILABLE)


class GraphTraversalError(ShukketsuError):
    """Raised when a knowledge graph query fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.DB_ERROR)
