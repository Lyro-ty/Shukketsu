"""Instructor-wrapped LLM clients for structured output.

Provides a generic function to get validated Pydantic models from
Ollama backends (Llama 70B reasoning, Qwen 4B routing). Uses
Instructor in JSON mode with automatic retry on validation failure.
"""

import logging
from enum import StrEnum

import httpx
import instructor
from langfuse import get_client, observe
from openai import APIConnectionError, AsyncOpenAI
from pydantic import BaseModel

from code.shukketsu import config
from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError
from code.shukketsu.resilience.retry import with_retry

logger = logging.getLogger(__name__)


class ModelBackend(StrEnum):
    """Available LLM backends for structured output."""

    REASONING = "reasoning"
    ROUTER = "router"


_clients: dict[ModelBackend, instructor.AsyncInstructor] = {}

_OLLAMA_OPENAI_URL = f"{config.OLLAMA_BASE_URL}/v1"


def _get_client(backend: ModelBackend) -> instructor.AsyncInstructor:
    """Return a cached Instructor client for the given backend.

    Creates the client on first call, then caches it. All backends
    use Ollama's OpenAI-compatible API with Mode.JSON.
    """
    if backend not in _clients:
        openai_client = AsyncOpenAI(
            base_url=_OLLAMA_OPENAI_URL,
            api_key="not-needed",
            timeout=httpx.Timeout(timeout=config.LLM_TIMEOUT_SECONDS, connect=10.0),
            max_retries=0,  # Our with_retry decorator handles retries; avoid 15-min waits
        )
        _clients[backend] = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )
        logger.info("Created Instructor client for %s backend", backend.value)

    return _clients[backend]


def clear_clients() -> None:
    """Clear the client cache. Used by tests to reset state."""
    _clients.clear()


@observe(as_type="generation")
@with_retry(max_attempts=2, base_delay=1.0, retryable=(LLMUnavailableError,))
async def get_structured_output[T: BaseModel](
    response_model: type[T],
    messages: list[dict[str, str]],
    *,
    backend: ModelBackend = ModelBackend.REASONING,
    model: str | None = None,
    temperature: float = config.STRUCTURED_TEMPERATURE,
    max_tokens: int = config.STRUCTURED_MAX_TOKENS,
    max_retries: int = config.STRUCTURED_MAX_RETRIES,
) -> T:
    """Get validated structured output from an LLM.

    Args:
        response_model: Pydantic model class to validate the response against.
        messages: Chat messages to send to the LLM.
        backend: Which LLM backend to use (reasoning or router).
        model: Model name override. Defaults to REASONING_MODEL for reasoning,
            ROUTER_MODEL for router.
        temperature: Sampling temperature (default 0.1 for consistency).
        max_tokens: Maximum tokens in the response (default 4096).
        max_retries: Number of validation retry attempts (default 3).

    Returns:
        A validated instance of response_model.

    Raises:
        LLMUnavailableError: If the backend server is unreachable or times out.
        StructuredOutputError: If all validation retries are exhausted.
    """
    if model is None:
        model = config.REASONING_MODEL if backend == ModelBackend.REASONING else config.ROUTER_MODEL

    langfuse = get_client()
    langfuse.update_current_generation(
        model=model,
        model_parameters={"temperature": str(temperature), "max_tokens": str(max_tokens)},
    )

    client = _get_client(backend)

    try:
        return await client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    except (httpx.ConnectError, APIConnectionError) as exc:
        raise LLMUnavailableError(f"Cannot connect to {backend.value} server. Is it running?") from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(
            f"{backend.value} server did not respond within {config.LLM_TIMEOUT_SECONDS}s."
        ) from exc
    except instructor.core.exceptions.InstructorRetryException as exc:
        raise StructuredOutputError(
            f"Failed to get valid {response_model.__name__} after {max_retries} retries."
        ) from exc
