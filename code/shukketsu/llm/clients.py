"""LLM client for vLLM streaming chat completions."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from code.shukketsu import config
from code.shukketsu.resilience.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    base_url=config.VLLM_BASE_URL,
    api_key="not-needed",
    timeout=httpx.Timeout(timeout=config.LLM_TIMEOUT_SECONDS, connect=10.0),
)


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str = config.REASONING_MODEL,
    temperature: float = config.CHAT_TEMPERATURE,
    max_tokens: int = config.CHAT_MAX_TOKENS,
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens from vLLM.

    Yields token strings one at a time. Raises LLMUnavailableError
    if the server is unreachable or the connection drops mid-stream.
    """
    try:
        stream: Any = await _client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except httpx.ConnectError:
        raise LLMUnavailableError(
            f"Cannot connect to LLM server at {config.VLLM_BASE_URL}. Is vLLM running on port 8000?"
        )
    except httpx.TimeoutException:
        raise LLMUnavailableError(
            f"LLM server at {config.VLLM_BASE_URL} did not respond within "
            f"{config.LLM_TIMEOUT_SECONDS} seconds. It may be loading the model."
        )

    try:
        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except httpx.ReadError as exc:
        raise LLMUnavailableError(f"Connection to LLM server lost during response: {exc}")
    finally:
        await stream.close()
