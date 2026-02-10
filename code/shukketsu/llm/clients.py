"""LLM client for Ollama streaming chat completions via OpenAI-compatible API."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import APIConnectionError, AsyncOpenAI

from code.shukketsu import config
from code.shukketsu.resilience.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

_OLLAMA_OPENAI_URL = f"{config.OLLAMA_BASE_URL}/v1"

_client = AsyncOpenAI(
    base_url=_OLLAMA_OPENAI_URL,
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
    """Stream chat completion tokens from Ollama.

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
    except (httpx.ConnectError, APIConnectionError) as exc:
        raise LLMUnavailableError(
            f"Cannot connect to Ollama at {_OLLAMA_OPENAI_URL}. Is Ollama running?"
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(
            f"Ollama at {_OLLAMA_OPENAI_URL} did not respond within "
            f"{config.LLM_TIMEOUT_SECONDS} seconds. The model may be loading."
        ) from exc

    try:
        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except httpx.ReadError as exc:
        raise LLMUnavailableError(f"Connection to Ollama lost during response: {exc}") from exc
    finally:
        await stream.close()
