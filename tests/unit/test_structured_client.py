"""Tests for the Instructor-based structured output client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError

# --- Simple test model ---


class SimpleResponse(BaseModel):
    """A trivial model for testing structured output."""

    name: str
    value: int


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    """Reset the client cache before each test."""
    from code.shukketsu.llm.structured import clear_clients

    clear_clients()


# --- Tests ---


class TestGetStructuredOutput:
    """Tests for the get_structured_output function."""

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_returns_validated_model(self, mock_get_client: MagicMock) -> None:
        """Should return a validated Pydantic model instance."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        expected = SimpleResponse(name="test", value=42)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=expected)
        mock_get_client.return_value = mock_client

        result = await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
        )

        assert result == expected
        assert isinstance(result, SimpleResponse)

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_default_model_vllm(self, mock_get_client: MagicMock) -> None:
        """VLLM backend should use REASONING_MODEL by default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=SimpleResponse(name="x", value=1))
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "llama-3.3-70b-instruct-awq"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_default_model_ollama(self, mock_get_client: MagicMock) -> None:
        """OLLAMA backend should use ROUTER_MODEL by default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=SimpleResponse(name="x", value=1))
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.OLLAMA,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "qwen3:4b"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_explicit_model_override(self, mock_get_client: MagicMock) -> None:
        """Passing model= should override the backend default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=SimpleResponse(name="x", value=1))
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
            model="custom-model",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_connection_error_raises_llm_unavailable(self, mock_get_client: MagicMock) -> None:
        """ConnectError should raise LLMUnavailableError, not retry."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="Cannot connect"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_api_connection_error_raises_llm_unavailable(self, mock_get_client: MagicMock) -> None:
        """APIConnectionError should raise LLMUnavailableError."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="Cannot connect"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_timeout_raises_llm_unavailable(self, mock_get_client: MagicMock) -> None:
        """TimeoutException should raise LLMUnavailableError."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=httpx.TimeoutException("Timed out"))
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="did not respond"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_retries_exhausted_raises_structured_output_error(self, mock_get_client: MagicMock) -> None:
        """InstructorRetryException should raise StructuredOutputError."""
        from instructor.core.exceptions import InstructorRetryException

        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=InstructorRetryException(
                n_attempts=3,
                messages=[],
                last_completion=None,
                total_usage=0,
            )
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(StructuredOutputError, match="Failed to get valid"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )


class TestRetryIntegration:
    """Tests for the @with_retry integration on get_structured_output."""

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_retries_on_transient_llm_unavailable(self, mock_get_client: MagicMock) -> None:
        """LLMUnavailableError should be retried once before propagating."""
        from code.shukketsu.llm.structured import get_structured_output

        expected = SimpleResponse(name="ok", value=1)
        mock_client = MagicMock()
        # First call fails with connection error, second succeeds
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[httpx.ConnectError("Connection refused"), expected]
        )
        mock_get_client.return_value = mock_client

        result = await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result == expected
        assert mock_client.chat.completions.create.call_count == 2


class TestClientCaching:
    """Tests for the client factory caching behavior."""

    @patch("code.shukketsu.llm.structured.instructor")
    def test_client_is_cached(self, mock_instructor: MagicMock) -> None:
        """Second call with same backend should reuse the cached client."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client

        mock_instructor.from_openai.return_value = MagicMock()

        client1 = _get_client(ModelBackend.VLLM)
        client2 = _get_client(ModelBackend.VLLM)

        assert client1 is client2
        assert mock_instructor.from_openai.call_count == 1

    @patch("code.shukketsu.llm.structured.instructor")
    def test_clear_clients_resets_cache(self, mock_instructor: MagicMock) -> None:
        """clear_clients should force new client creation on next call."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client, clear_clients

        mock_instructor.from_openai.return_value = MagicMock()

        _get_client(ModelBackend.VLLM)
        clear_clients()
        _get_client(ModelBackend.VLLM)

        assert mock_instructor.from_openai.call_count == 2

    @patch("code.shukketsu.llm.structured.instructor")
    def test_ollama_url_has_v1_suffix(self, mock_instructor: MagicMock) -> None:
        """Ollama client should be created with /v1 appended to base URL."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client

        mock_instructor.from_openai.return_value = MagicMock()

        _get_client(ModelBackend.OLLAMA)

        # Inspect the AsyncOpenAI constructor call
        from_openai_call = mock_instructor.from_openai.call_args
        openai_client = from_openai_call.args[0]
        assert str(openai_client.base_url).rstrip("/").endswith("/v1")
