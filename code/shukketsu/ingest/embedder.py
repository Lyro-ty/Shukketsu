"""Text embedder using nomic-embed-text via Ollama's OpenAI-compatible API."""

import logging

import httpx
from openai import APIConnectionError, AsyncOpenAI

from code.shukketsu import config
from code.shukketsu.resilience.errors import EmbeddingError

logger = logging.getLogger(__name__)

_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def get_embedder() -> "Embedder":
    """Create an Embedder with the default Ollama client."""
    client = AsyncOpenAI(
        base_url=config.OLLAMA_BASE_URL + "/v1",
        api_key="not-needed",
        timeout=httpx.Timeout(timeout=config.LLM_TIMEOUT_SECONDS, connect=10.0),
    )
    return Embedder(client=client)


class Embedder:
    """Batch text embedder using nomic-embed-text via Ollama.

    Uses the OpenAI-compatible /v1/embeddings endpoint with batch input
    for efficient embedding of multiple texts in a single API call.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call.

        Adds 'search_document: ' prefix required by nomic-embed-text.

        Args:
            texts: List of texts to embed.

        Returns:
            List of 768-dim float vectors, one per input text.

        Raises:
            EmbeddingError: If the embedding model is unreachable.
        """
        prefixed = [_DOCUMENT_PREFIX + t for t in texts]
        return await self._call(prefixed)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Adds 'search_query: ' prefix required by nomic-embed-text.

        Args:
            query: The search query to embed.

        Returns:
            A 768-dim float vector.

        Raises:
            EmbeddingError: If the embedding model is unreachable.
        """
        result = await self._call([_QUERY_PREFIX + query])
        return result[0]

    async def _call(self, texts: list[str]) -> list[list[float]]:
        """Send texts to the embedding API and return vectors."""
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=config.EMBEDDING_MODEL,
            )
        except (httpx.ConnectError, APIConnectionError, httpx.TimeoutException) as exc:
            raise EmbeddingError(f"Cannot connect to embedding model at {config.OLLAMA_BASE_URL}: {exc}") from exc
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

        return [item.embedding for item in response.data]
