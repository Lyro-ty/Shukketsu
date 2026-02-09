"""Tests for the Ollama embedder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from code.shukketsu.ingest.embedder import Embedder
from code.shukketsu.resilience.errors import EmbeddingError

EMBEDDING_DIM = 768


def _mock_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    return [0.1] * dim


class TestEmbedTexts:
    """Tests for Embedder.embed_texts()."""

    async def test_adds_search_document_prefix(self) -> None:
        mock_client = MagicMock()
        embedding_obj = MagicMock()
        embedding_obj.embedding = _mock_embedding()
        mock_response = MagicMock()
        mock_response.data = [embedding_obj]
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = Embedder(client=mock_client)
        await embedder.embed_texts(["hello world"])

        call_args = mock_client.embeddings.create.call_args
        input_arg = call_args.kwargs.get("input") or call_args[1].get("input")
        assert input_arg == ["search_document: hello world"]

    async def test_batch_sends_all_texts(self) -> None:
        mock_client = MagicMock()
        emb1 = MagicMock()
        emb1.embedding = _mock_embedding()
        emb2 = MagicMock()
        emb2.embedding = _mock_embedding()
        emb3 = MagicMock()
        emb3.embedding = _mock_embedding()
        mock_response = MagicMock()
        mock_response.data = [emb1, emb2, emb3]
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = Embedder(client=mock_client)
        result = await embedder.embed_texts(["a", "b", "c"])

        # Single API call with all 3 texts
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        input_arg = call_args.kwargs.get("input") or call_args[1].get("input")
        assert len(input_arg) == 3
        assert len(result) == 3

    async def test_connection_error_raises_embedding_error(self) -> None:
        mock_client = MagicMock()
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("Connection refused"))

        embedder = Embedder(client=mock_client)
        with pytest.raises(EmbeddingError, match="Connection refused"):
            await embedder.embed_texts(["hello"])


class TestEmbedQuery:
    """Tests for Embedder.embed_query()."""

    async def test_adds_search_query_prefix(self) -> None:
        mock_client = MagicMock()
        embedding_obj = MagicMock()
        embedding_obj.embedding = _mock_embedding()
        mock_response = MagicMock()
        mock_response.data = [embedding_obj]
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = Embedder(client=mock_client)
        await embedder.embed_query("hit cap")

        call_args = mock_client.embeddings.create.call_args
        input_arg = call_args.kwargs.get("input") or call_args[1].get("input")
        assert input_arg == ["search_query: hit cap"]
