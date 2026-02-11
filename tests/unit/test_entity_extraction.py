"""Tests for LLM-powered entity extraction."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.rag.entities import (
    EXTRACTION_SYSTEM_PROMPT,
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
    build_extraction_messages,
    extract_entities_from_chunk,
)
from code.shukketsu.resilience.errors import EntityExtractionError


class TestExtractionPrompt:
    """Tests for the extraction prompt and message builder."""

    def test_system_prompt_exists(self) -> None:
        assert len(EXTRACTION_SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_entity_types(self) -> None:
        for et in ["item", "spell", "talent", "boss", "instance", "stat"]:
            assert et in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_relation_types(self) -> None:
        for rt in ["drops_from", "has_stat", "available_in"]:
            assert rt in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_build_messages_structure(self) -> None:
        messages = build_extraction_messages("Some WoW text about rogues.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Some WoW text about rogues." in messages[1]["content"]

    def test_build_messages_includes_system_prompt(self) -> None:
        messages = build_extraction_messages("text")
        assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT


class TestExtractEntities:
    """Tests for extract_entities_from_chunk()."""

    async def test_returns_chunk_extraction(self) -> None:
        mock_result = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Dragonspine Trophy", entity_type=EntityType.ITEM),
                ExtractedEntity(name="Gruul", entity_type=EntityType.BOSS),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Dragonspine Trophy",
                    target="Gruul",
                    relation_type=RelationType.DROPS_FROM,
                ),
            ],
        )

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await extract_entities_from_chunk("DST drops from Gruul.")

        assert isinstance(result, ChunkExtraction)
        assert len(result.entities) == 2
        assert len(result.relationships) == 1

    async def test_empty_extraction(self) -> None:
        mock_result = ChunkExtraction(entities=[], relationships=[])

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await extract_entities_from_chunk("No entities here.")

        assert result.entities == []
        assert result.relationships == []

    async def test_wraps_llm_errors(self) -> None:
        """LLM failures should be wrapped in EntityExtractionError."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            side_effect=StructuredOutputError("Failed after retries"),
        ):
            with pytest.raises(EntityExtractionError, match="Failed"):
                await extract_entities_from_chunk("text")

    async def test_passes_correct_model(self) -> None:
        mock_result = ChunkExtraction(entities=[], relationships=[])

        with patch(
            "code.shukketsu.rag.entities.get_structured_output",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await extract_entities_from_chunk("text")

        call_kwargs = mock_fn.call_args
        assert call_kwargs[0][0] is ChunkExtraction  # response_model
        assert call_kwargs[1]["backend"].value == "reasoning"
