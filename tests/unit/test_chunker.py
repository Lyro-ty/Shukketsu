"""Tests for the recursive text chunker."""

from code.shukketsu.ingest.chunker import Chunk, chunk_text


class TestChunkText:
    """Tests for chunk_text()."""

    def test_empty_input_returns_empty_list(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert chunk_text("   \n\n  ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "The hit cap for combat rogues is 9%."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_splits_on_h2_headers(self) -> None:
        section_a = "A " * 300  # ~300 tokens, well over min
        section_b = "B " * 300
        text = f"## Section One\n{section_a}\n## Section Two\n{section_b}"
        chunks = chunk_text(text, max_tokens=400)
        assert len(chunks) >= 2
        assert "Section One" in chunks[0].content
        assert "Section Two" in chunks[-1].content

    def test_splits_on_paragraphs(self) -> None:
        para_a = "Alpha. " * 80  # ~80 tokens each
        para_b = "Bravo. " * 80
        para_c = "Charlie. " * 80
        para_d = "Delta. " * 80
        para_e = "Echo. " * 80
        text = f"{para_a}\n\n{para_b}\n\n{para_c}\n\n{para_d}\n\n{para_e}"
        chunks = chunk_text(text, max_tokens=200)
        assert len(chunks) >= 2

    def test_splits_on_sentences(self) -> None:
        # Single paragraph that's too long, must split on sentences
        text = "Sentence one. " * 200  # ~400 tokens as one paragraph
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) >= 2

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        text = "Word " * 2000  # ~2000 tokens
        chunks = chunk_text(text, max_tokens=400)
        for chunk in chunks:
            assert chunk.token_estimate <= 400 + 10  # small tolerance for boundary

    def test_chunk_indices_are_sequential(self) -> None:
        text = "Word " * 2000
        chunks = chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_has_correct_fields(self) -> None:
        text = "Hello world, this is a test."
        chunks = chunk_text(text)
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.char_count == len(text)
        assert chunk.token_estimate == len(text) // 4
