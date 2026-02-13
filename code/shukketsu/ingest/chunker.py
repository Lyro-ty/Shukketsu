"""Recursive text chunker for the ingest pipeline.

Splits text into ~400-token chunks using semantic boundaries (headers,
paragraphs, sentences) with recursive fallback to smaller separators.
"""

import re
from dataclasses import dataclass

from code.shukketsu import config

# Separators tried in priority order. Each is a regex pattern.
_SEPARATORS = [
    r"\n## ",  # Markdown H2 headers
    r"\n### ",  # Markdown H3 headers
    r"\n\n",  # Paragraph breaks
    r"\n",  # Line breaks
    r"\. ",  # Sentence endings
    r" ",  # Word boundaries
]

# Header separators always split regardless of token count, because they
# represent meaningful document structure boundaries.
_ALWAYS_SPLIT = {r"\n## ", r"\n### "}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length (~4 chars/token)."""
    return len(text) // 4


@dataclass(frozen=True)
class Chunk:
    """A text chunk produced by the chunker."""

    content: str
    chunk_index: int
    char_count: int
    token_estimate: int


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text on a separator, keeping the separator with the next segment."""
    if separator == r" " or separator == r"\n":
        parts = text.split(separator.replace("\\n", "\n"))
    elif separator == r"\n\n":
        parts = text.split("\n\n")
    elif separator == r"\. ":
        # Split on ". " but keep the period with the preceding text
        raw = re.split(r"(?<=\.) ", text)
        parts = raw
    else:
        # Header separators: split and keep separator with following text
        sep_literal = separator.replace("\\n", "\n")
        raw = text.split(sep_literal)
        parts = [raw[0]] + [sep_literal.lstrip("\n") + part for part in raw[1:]]
    return [p for p in parts if p.strip()]


def _joiner_for_separator(separator: str) -> str:
    """Return the join string appropriate for a given separator."""
    if separator == r"\n\n":
        return "\n\n"
    if separator in (r"\n", r"\n## ", r"\n### "):
        return "\n"
    return " "


def _recursive_split(text: str, max_tokens: int, sep_index: int = 0) -> list[str]:
    """Recursively split text until all pieces are under max_tokens.

    Header separators (H2, H3) always split regardless of token count
    because they mark meaningful document structure boundaries. Other
    separators only trigger when text exceeds max_tokens.
    """
    under_limit = _estimate_tokens(text) <= max_tokens

    # Try header-level separators even when under the token limit
    if under_limit:
        for i, sep in enumerate(_SEPARATORS):
            if sep not in _ALWAYS_SPLIT:
                continue
            if i < sep_index:
                continue
            parts = _split_on_separator(text, sep)
            if len(parts) > 1:
                # Recurse on each section (may need further splitting)
                result: list[str] = []
                for part in parts:
                    result.extend(_recursive_split(part, max_tokens, i + 1))
                return result
        return [text]

    if sep_index >= len(_SEPARATORS):
        # Last resort: hard-split by characters
        max_chars = max_tokens * 4
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    parts = _split_on_separator(text, _SEPARATORS[sep_index])

    # If the separator didn't help (only 1 part), try the next separator
    if len(parts) <= 1:
        return _recursive_split(text, max_tokens, sep_index + 1)

    # Greedily merge parts into chunks up to max_tokens, recurse on oversized
    joiner = _joiner_for_separator(_SEPARATORS[sep_index])
    merged: list[str] = []
    current = ""
    for part in parts:
        candidate = current + joiner + part if current else part
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = part
    if current:
        merged.append(current)

    # Recurse on any pieces still over max_tokens
    result = []
    for piece in merged:
        if _estimate_tokens(piece) > max_tokens:
            result.extend(_recursive_split(piece, max_tokens, sep_index + 1))
        else:
            result.append(piece)

    return result


def chunk_text(
    text: str,
    *,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
    min_tokens: int = config.CHUNK_MIN_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split text into chunks using recursive separator-based splitting.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk (~4 chars/token).
        min_tokens: Minimum tokens; smaller chunks merge with neighbor.
        overlap_tokens: Tokens of overlap prepended from previous chunk.

    Returns:
        List of Chunk objects with sequential chunk_index values.
    """
    stripped = text.strip()
    if not stripped:
        return []

    raw_pieces = _recursive_split(stripped, max_tokens)

    # Filter empty pieces
    pieces = [p.strip() for p in raw_pieces if p.strip()]

    # Merge small chunks with their next neighbor
    merged: list[str] = []
    i = 0
    while i < len(pieces):
        current = pieces[i]
        while _estimate_tokens(current) < min_tokens and i + 1 < len(pieces):
            i += 1
            current = current + "\n\n" + pieces[i]
        merged.append(current)
        i += 1

    # Add overlap from previous chunk
    overlap_chars = overlap_tokens * 4
    overlapped: list[str] = []
    for i, content in enumerate(merged):
        if i > 0 and overlap_chars > 0:
            prev = merged[i - 1]
            overlap_text = prev[-overlap_chars:]
            # Find a clean word boundary for the overlap
            space_idx = overlap_text.find(" ")
            if space_idx >= 0:
                overlap_text = overlap_text[space_idx + 1 :]
            content = overlap_text + " " + content
        overlapped.append(content)

    # Build Chunk objects with sequential indices
    chunks: list[Chunk] = []
    for i, content in enumerate(overlapped):
        content = content.strip()
        if not content:
            continue
        chunks.append(
            Chunk(
                content=content,
                chunk_index=i,
                char_count=len(content),
                token_estimate=_estimate_tokens(content),
            )
        )

    return chunks
