"""Reciprocal Rank Fusion and FTS5 query utilities."""

RRF_K = 60


def compute_rrf(ranked_lists: list[dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each dict maps document_id to 0-based rank (lower = better).
        k: Smoothing constant (default 60). Prevents top items from dominating.

    Returns:
        Dict mapping document_id to RRF score (higher = better).
    """
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def escape_fts_query(query: str) -> str:
    """Escape a raw query string for safe FTS5 MATCH.

    Wraps each whitespace-delimited token in double quotes to treat
    them as literals, preventing FTS5 syntax injection.

    Args:
        query: Raw search query string.

    Returns:
        FTS5-safe query with each token quoted.
    """
    tokens = query.split()
    if not tokens:
        return ""
    escaped = []
    for token in tokens:
        # Double any internal quotes for FTS5 escaping
        safe = token.replace('"', '""')
        escaped.append(f'"{safe}"')
    return " ".join(escaped)
