"""Tests for RRF fusion and FTS5 query escaping."""

from code.shukketsu.rag.fusion import compute_rrf, escape_fts_query


class TestComputeRrf:
    """Tests for Reciprocal Rank Fusion."""

    def test_disjoint_lists(self) -> None:
        """Documents in only one list get single-source RRF scores."""
        vec_ranks = {1: 0, 2: 1}  # doc 1 rank 0, doc 2 rank 1
        fts_ranks = {3: 0, 4: 1}  # doc 3 rank 0, doc 4 rank 1
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        assert len(scores) == 4
        # All same-ranked docs have equal scores
        assert scores[1] == scores[3]
        assert scores[2] == scores[4]
        # Rank 0 scores higher than rank 1
        assert scores[1] > scores[2]

    def test_overlapping_ranks_higher(self) -> None:
        """Documents appearing in both lists score higher than single-list docs."""
        vec_ranks = {1: 0, 2: 1}
        fts_ranks = {1: 0, 3: 1}  # doc 1 in both lists
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        # Doc 1 (in both) beats doc 2 and doc 3 (in one each)
        assert scores[1] > scores[2]
        assert scores[1] > scores[3]

    def test_one_empty_list(self) -> None:
        """Degrades gracefully to single-source ranking."""
        vec_ranks = {1: 0, 2: 1, 3: 2}
        fts_ranks: dict[int, int] = {}
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        assert len(scores) == 3
        assert scores[1] > scores[2] > scores[3]

    def test_both_empty(self) -> None:
        """Empty inputs return empty result."""
        scores = compute_rrf([{}, {}], k=60)
        assert scores == {}

    def test_all_overlap_same_order(self) -> None:
        """Both lists agree on ranking — scores are doubled."""
        vec_ranks = {1: 0, 2: 1}
        fts_ranks = {1: 0, 2: 1}
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        # Doc 1 score = 2 * 1/(60+0) = 2/60
        expected_1 = 2.0 / 60.0
        assert abs(scores[1] - expected_1) < 1e-9

    def test_custom_k_value(self) -> None:
        """Different k values change the score distribution."""
        ranks = [{1: 0, 2: 1}]
        scores_k10 = compute_rrf(ranks, k=10)
        scores_k100 = compute_rrf(ranks, k=100)
        # Smaller k = bigger spread between rank 0 and rank 1
        spread_k10 = scores_k10[1] - scores_k10[2]
        spread_k100 = scores_k100[1] - scores_k100[2]
        assert spread_k10 > spread_k100


class TestEscapeFtsQuery:
    """Tests for FTS5 query escaping."""

    def test_simple_query(self) -> None:
        assert escape_fts_query("DST proc rate") == '"DST" "proc" "rate"'

    def test_single_token(self) -> None:
        assert escape_fts_query("DST") == '"DST"'

    def test_empty_string(self) -> None:
        assert escape_fts_query("") == ""

    def test_extra_whitespace(self) -> None:
        assert escape_fts_query("  DST   proc  ") == '"DST" "proc"'

    def test_special_characters_escaped(self) -> None:
        """Quotes and FTS5 operators inside tokens are handled."""
        result = escape_fts_query('rogue "combat" spec')
        # Each token is individually quoted; inner quotes are doubled for FTS5
        assert '"rogue"' in result
        assert '"spec"' in result
