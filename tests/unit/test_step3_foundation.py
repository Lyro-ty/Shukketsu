"""Tests for Phase 2 Step 3 foundation: config, errors, circuit breaker."""

from code.shukketsu import config
from code.shukketsu.resilience.errors import (
    FailureMode,
    GraphTraversalError,
    RerankerError,
    ShukketsuError,
)


class TestNewErrorClasses:
    """Tests for RerankerError and GraphTraversalError."""

    def test_reranker_error_inherits_shukketsu_error(self) -> None:
        err = RerankerError("reranker failed")
        assert isinstance(err, ShukketsuError)
        assert err.failure_mode == FailureMode.MODEL_UNAVAILABLE
        assert "reranker failed" in str(err)

    def test_graph_traversal_error_inherits_shukketsu_error(self) -> None:
        err = GraphTraversalError("graph query failed")
        assert isinstance(err, ShukketsuError)
        assert err.failure_mode == FailureMode.DB_ERROR
        assert "graph query failed" in str(err)


class TestNewConfig:
    """Tests for new config constants."""

    def test_graph_search_default_top_k_exists(self) -> None:
        assert hasattr(config, "GRAPH_SEARCH_DEFAULT_TOP_K")
        assert isinstance(config.GRAPH_SEARCH_DEFAULT_TOP_K, int)
        assert config.GRAPH_SEARCH_DEFAULT_TOP_K == 20

    def test_reranker_fetch_multiplier_exists(self) -> None:
        assert hasattr(config, "RERANKER_FETCH_MULTIPLIER")
        assert isinstance(config.RERANKER_FETCH_MULTIPLIER, int)
        assert config.RERANKER_FETCH_MULTIPLIER == 3

    def test_reranker_top_k_exists(self) -> None:
        assert hasattr(config, "RERANKER_TOP_K")
        assert isinstance(config.RERANKER_TOP_K, int)
        assert config.RERANKER_TOP_K == 5

    def test_cb_reranker_constants_exist(self) -> None:
        assert config.CB_RERANKER_FAILURE_THRESHOLD == 3
        assert config.CB_RERANKER_RECOVERY_TIMEOUT == 30.0


class TestRerankerBreaker:
    """Tests for the reranker circuit breaker instance."""

    def test_breaker_exists(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import reranker_breaker

        assert reranker_breaker.name == "reranker"

    def test_breaker_has_correct_threshold(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import reranker_breaker

        assert reranker_breaker._failure_threshold == 3

    def test_reset_all_breakers_includes_new_breaker(self) -> None:
        from code.shukketsu.resilience.circuit_breaker import (
            reranker_breaker,
            reset_all_breakers,
        )

        # Force breaker into failure state
        reranker_breaker._failure_count = 10
        reset_all_breakers()
        assert reranker_breaker._failure_count == 0
