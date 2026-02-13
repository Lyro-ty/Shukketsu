"""Tests for eval runner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.evals.runner import EvalRunner


def _make_item(
    qid: str = "t1_q01",
    tier: str = "retrieval",
    complexity: str = "moderate",
    question: str = "What is the hit cap?",
    ground_truth: str = "9%",
    expected_tools: list[str] | None = None,
    sim_validation: dict | None = None,
) -> dict:
    return {
        "id": f"item-{qid}",
        "input": {"question": question},
        "expected_output": ground_truth,
        "metadata": {
            "id": qid,
            "tier": tier,
            "complexity": complexity,
            "expected_tools": expected_tools or ["rag_search"],
            "key_facts": ["9%"],
            "sim_validation": sim_validation,
        },
    }


@pytest.fixture
def runner() -> EvalRunner:
    dm = MagicMock()
    dm.create_run.return_value = "test-run"
    client = MagicMock()
    # Mock span creation — returns object with .trace_id
    mock_span = MagicMock()
    mock_span.trace_id = "trace-auto"
    client.start_span.return_value = mock_span
    # Mock low-level API for dataset run items
    client.api.dataset_run_items.create.return_value = MagicMock()
    return EvalRunner(dataset_manager=dm, langfuse_client=client)


def _setup_mocks(
    mock_extract: AsyncMock,
    mock_faith: AsyncMock,
    mock_acc: AsyncMock,
    mock_rel: AsyncMock,
    mock_classify: AsyncMock,
    trivial_answer: str = "9%",
) -> None:
    """Configure common mock return values."""
    from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

    mock_extract.return_value = []
    mock_faith.return_value = []
    mock_rel.return_value = 0.9
    mock_acc.return_value = 0.8
    mock_classify.return_value = RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=trivial_answer,
    )


class TestRun:
    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_runs_all_items(self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify)

        runner._dm.get_dataset_items.return_value = [_make_item(), _make_item(qid="t1_q02")]

        report = await runner.run()
        assert len(report.question_results) == 2

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_tier_filter(self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, trivial_answer="answer")

        runner._dm.get_dataset_items.return_value = [
            _make_item(qid="t1_q01", tier="retrieval"),
            _make_item(qid="t3_q01", tier="simulation"),
        ]

        report = await runner.run(tier="retrieval")
        assert len(report.question_results) == 1
        assert report.question_results[0].tier == "retrieval"

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_progress_callback(self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify)

        runner._dm.get_dataset_items.return_value = [_make_item()]
        progress = AsyncMock()

        await runner.run(on_progress=progress)
        progress.assert_called_once_with(1, 1)

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_agent_error_records_result(
        self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.0
        mock_acc.return_value = 0.0
        mock_classify.side_effect = Exception("Router down")

        runner._dm.get_dataset_items.return_value = [_make_item()]

        report = await runner.run()
        assert len(report.question_results) == 1
        # Answer will be "ERROR: ..." which judges score as 0.0

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_scores_recorded_to_langfuse(
        self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify)

        runner._dm.get_dataset_items.return_value = [_make_item()]

        await runner.run()
        # Should have 4 per-question scores + 1 run summary score = 5 total
        assert runner._client.create_score.call_count == 5
        runner._client.api.dataset_run_items.create.assert_called_once()

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_dps_from_answer")
    async def test_sim_tier_blends_accuracy(
        self, mock_dps, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, trivial_answer="1150 DPS")
        mock_dps.return_value = 1150.0

        runner._dm.get_dataset_items.return_value = [
            _make_item(
                qid="t3_q01",
                tier="simulation",
                sim_validation={"expected_dps_min": 1100, "expected_dps_max": 1200},
            )
        ]

        report = await runner.run()
        # sim_accuracy = 1.0 (within range), domain = 0.8, blend = 0.9
        assert report.question_results[0].domain_accuracy == pytest.approx(0.9)

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    async def test_run_name_passed_to_dataset_manager(
        self, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        _setup_mocks(mock_extract, mock_faith, mock_acc, mock_rel, mock_classify)
        runner._dm.get_dataset_items.return_value = [_make_item()]

        await runner.run(run_name="custom-run")
        runner._dm.create_run.assert_called_once_with(run_name="custom-run")
