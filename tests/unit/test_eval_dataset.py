"""Tests for eval dataset manager."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from code.shukketsu.evals.dataset import EvalDatasetManager


def _sample_questions() -> list[dict]:
    return [
        {
            "id": "t1_q01",
            "tier": "retrieval",
            "complexity": "trivial",
            "category": "retrieval",
            "spec": "general",
            "question": "What is the hit cap?",
            "ground_truth": "9% for specials",
            "key_facts": ["9%"],
            "expected_tools": ["rag_search"],
            "sim_validation": None,
        },
        {
            "id": "t3_q01",
            "tier": "simulation",
            "complexity": "complex",
            "category": "analysis",
            "spec": "combat",
            "question": "What DPS does P1 BiS combat do?",
            "ground_truth": "1100-1250 DPS",
            "key_facts": ["1100-1250"],
            "expected_tools": ["sim_run"],
            "sim_validation": {"expected_dps_min": 1100, "expected_dps_max": 1250},
        },
    ]


class TestSyncDataset:
    def test_sync_creates_dataset_and_items(self, tmp_path: Path) -> None:
        questions = _sample_questions()
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        count = manager.sync_dataset(dataset_path=qpath)

        assert count == 2
        client.create_dataset.assert_called_once()
        assert client.create_dataset_item.call_count == 2

    def test_sync_upserts_are_idempotent(self, tmp_path: Path) -> None:
        questions = _sample_questions()[:1]
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        manager.sync_dataset(dataset_path=qpath)
        manager.sync_dataset(dataset_path=qpath)

        assert client.create_dataset.call_count == 2  # Idempotent call
        assert client.create_dataset_item.call_count == 2

    def test_sync_passes_metadata(self, tmp_path: Path) -> None:
        questions = _sample_questions()[:1]
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        manager.sync_dataset(dataset_path=qpath)

        call_kwargs = client.create_dataset_item.call_args
        assert call_kwargs.kwargs["metadata"]["tier"] == "retrieval"


class TestCreateRun:
    def test_auto_generates_name(self) -> None:
        manager = EvalDatasetManager(langfuse_client=MagicMock())
        name = manager.create_run()
        assert name.startswith("run-")

    def test_uses_custom_name(self) -> None:
        manager = EvalDatasetManager(langfuse_client=MagicMock())
        name = manager.create_run(run_name="my-run")
        assert name == "my-run"


class TestGetDatasetItems:
    def test_returns_items(self) -> None:
        client = MagicMock()
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.input = {"question": "What?"}
        mock_item.expected_output = "Answer"
        mock_item.metadata = {"tier": "retrieval"}
        mock_dataset = MagicMock()
        mock_dataset.items = [mock_item]
        client.get_dataset.return_value = mock_dataset

        manager = EvalDatasetManager(langfuse_client=client)
        items = manager.get_dataset_items()
        assert len(items) == 1
        assert items[0]["input"] == {"question": "What?"}


class TestListRuns:
    def test_returns_runs(self) -> None:
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "run-20260213"
        mock_run.created_at = "2026-02-13T00:00:00Z"
        mock_run.metadata = {}
        mock_response = MagicMock()
        mock_response.data = [mock_run]
        client.get_dataset_runs.return_value = mock_response

        manager = EvalDatasetManager(langfuse_client=client)
        runs = manager.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_name"] == "run-20260213"

    def test_empty_runs(self) -> None:
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []
        client.get_dataset_runs.return_value = mock_response

        manager = EvalDatasetManager(langfuse_client=client)
        runs = manager.list_runs()
        assert runs == []


class TestGetRunDetail:
    def test_returns_run_detail(self) -> None:
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "run-test"
        mock_run.created_at = "2026-02-13T00:00:00Z"
        mock_run.metadata = {"total": 60}
        client.get_dataset_run.return_value = mock_run

        manager = EvalDatasetManager(langfuse_client=client)
        detail = manager.get_run_detail("run-test")
        assert detail["run_name"] == "run-test"
        assert detail["metadata"]["total"] == 60
