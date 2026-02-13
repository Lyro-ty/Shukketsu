"""Tests for fine-tuning export CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from code.shukketsu.evals.export import TrainingExporter


def _mock_trace(
    input_text: str = "What is hit cap?",
    output_text: str | None = "9%",
    feedback: float = 1.0,
    trajectory: float = 0.8,
) -> MagicMock:
    """Create a mock Langfuse trace with scores."""
    trace = MagicMock()
    trace.input = input_text
    trace.output = output_text
    score_fb = MagicMock()
    score_fb.name = "user_feedback"
    score_fb.value = feedback
    score_traj = MagicMock()
    score_traj.name = "trajectory_precision"
    score_traj.value = trajectory
    trace.scores = [score_fb, score_traj]
    trace.observations = []
    return trace


class TestShareGPTExport:
    """Tests for ShareGPT format export."""

    def test_produces_valid_jsonl(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "conversations" in record
        assert record["conversations"][1]["from"] == "human"

    def test_skips_empty_output(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(output_text=None)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text().strip() == ""


class TestFunctionCallingExport:
    """Tests for function-calling format export."""

    def test_produces_valid_format(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="function_calling", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert "messages" in record
        assert record["messages"][0]["role"] == "system"


class TestFiltering:
    """Tests for score-based trace filtering."""

    def test_filters_by_feedback_score(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(feedback=0.0), _mock_trace(feedback=1.0)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            format="sharegpt",
            min_feedback_score=1.0,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_filters_by_trajectory(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(trajectory=0.3), _mock_trace(trajectory=0.9)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            format="sharegpt",
            min_trajectory_precision=0.7,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_empty_result_set(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = []
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text() == ""

    def test_output_path_created(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        nested = tmp_path / "subdir" / "out.jsonl"
        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=nested)
        assert path.exists()
