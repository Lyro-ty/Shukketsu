"""Tests for fine-tuning export CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from code.shukketsu.evals.export import TrainingExporter


def _mock_trace_summary() -> MagicMock:
    """Create a mock TraceWithDetails (list summary — scores are IDs only)."""
    summary = MagicMock()
    summary.id = "trace-abc"
    return summary


def _mock_full_trace(
    input_text: str = "What is hit cap?",
    output_text: str | None = "9%",
    feedback: float = 1.0,
    trajectory: float = 0.8,
) -> MagicMock:
    """Create a mock TraceWithFullDetails (full trace with ScoreV1 objects)."""
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


def _setup_client(
    client: MagicMock,
    traces: list[MagicMock] | None = None,
) -> None:
    """Wire up mock client with list + get endpoints.

    Each trace_summary in list maps to a corresponding full trace via get().
    """
    if traces is None:
        traces = [_mock_full_trace()]
    resp = MagicMock()
    resp.data = [_mock_trace_summary() for _ in traces]
    client.api.trace.list.return_value = resp
    client.api.trace.get.side_effect = traces


class TestShareGPTExport:
    """Tests for ShareGPT format export."""

    def test_produces_valid_jsonl(self, tmp_path: Path) -> None:
        client = MagicMock()
        _setup_client(client, [_mock_full_trace()])

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="sharegpt", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "conversations" in record
        assert record["conversations"][1]["from"] == "human"

    def test_skips_empty_output(self, tmp_path: Path) -> None:
        client = MagicMock()
        _setup_client(client, [_mock_full_trace(output_text=None)])

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text().strip() == ""


class TestFunctionCallingExport:
    """Tests for function-calling format export."""

    def test_produces_valid_format(self, tmp_path: Path) -> None:
        client = MagicMock()
        _setup_client(client, [_mock_full_trace()])

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="function_calling", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert "messages" in record
        assert record["messages"][0]["role"] == "system"

    def test_tool_calls_include_id_and_type(self, tmp_path: Path) -> None:
        """Function-calling format must include id, type fields per OpenAI spec."""
        trace = _mock_full_trace()
        # Add a tool observation
        tool_obs = MagicMock()
        tool_obs.type = "tool"
        tool_obs.name = "rag_search"
        tool_obs.input = {"query": "hit cap"}
        tool_obs.output = "The hit cap is 9%"
        trace.observations = [tool_obs]

        client = MagicMock()
        _setup_client(client, [trace])

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="function_calling", output_path=tmp_path / "out.jsonl")

        record = json.loads(path.read_text().strip())
        messages = record["messages"]

        # Find the assistant tool_call message
        tool_call_msgs = [m for m in messages if m.get("tool_calls")]
        assert len(tool_call_msgs) == 1

        tc = tool_call_msgs[0]["tool_calls"][0]
        assert "id" in tc, "tool_call must have an id field"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "rag_search"

        # Find the tool response message
        tool_response_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_response_msgs) == 1
        assert tool_response_msgs[0]["tool_call_id"] == tc["id"]
        assert tool_response_msgs[0]["content"] == "The hit cap is 9%"


class TestFiltering:
    """Tests for score-based trace filtering."""

    def test_filters_by_feedback_score(self, tmp_path: Path) -> None:
        client = MagicMock()
        traces = [_mock_full_trace(feedback=0.0), _mock_full_trace(feedback=1.0)]
        _setup_client(client, traces)

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            output_format="sharegpt",
            min_feedback_score=1.0,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_filters_by_trajectory(self, tmp_path: Path) -> None:
        client = MagicMock()
        traces = [_mock_full_trace(trajectory=0.3), _mock_full_trace(trajectory=0.9)]
        _setup_client(client, traces)

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            output_format="sharegpt",
            min_trajectory_precision=0.7,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_empty_result_set(self, tmp_path: Path) -> None:
        client = MagicMock()
        _setup_client(client, [])

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text() == ""

    def test_output_path_created(self, tmp_path: Path) -> None:
        client = MagicMock()
        _setup_client(client, [_mock_full_trace()])

        nested = tmp_path / "subdir" / "out.jsonl"
        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(output_format="sharegpt", output_path=nested)
        assert path.exists()
