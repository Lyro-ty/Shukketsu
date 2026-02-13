"""Fine-tuning dataset export from Langfuse traces.

Exports positively-scored conversation traces as JSONL files in
ShareGPT or function-calling format for fine-tuning.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse import Langfuse

from code.shukketsu import config

logger = logging.getLogger(__name__)


class TrainingExporter:
    """Exports Langfuse traces as JSONL training data."""

    def __init__(self, langfuse_client: Langfuse) -> None:
        self._client = langfuse_client

    def export(
        self,
        format: str = "sharegpt",
        min_feedback_score: float = 1.0,
        min_trajectory_precision: float = 0.7,
        output_path: Path | None = None,
    ) -> Path:
        """Export positively-scored traces as JSONL.

        Args:
            format: "sharegpt" or "function_calling".
            min_feedback_score: Minimum user feedback score to include.
            min_trajectory_precision: Minimum trajectory precision to include.
            output_path: Where to write the file. Auto-generated if None.

        Returns:
            Path to the written JSONL file.
        """
        if output_path is None:
            config.EXPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
            output_path = config.EXPORT_OUTPUT_DIR / f"training-{format}-{timestamp}.jsonl"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        traces = self._fetch_scored_traces(min_feedback_score, min_trajectory_precision)

        count = 0
        with open(output_path, "w") as f:
            for trace in traces:
                if format == "sharegpt":
                    record = self._to_sharegpt(trace)
                else:
                    record = self._to_function_calling(trace)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1

        logger.info("Exported %d traces to %s (format=%s)", count, output_path, format)
        return output_path

    def _fetch_scored_traces(
        self,
        min_feedback: float,
        min_trajectory: float,
    ) -> list[dict[str, Any]]:
        """Fetch traces with positive feedback from Langfuse."""
        try:
            traces_response = self._client.fetch_traces(
                tags=["chat"],
                limit=500,
            )
            scored: list[dict[str, Any]] = []
            for trace in traces_response.data:
                scores = {s.name: s.value for s in (trace.scores or [])}
                feedback = scores.get("user_feedback", -1)
                trajectory = scores.get("trajectory_precision", -1)

                if feedback >= min_feedback and trajectory >= min_trajectory:
                    scored.append(
                        {
                            "input": trace.input,
                            "output": trace.output,
                            "scores": scores,
                            "observations": trace.observations or [],
                        }
                    )
            return scored
        except Exception:
            logger.warning("Failed to fetch traces from Langfuse", exc_info=True)
            return []

    def _to_sharegpt(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a trace to ShareGPT format."""
        input_text = trace.get("input")
        output_text = trace.get("output")
        if not input_text or not output_text:
            return None

        return {
            "conversations": [
                {"from": "system", "value": config.SYSTEM_PROMPT},
                {"from": "human", "value": str(input_text)},
                {"from": "gpt", "value": str(output_text)},
            ]
        }

    def _to_function_calling(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a trace to function-calling format."""
        input_text = trace.get("input")
        output_text = trace.get("output")
        if not input_text or not output_text:
            return None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": str(input_text)},
        ]

        # Extract tool calls from observations
        for obs in trace.get("observations", []):
            if hasattr(obs, "type") and obs.type == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": obs.name or "unknown",
                                    "arguments": json.dumps(obs.input or {}),
                                }
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": str(obs.output or ""),
                    }
                )

        messages.append({"role": "assistant", "content": str(output_text)})
        return {"messages": messages}


def main() -> None:
    """CLI entry point for fine-tuning export."""
    import argparse

    from code.shukketsu.observability.tracer import get_client, init_langfuse

    parser = argparse.ArgumentParser(description="Export Langfuse traces as training data")
    parser.add_argument("--format", choices=["sharegpt", "function_calling"], default="sharegpt")
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum feedback score")
    parser.add_argument("--min-trajectory", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    init_langfuse()
    client = get_client()
    exporter = TrainingExporter(langfuse_client=client)
    output = args.output and Path(args.output)
    path = exporter.export(
        format=args.format,
        min_feedback_score=args.min_score,
        min_trajectory_precision=args.min_trajectory,
        output_path=output,
    )
    print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
