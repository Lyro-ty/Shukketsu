"""Langfuse dataset manager for eval pipeline.

Syncs questions from a local JSON file into a Langfuse Dataset,
creates runs, and fetches run summaries for the dashboard.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse import Langfuse

from code.shukketsu import config

logger = logging.getLogger(__name__)


class EvalDatasetManager:
    """Manages the Langfuse Dataset lifecycle for evaluations."""

    def __init__(self, langfuse_client: Langfuse) -> None:
        self._client = langfuse_client
        self._dataset_name = config.EVAL_DATASET_NAME

    def sync_dataset(self, dataset_path: Path | None = None) -> int:
        """Load questions from JSON, upsert into Langfuse Dataset.

        Returns the number of items synced.
        """
        path = dataset_path or config.EVAL_DATASET_PATH
        with open(path) as f:
            questions: list[dict[str, Any]] = json.load(f)

        # Create dataset if it doesn't exist (idempotent)
        self._client.create_dataset(name=self._dataset_name)

        for q in questions:
            self._client.create_dataset_item(
                dataset_name=self._dataset_name,
                input={"question": q["question"]},
                expected_output=q.get("ground_truth", ""),
                metadata={
                    "id": q["id"],
                    "tier": q["tier"],
                    "complexity": q["complexity"],
                    "category": q.get("category", ""),
                    "spec": q.get("spec", ""),
                    "expected_tools": q.get("expected_tools", []),
                    "key_facts": q.get("key_facts", []),
                    "sim_validation": q.get("sim_validation"),
                },
            )

        logger.info("Synced %d questions to Langfuse dataset '%s'", len(questions), self._dataset_name)
        return len(questions)

    def create_run(self, run_name: str | None = None) -> str:
        """Create a new DatasetRun. Auto-generates name if not provided.

        Returns the run name.
        """
        if run_name is None:
            run_name = f"run-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
        return run_name

    def get_dataset_items(self) -> list[dict[str, Any]]:
        """Fetch all dataset items from Langfuse.

        Returns list of dicts with id, input, expected_output, metadata.
        """
        dataset = self._client.get_dataset(name=self._dataset_name)
        items: list[dict[str, Any]] = []
        for item in dataset.items:
            items.append(
                {
                    "id": item.id,
                    "input": item.input,
                    "expected_output": item.expected_output,
                    "metadata": item.metadata or {},
                }
            )
        return items

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent dataset runs.

        Returns list of {run_name, created_at, metadata}.
        """
        runs_response = self._client.get_dataset_runs(
            dataset_name=self._dataset_name,
            page=1,
            limit=limit,
        )
        runs: list[dict[str, Any]] = []
        for run in runs_response.data:
            runs.append(
                {
                    "run_name": run.name,
                    "created_at": str(run.created_at),
                    "metadata": run.metadata or {},
                }
            )
        return runs

    def get_run_detail(self, run_name: str) -> dict[str, Any]:
        """Get detailed info for a specific run."""
        run = self._client.get_dataset_run(
            dataset_name=self._dataset_name,
            run_name=run_name,
        )
        return {
            "run_name": run.name,
            "created_at": str(run.created_at),
            "metadata": run.metadata or {},
        }
