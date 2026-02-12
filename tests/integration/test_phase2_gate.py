"""Tier 3: Phase gate threshold tests — formal evaluation.

Run with: python3 -m pytest tests/integration/test_phase2_gate.py -m e2e -v
"""

import pytest

from code.shukketsu.evals.metrics import (
    ACCURACY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    TRAJECTORY_THRESHOLD,
)
from code.shukketsu.evals.phase2_gate import run_phase_gate

pytestmark = pytest.mark.e2e


@pytest.fixture
async def phase_gate_report(seeded_db, tmp_path):  # type: ignore[no-untyped-def]
    """Run the full phase gate evaluation once for all tests."""
    report = await run_phase_gate(db_path=tmp_path / "gate.db")
    return report


class TestPhaseGate:
    async def test_faithfulness_above_threshold(self, phase_gate_report) -> None:  # type: ignore[no-untyped-def]
        """avg_faithfulness >= 0.8"""
        assert phase_gate_report.avg_faithfulness >= FAITHFULNESS_THRESHOLD

    async def test_trajectory_precision_above_threshold(self, phase_gate_report) -> None:  # type: ignore[no-untyped-def]
        """avg_trajectory_precision >= 0.7"""
        assert phase_gate_report.avg_trajectory_precision >= TRAJECTORY_THRESHOLD

    async def test_domain_accuracy_above_threshold(self, phase_gate_report) -> None:  # type: ignore[no-untyped-def]
        """avg_domain_accuracy >= 0.7"""
        assert phase_gate_report.avg_domain_accuracy >= ACCURACY_THRESHOLD
