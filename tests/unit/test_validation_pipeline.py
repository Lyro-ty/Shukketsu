"""Tests for validation pipeline orchestrator."""

import sqlite3

from code.shukketsu.sim.validation_pipeline import ValidationPipeline


class TestValidationPipeline:
    """Tests for the ValidationPipeline class."""

    def test_construction(self, test_db: sqlite3.Connection) -> None:
        """Pipeline can be constructed with a DB connection."""
        pipeline = ValidationPipeline(test_db)
        assert pipeline is not None

    async def test_run_no_fights_returns_empty_report(self, test_db: sqlite3.Connection) -> None:
        """No valid fights -> report with 0 included fights."""
        pipeline = ValidationPipeline(test_db)
        report = await pipeline.run_validation("NonexistentPlayer", race="orc")
        assert report.included_fights == 0
        assert report.overall_status == "pass"

    async def test_run_stores_report_in_db(self, test_db: sqlite3.Connection) -> None:
        """Validation report is stored in validation_runs table."""
        pipeline = ValidationPipeline(test_db)
        report = await pipeline.run_validation("Lyroo", race="orc")
        row = test_db.execute("SELECT * FROM validation_runs ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert row["character_name"] == "Lyroo"
        assert row["run_type"] == "wcl"
        assert row["overall_status"] == report.overall_status

    async def test_progress_callback_fires(self, test_db: sqlite3.Connection) -> None:
        """Progress callback is invoked during pipeline run."""
        calls: list[str] = []
        pipeline = ValidationPipeline(test_db)
        await pipeline.run_validation("Lyroo", race="orc", on_progress=lambda msg: calls.append(msg))
        assert len(calls) >= 2  # At least "filtering fights" and "no valid fights"

    async def test_report_json_stored_as_valid_json(self, test_db: sqlite3.Connection) -> None:
        """report_json column contains valid JSON that round-trips."""
        import json

        pipeline = ValidationPipeline(test_db)
        await pipeline.run_validation("TestPlayer", race="human")
        row = test_db.execute("SELECT report_json FROM validation_runs ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        parsed = json.loads(row["report_json"])
        assert parsed["character_name"] == "TestPlayer"
        assert "overall_dps_drift_pct" in parsed

    async def test_empty_db_returns_zero_total_fights(self, test_db: sqlite3.Connection) -> None:
        """Empty DB has no WCL fights, so total_fights is 0."""
        pipeline = ValidationPipeline(test_db)
        report = await pipeline.run_validation("Nobody", race="human")
        assert report.total_fights == 0
        assert report.excluded_fights == 0
        assert report.included_fights == 0
        assert report.per_fight == []
        assert report.per_boss == {}
