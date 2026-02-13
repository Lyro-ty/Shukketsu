"""Tests for eval dashboard routes."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_dm(runs=None):
    dm = MagicMock()
    dm.list_runs.return_value = runs or []
    dm.sync_dataset.return_value = 60
    return dm


class TestDashboard:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_200(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/")
        assert resp.status_code == 200

    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_empty_state(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/")
        assert resp.status_code == 200


class TestHistory:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_json(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm(
            [
                {"run_name": "run-1", "created_at": "2026-02-13", "metadata": {}},
            ]
        )
        client = TestClient(_get_app())
        resp = client.get("/evals/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert len(data["runs"]) == 1

    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_empty_history(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/history")
        assert resp.json()["runs"] == []


class TestLatest:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_fragment(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/latest")
        assert resp.status_code == 200


class TestRunStatus:
    @patch("code.shukketsu.web.routers.evals._current_run", None)
    def test_idle_status(self) -> None:
        client = TestClient(_get_app())
        resp = client.get("/evals/run/status")
        assert resp.status_code == 200


class TestTriggerRun:
    @patch("code.shukketsu.web.routers.evals._run_eval_background")
    @patch("code.shukketsu.web.routers.evals._current_run", None)
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_trigger_returns_status(self, mock_dm_fn: MagicMock, mock_bg: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.post("/evals/run")
        assert resp.status_code == 200

    @patch("code.shukketsu.web.routers.evals._run_eval_background")
    @patch("code.shukketsu.web.routers.evals._current_run", None)
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_tier_filter_accepted(self, mock_dm_fn: MagicMock, mock_bg: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.post("/evals/run?tier=retrieval")
        assert resp.status_code == 200
