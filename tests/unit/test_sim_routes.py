"""Tests for simulation web UI routes."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def client():
    """Async HTTP client bound to the FastAPI app."""
    transport = ASGITransport(app=_get_app())
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# GET /sim/ — main page
# ---------------------------------------------------------------------------


class TestSimPage:
    @pytest.mark.asyncio
    async def test_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_contains_title(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert "DPS Simulation" in response.text

    @pytest.mark.asyncio
    async def test_has_import_form(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert "import_text" in response.text

    @pytest.mark.asyncio
    async def test_has_preset_options(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert "full_25man" in response.text or "Full 25Man" in response.text

    @pytest.mark.asyncio
    async def test_includes_chart_js(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert "chart.js" in response.text.lower() or "Chart" in response.text

    @pytest.mark.asyncio
    async def test_includes_sim_charts_script(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert "sim-charts.js" in response.text


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    @pytest.mark.asyncio
    async def test_sim_page_has_sim_nav_link(self, client: AsyncClient) -> None:
        response = await client.get("/sim/")
        assert 'href="/sim/"' in response.text

    @pytest.mark.asyncio
    async def test_chat_page_has_sim_nav_link(self, client: AsyncClient) -> None:
        """The Sim nav link should appear on all pages via base.html."""
        response = await client.get("/chat")
        assert 'href="/sim/"' in response.text


# ---------------------------------------------------------------------------
# POST /sim/import
# ---------------------------------------------------------------------------


class TestSimImport:
    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.sim.SimRunner")
    async def test_valid_import_returns_gear_table(self, mock_runner_cls: MagicMock, client: AsyncClient) -> None:
        """POST /sim/import with valid data returns gear table HTML."""
        mock_runner = MagicMock()
        mock_config = MagicMock()
        mock_config.gear = {}
        mock_runner.build_config_from_import.return_value = mock_config
        mock_runner_cls.return_value = mock_runner

        response = await client.post("/sim/import", data={"import_text": "rogue=Test"})
        assert response.status_code == 200
        assert "Gear" in response.text

    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.sim.SimRunner")
    async def test_invalid_import_returns_error(self, mock_runner_cls: MagicMock, client: AsyncClient) -> None:
        """POST /sim/import with bad data returns error partial."""
        mock_runner = MagicMock()
        mock_runner.build_config_from_import.side_effect = ValueError("Bad import string")
        mock_runner_cls.return_value = mock_runner

        response = await client.post("/sim/import", data={"import_text": "garbage"})
        assert response.status_code == 200
        assert "Import failed" in response.text


# ---------------------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------------------


class TestApiPresets:
    @pytest.mark.asyncio
    async def test_returns_list(self, client: AsyncClient) -> None:
        response = await client.get("/api/sim/presets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "full_25man" in data

    @pytest.mark.asyncio
    async def test_contains_expected_presets(self, client: AsyncClient) -> None:
        response = await client.get("/api/sim/presets")
        data = response.json()
        assert "self_only" in data
        assert "full_10man" in data


class TestApiItems:
    @pytest.mark.asyncio
    async def test_valid_slot_returns_list(self, client: AsyncClient) -> None:
        response = await client.get("/api/sim/items/trinket_1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_invalid_slot_returns_empty(self, client: AsyncClient) -> None:
        response = await client.get("/api/sim/items/invalid_slot")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_item_has_expected_fields(self, client: AsyncClient) -> None:
        """Items for a known slot should have id, name, item_level, phase."""
        response = await client.get("/api/sim/items/main_hand")
        data = response.json()
        if data:  # There are curated main_hand items
            item = data[0]
            assert "id" in item
            assert "name" in item
            assert "item_level" in item
            assert "phase" in item
