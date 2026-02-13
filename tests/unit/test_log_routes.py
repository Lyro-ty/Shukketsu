"""Tests for combat log upload routes."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from code.shukketsu.sim.comparator import AbilityMetrics, WCLFightMetrics


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def client():
    """Async HTTP client bound to the FastAPI app."""
    transport = ASGITransport(app=_get_app())
    return AsyncClient(transport=transport, base_url="http://test")


def _make_fight(
    total_damage: int = 100000,
    active_dps: float = 1500.0,
    fight_duration_ms: int = 66000,
) -> WCLFightMetrics:
    """Create a WCLFightMetrics fixture."""
    return WCLFightMetrics(
        total_damage=total_damage,
        active_dps=active_dps,
        fight_duration_ms=fight_duration_ms,
        ability_breakdown={
            "sinister_strike": AbilityMetrics(damage_total=60000, damage_pct=60.0, cast_count=40),
            "melee": AbilityMetrics(damage_total=40000, damage_pct=40.0, cast_count=0),
        },
        buff_uptimes={"slice_and_dice": 0.95},
        proc_counts={},
    )


# ---------------------------------------------------------------------------
# GET /logs/ — upload page
# ---------------------------------------------------------------------------


class TestLogsPage:
    @pytest.mark.asyncio
    async def test_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/logs/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_contains_title(self, client: AsyncClient) -> None:
        response = await client.get("/logs/")
        assert "Combat Log Upload" in response.text

    @pytest.mark.asyncio
    async def test_has_upload_form(self, client: AsyncClient) -> None:
        response = await client.get("/logs/")
        assert "character_name" in response.text
        assert 'type="file"' in response.text

    @pytest.mark.asyncio
    async def test_has_logs_nav_link(self, client: AsyncClient) -> None:
        response = await client.get("/logs/")
        assert 'href="/logs/"' in response.text


# ---------------------------------------------------------------------------
# POST /logs/upload — valid log
# ---------------------------------------------------------------------------


class TestLogsUploadValid:
    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_valid_log_returns_200(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """POST /logs/upload with a valid log returns 200 and fight data."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [_make_fight()]
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"fake log content", "text/plain")},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_valid_log_contains_dps(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """Response contains the DPS value from parsed fight."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [_make_fight(active_dps=1500.0)]
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"fake log content", "text/plain")},
        )
        assert "1500.0" in response.text

    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_valid_log_contains_character_name(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """Response contains the character name."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [_make_fight()]
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"fake log content", "text/plain")},
        )
        assert "Lyroo" in response.text

    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_valid_log_contains_ability_breakdown(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """Response contains ability breakdown table."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [_make_fight()]
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"fake log content", "text/plain")},
        )
        assert "Sinister Strike" in response.text
        assert "60.0%" in response.text

    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_parser_called_with_correct_args(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """CLEUParser.parse is called with the file content and character name."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [_make_fight()]
        mock_parser_cls.return_value = mock_parser

        await client.post(
            "/logs/upload",
            data={"character_name": "TestChar"},
            files={"file": ("log.txt", b"some log data", "text/plain")},
        )
        mock_parser.parse.assert_called_once_with("some log data", "TestChar")


# ---------------------------------------------------------------------------
# POST /logs/upload — missing fields
# ---------------------------------------------------------------------------


class TestLogsUploadMissingFields:
    @pytest.mark.asyncio
    async def test_missing_character_name_returns_422(self, client: AsyncClient) -> None:
        """POST without character_name returns 422 validation error."""
        response = await client.post(
            "/logs/upload",
            files={"file": ("WoWCombatLog.txt", b"fake log content", "text/plain")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_file_returns_422(self, client: AsyncClient) -> None:
        """POST without file returns 422 validation error."""
        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /logs/upload — empty file
# ---------------------------------------------------------------------------


class TestLogsUploadEmptyFile:
    @pytest.mark.asyncio
    async def test_empty_file_returns_error(self, client: AsyncClient) -> None:
        """POST with an empty file returns 200 with error message."""
        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"", "text/plain")},
        )
        assert response.status_code == 200
        assert "empty" in response.text.lower()


# ---------------------------------------------------------------------------
# POST /logs/upload — no encounters found
# ---------------------------------------------------------------------------


class TestLogsUploadNoEncounters:
    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_no_encounters_returns_error(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """POST with log that has no encounters returns error message."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = []
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"some log without encounters", "text/plain")},
        )
        assert response.status_code == 200
        assert "No encounters found" in response.text


# ---------------------------------------------------------------------------
# POST /logs/upload — file too large
# ---------------------------------------------------------------------------


class TestLogsUploadTooLarge:
    @pytest.mark.asyncio
    async def test_oversized_file_returns_error(self, client: AsyncClient) -> None:
        """POST with a file exceeding 50MB limit returns error message."""
        # Create content just over the limit (50MB + 1 byte)
        # We patch _MAX_UPLOAD_BYTES to a small value to avoid allocating 50MB in tests
        with patch("code.shukketsu.web.routers.logs._MAX_UPLOAD_BYTES", 100):
            response = await client.post(
                "/logs/upload",
                data={"character_name": "Lyroo"},
                files={"file": ("WoWCombatLog.txt", b"x" * 101, "text/plain")},
            )
        assert response.status_code == 200
        assert "too large" in response.text.lower()


# ---------------------------------------------------------------------------
# POST /logs/upload — parser exception
# ---------------------------------------------------------------------------


class TestLogsUploadParserError:
    @pytest.mark.asyncio
    @patch("code.shukketsu.web.routers.logs.CLEUParser")
    async def test_parser_exception_returns_error(self, mock_parser_cls: MagicMock, client: AsyncClient) -> None:
        """POST that causes a parser exception returns error partial."""
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = ValueError("Malformed log line")
        mock_parser_cls.return_value = mock_parser

        response = await client.post(
            "/logs/upload",
            data={"character_name": "Lyroo"},
            files={"file": ("WoWCombatLog.txt", b"bad data", "text/plain")},
        )
        assert response.status_code == 200
        assert "Parse error" in response.text


# ---------------------------------------------------------------------------
# Navigation — Logs link appears on other pages
# ---------------------------------------------------------------------------


class TestLogsNavigation:
    @pytest.mark.asyncio
    async def test_chat_page_has_logs_nav_link(self, client: AsyncClient) -> None:
        """The Logs nav link should appear on all pages via base.html."""
        response = await client.get("/chat")
        assert 'href="/logs/"' in response.text

    @pytest.mark.asyncio
    async def test_sim_page_has_logs_nav_link(self, client: AsyncClient) -> None:
        """The Logs nav link should appear on the sim page."""
        response = await client.get("/sim/")
        assert 'href="/logs/"' in response.text
