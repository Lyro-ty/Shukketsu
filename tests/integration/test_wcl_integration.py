"""Integration tests for WCL API — requires real credentials.

Run with: python3 -m pytest tests/integration/test_wcl_integration.py -v -m integration
"""

import pytest

from code.shukketsu import config

pytestmark = pytest.mark.integration

# Skip all tests if credentials not configured
requires_wcl = pytest.mark.skipif(
    not config.WCL_CLIENT_ID or not config.WCL_CLIENT_SECRET,
    reason="WCL credentials not configured",
)


@requires_wcl
class TestWCLIntegration:
    """Integration tests that hit the real WCL v2 API."""

    async def test_auth_acquires_real_token(self) -> None:
        """Verify OAuth2 flow works with real credentials."""
        from code.shukketsu.apis.wcl.auth import WCLAuth

        auth = WCLAuth()
        token = await auth.get_token()
        assert isinstance(token, str)
        assert len(token) > 20  # JWT tokens are long

    async def test_query_rate_limit(self) -> None:
        """Verify rate limit data is returned."""
        from code.shukketsu.apis.wcl.auth import WCLAuth
        from code.shukketsu.apis.wcl.client import WCLClient

        auth = WCLAuth()
        client = WCLClient(auth)
        data = await client.check_rate_limit("fresh")
        assert "pointsSpentThisHour" in data
        assert "limitPerHour" in data
        assert data["limitPerHour"] > 0

    async def test_query_zone_metadata(self) -> None:
        """Verify zone metadata query works (BWL zone 1034)."""
        from code.shukketsu.apis.wcl.auth import WCLAuth
        from code.shukketsu.apis.wcl.client import WCLClient
        from code.shukketsu.apis.wcl.queries import build_zone_metadata_query

        auth = WCLAuth()
        client = WCLClient(auth)
        query, variables = build_zone_metadata_query(1034)
        data = await client.query(query, variables, endpoint="fresh")
        zone = data.get("worldData", {}).get("zone", {})
        assert zone.get("name") == "Blackwing Lair"
        assert len(zone.get("encounters", [])) > 0

    async def test_query_character_lyroo(self) -> None:
        """Verify character lookup for Lyroo on fresh endpoint."""
        from code.shukketsu.apis.wcl.auth import WCLAuth
        from code.shukketsu.apis.wcl.client import WCLClient
        from code.shukketsu.apis.wcl.queries import build_character_query

        auth = WCLAuth()
        client = WCLClient(auth)
        query, variables = build_character_query(104956434, report_limit=3)
        data = await client.query(query, variables, endpoint="fresh")
        char = data.get("characterData", {}).get("character", {})
        assert char.get("name") == "Lyroo"
        assert char.get("classID") == 8  # Rogue
