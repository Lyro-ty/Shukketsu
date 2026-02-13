"""Tests for the rate-limit-aware WCL GraphQL client.

Verifies query execution, header construction, error handling,
rate limit tracking, 429 retry logic, and circuit breaker integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.apis.wcl.auth import WCLAuth
from code.shukketsu.apis.wcl.client import WCLClient
from code.shukketsu.resilience.circuit_breaker import wcl_breaker
from code.shukketsu.resilience.errors import CircuitOpenError, WCLQueryError, WCLRateLimitError


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> MagicMock:
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {"data": {"test": "value"}}
    resp.text = str(json_data)
    resp.headers = headers or {}
    return resp


@pytest.fixture
def mock_auth() -> WCLAuth:
    """Provide a WCLAuth with a mocked get_token."""
    auth = WCLAuth(client_id="test", client_secret="test")
    auth.get_token = AsyncMock(return_value="test-token")
    return auth


@pytest.fixture(autouse=True)
def _reset_wcl_breaker() -> None:
    """Reset the WCL circuit breaker before each test."""
    wcl_breaker.reset()


def _patch_http(mock_http: AsyncMock):
    """Patch _get_http to return a shared mock client."""
    return patch.object(WCLClient, "_get_http", new_callable=AsyncMock, return_value=mock_http)


class TestWCLClient:
    """Tests for WCLClient."""

    async def test_query_sends_correct_headers(self, mock_auth: WCLAuth) -> None:
        """Bearer token is sent in Authorization header."""
        mock_resp = _make_mock_response()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)
            await client.query("{ test }")

            call_kwargs = mock_http.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer test-token"
            assert headers["Content-Type"] == "application/json"

    async def test_query_returns_data(self, mock_auth: WCLAuth) -> None:
        """Successful response returns the 'data' dict."""
        expected = {"worldData": {"zone": {"name": "Karazhan"}}}
        mock_resp = _make_mock_response(json_data={"data": expected})
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)
            result = await client.query("{ worldData { zone { name } } }")

            assert result == expected

    async def test_query_raises_on_graphql_errors(self, mock_auth: WCLAuth) -> None:
        """Response with 'errors' key raises WCLQueryError."""
        error_data = {
            "errors": [{"message": "Field 'invalid' not found"}],
            "data": None,
        }
        mock_resp = _make_mock_response(json_data=error_data)
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            with pytest.raises(WCLQueryError, match="GraphQL errors"):
                await client.query("{ invalid }")

    async def test_query_handles_archived_report(self, mock_auth: WCLAuth) -> None:
        """'archived' in GraphQL error message raises WCLQueryError with archived text."""
        error_data = {
            "errors": [{"message": "This report has been archived and is no longer available"}],
            "data": None,
        }
        mock_resp = _make_mock_response(json_data=error_data)
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            with pytest.raises(WCLQueryError, match="archived"):
                await client.query('{ reportData { report(code: "old") { title } } }')

    async def test_query_selects_correct_endpoint(self, mock_auth: WCLAuth) -> None:
        """'fresh' and 'classic' map to the correct WCL API URLs."""
        from code.shukketsu import config

        mock_resp = _make_mock_response()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            await client.query("{ test }", endpoint="fresh")
            fresh_url = mock_http.post.call_args_list[0].args[0]
            assert fresh_url == config.WCL_FRESH_ENDPOINT

            await client.query("{ test }", endpoint="classic")
            classic_url = mock_http.post.call_args_list[1].args[0]
            assert classic_url == config.WCL_CLASSIC_ENDPOINT

    async def test_rate_limit_tracking(self, mock_auth: WCLAuth) -> None:
        """After WCL_RATE_CHECK_INTERVAL queries, _check_and_throttle is called."""
        from code.shukketsu import config

        mock_resp = _make_mock_response()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            with patch.object(client, "_check_and_throttle", new_callable=AsyncMock) as mock_throttle:
                # Execute queries up to the check interval
                for _ in range(config.WCL_RATE_CHECK_INTERVAL):
                    await client.query("{ test }")

                mock_throttle.assert_called_once()

    async def test_rate_limit_sleeps_when_near_budget(self, mock_auth: WCLAuth) -> None:
        """When rate limit points approach budget, asyncio.sleep is called."""
        from code.shukketsu import config

        mock_resp = _make_mock_response()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            # Mock check_rate_limit to return high usage
            high_usage = {
                "pointsSpentThisHour": config.WCL_RATE_LIMIT_BUDGET - 100,
                "limitPerHour": config.WCL_RATE_LIMIT_BUDGET,
                "pointsResetIn": 42,
            }
            with (
                patch.object(client, "check_rate_limit", new_callable=AsyncMock, return_value=high_usage),
                patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                await client._check_and_throttle("fresh")

                mock_sleep.assert_called_once_with(42.0)

    async def test_query_retries_on_429(self, mock_auth: WCLAuth) -> None:
        """First call returns 429, second call succeeds."""
        mock_429 = _make_mock_response(status_code=429, headers={"Retry-After": "0.01"})
        mock_ok = _make_mock_response(json_data={"data": {"retried": True}})

        mock_http = AsyncMock()
        mock_http.post.side_effect = [mock_429, mock_ok]

        with (
            _patch_http(mock_http),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            client = WCLClient(auth=mock_auth)
            result = await client.query("{ test }")

            assert result == {"retried": True}
            assert mock_http.post.call_count == 2

    async def test_query_raises_on_persistent_429(self, mock_auth: WCLAuth) -> None:
        """Three consecutive 429 responses raise WCLRateLimitError."""
        mock_429 = _make_mock_response(status_code=429, headers={"Retry-After": "0.01"})

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_429

        with (
            _patch_http(mock_http),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            client = WCLClient(auth=mock_auth)

            with pytest.raises(WCLRateLimitError):
                await client.query("{ test }")

    async def test_circuit_breaker_integration(self, mock_auth: WCLAuth) -> None:
        """After CB_WCL_FAILURE_THRESHOLD failures, CircuitOpenError is raised."""
        from code.shukketsu import config

        mock_500 = _make_mock_response(status_code=500, json_data={})
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_500

        with _patch_http(mock_http):
            client = WCLClient(auth=mock_auth)

            # Trip the circuit breaker with enough failures
            for _ in range(config.CB_WCL_FAILURE_THRESHOLD):
                with pytest.raises(WCLQueryError):
                    await client.query("{ test }")

            # Next call should be rejected by the open circuit breaker
            with pytest.raises(CircuitOpenError):
                await client.query("{ test }")

    async def test_close_releases_http_client(self, mock_auth: WCLAuth) -> None:
        """close() calls aclose() on the underlying httpx client."""
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()

        client = WCLClient(auth=mock_auth)
        client._http = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._http is None

    async def test_close_noop_when_no_client(self, mock_auth: WCLAuth) -> None:
        """close() is safe to call when no HTTP client was created."""
        client = WCLClient(auth=mock_auth)
        await client.close()  # Should not raise
