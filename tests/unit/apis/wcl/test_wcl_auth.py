"""Tests for WCL OAuth2 token management.

Verifies token acquisition, caching, expiry refresh, error handling,
and correct use of POST body params (not Basic Auth).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from code.shukketsu.apis.wcl.auth import WCLAuth
from code.shukketsu.resilience.errors import WCLAuthError


@pytest.fixture
def mock_client():
    """Provide a mock httpx.AsyncClient with a successful token response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "test-token-123", "expires_in": 31104000}

    client = AsyncMock()
    client.post.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield client, mock_response


class TestWCLAuth:
    """Tests for WCLAuth token manager."""

    async def test_get_token_acquires_new_token(self, mock_client: tuple[AsyncMock, MagicMock]) -> None:
        """First call to get_token() acquires a new token via HTTP POST."""
        client, _ = mock_client
        auth = WCLAuth(client_id="test-id", client_secret="test-secret")

        token = await auth.get_token()

        assert token == "test-token-123"
        client.post.assert_called_once()

    async def test_get_token_caches_token(self, mock_client: tuple[AsyncMock, MagicMock]) -> None:
        """Second call to get_token() returns cached token without HTTP request."""
        client, _ = mock_client
        auth = WCLAuth(client_id="test-id", client_secret="test-secret")

        token1 = await auth.get_token()
        token2 = await auth.get_token()

        assert token1 == token2 == "test-token-123"
        client.post.assert_called_once()

    async def test_get_token_refreshes_expired(self, mock_client: tuple[AsyncMock, MagicMock]) -> None:
        """Expired token triggers a new HTTP request."""
        client, _ = mock_client
        auth = WCLAuth(client_id="test-id", client_secret="test-secret")

        # Acquire initial token
        await auth.get_token()

        # Force expiry by setting _expires_at in the past
        auth._expires_at = time.time() - 100

        # Second call should re-acquire
        await auth.get_token()

        assert client.post.call_count == 2

    async def test_get_token_raises_on_auth_failure(self, mock_client: tuple[AsyncMock, MagicMock]) -> None:
        """HTTP 401 response raises WCLAuthError."""
        client, mock_response = mock_client
        mock_response.status_code = 401
        mock_response.text = "invalid_client"

        auth = WCLAuth(client_id="bad-id", client_secret="bad-secret")

        with pytest.raises(WCLAuthError, match="HTTP 401"):
            await auth.get_token()

    async def test_get_token_raises_on_network_error(self) -> None:
        """Network connection failure raises WCLAuthError."""
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            auth = WCLAuth(client_id="test-id", client_secret="test-secret")

            with pytest.raises(WCLAuthError, match="Cannot connect"):
                await auth.get_token()

    async def test_get_token_uses_body_params(self, mock_client: tuple[AsyncMock, MagicMock]) -> None:
        """Token request sends credentials as POST body params, not Basic Auth."""
        client, _ = mock_client
        auth = WCLAuth(client_id="my-client-id", client_secret="my-client-secret")

        await auth.get_token()

        call_kwargs = client.post.call_args
        post_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")

        assert post_data is not None
        assert post_data["grant_type"] == "client_credentials"
        assert post_data["client_id"] == "my-client-id"
        assert post_data["client_secret"] == "my-client-secret"

        # Verify no auth header was used
        assert "auth" not in (call_kwargs.kwargs or {})
