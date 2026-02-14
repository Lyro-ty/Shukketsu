"""OAuth2 client credentials token management for WCL API."""

import asyncio
import logging
import time

import httpx

from code.shukketsu import config
from code.shukketsu.resilience.errors import WCLAuthError

logger = logging.getLogger(__name__)


class WCLAuth:
    """Manages OAuth2 client credentials token for WCL v2 API.

    Tokens are cached in memory and refreshed on expiry.
    Uses POST body params (not Basic Auth) per WCL requirements.
    """

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self._client_id = client_id or config.WCL_CLIENT_ID
        self._client_secret = client_secret or config.WCL_CLIENT_SECRET
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid access token, acquiring one if needed."""
        if self._token and time.time() < self._expires_at:
            return self._token
        async with self._lock:
            # Re-check after acquiring lock (another coroutine may have refreshed)
            if self._token and time.time() < self._expires_at:
                return self._token
            return await self._acquire_token()

    async def _acquire_token(self) -> str:
        """Acquire a new token via OAuth2 client credentials flow."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.WCL_TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    timeout=config.WCL_QUERY_TIMEOUT,
                )
        except httpx.ConnectError as exc:
            raise WCLAuthError(f"Cannot connect to WCL token endpoint: {exc}") from exc
        except httpx.HTTPError as exc:
            raise WCLAuthError(f"HTTP error during token acquisition: {exc}") from exc

        if response.status_code != 200:
            body_preview = response.text[:200] if response.text else "(empty)"
            raise WCLAuthError(f"WCL token request failed (HTTP {response.status_code}): {body_preview}")

        data = response.json()
        if "access_token" not in data:
            raise WCLAuthError(f"WCL token response missing 'access_token': {list(data.keys())}")
        self._token = data["access_token"]
        # Set expiry with 60-second buffer
        self._expires_at = time.time() + data.get("expires_in", 3600) - 60
        logger.info("WCL OAuth2 token acquired (expires in %ds)", data.get("expires_in", 0))
        return self._token
