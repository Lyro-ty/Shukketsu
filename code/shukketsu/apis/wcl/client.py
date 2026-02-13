"""Rate-limit-aware async GraphQL client for WCL v2 API."""

import asyncio
import logging

import httpx

from code.shukketsu import config
from code.shukketsu.apis.wcl.auth import WCLAuth
from code.shukketsu.resilience.circuit_breaker import wcl_breaker
from code.shukketsu.resilience.errors import WCLQueryError, WCLRateLimitError

logger = logging.getLogger(__name__)


class WCLClient:
    """Rate-limit-aware async GraphQL client for WCL v2 API."""

    ENDPOINTS = {
        "classic": config.WCL_CLASSIC_ENDPOINT,
        "fresh": config.WCL_FRESH_ENDPOINT,
    }

    def __init__(self, auth: WCLAuth) -> None:
        self._auth = auth
        self._query_count = 0
        self._points_spent = 0.0
        self._points_limit = float(config.WCL_RATE_LIMIT_BUDGET)
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        """Return a shared httpx client, creating it lazily."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient()
        return self._http

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    async def query(
        self,
        graphql: str,
        variables: dict | None = None,
        endpoint: str = "fresh",
    ) -> dict:
        """Execute a GraphQL query against the WCL API.

        Args:
            graphql: GraphQL query string.
            variables: Query variables.
            endpoint: "fresh" or "classic".

        Returns:
            The "data" dict from the GraphQL response.

        Raises:
            WCLQueryError: On GraphQL errors.
            WCLRateLimitError: When rate limit is exceeded.
        """
        url = self.ENDPOINTS.get(endpoint)
        if not url:
            raise WCLQueryError(f"Unknown endpoint: {endpoint!r}")

        async def _execute() -> dict:
            token = await self._auth.get_token()
            payload: dict = {"query": graphql}
            if variables:
                payload["variables"] = variables

            client = await self._get_http()
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=config.WCL_QUERY_TIMEOUT,
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "60"))
                raise WCLRateLimitError(
                    f"WCL rate limited (429), retry after {retry_after}s",
                    points_reset_in=retry_after,
                )

            if response.status_code != 200:
                raise WCLQueryError(f"WCL API returned HTTP {response.status_code}: {response.text[:200]}")

            result = response.json()

            if "errors" in result:
                error_msgs = [e.get("message", "Unknown") for e in result["errors"]]
                combined = "; ".join(error_msgs)
                if "archived" in combined.lower():
                    raise WCLQueryError(f"Report is archived: {combined}")
                raise WCLQueryError(f"GraphQL errors: {combined}")

            data: dict = result.get("data", {})
            return data

        # Execute through circuit breaker with retry on 429
        retries = 0
        max_retries = 3
        while True:
            try:
                result = await wcl_breaker.call(_execute)
                self._query_count += 1
                # Periodically check rate limit
                if self._query_count % config.WCL_RATE_CHECK_INTERVAL == 0:
                    await self._check_and_throttle(endpoint)
                result_data: dict = result
                return result_data
            except WCLRateLimitError as exc:
                retries += 1
                if retries >= max_retries:
                    raise
                sleep_time = exc.points_reset_in
                logger.warning("Rate limited, sleeping %.1fs (attempt %d/%d)", sleep_time, retries, max_retries)
                await asyncio.sleep(sleep_time)

    async def check_rate_limit(self, endpoint: str = "fresh") -> dict:
        """Query WCL rate limit status.

        Returns:
            Dict with pointsSpentThisHour, limitPerHour, pointsResetIn.
        """
        from code.shukketsu.apis.wcl.queries import RATE_LIMIT

        data = await self.query(RATE_LIMIT, endpoint=endpoint)
        rate_data: dict = data.get("rateLimitData", {})
        return rate_data

    async def _check_and_throttle(self, endpoint: str) -> None:
        """Check rate limit and sleep if approaching budget."""
        try:
            rate_data = await self.check_rate_limit(endpoint)
            spent = rate_data.get("pointsSpentThisHour", 0)
            self._points_spent = float(spent)
            budget = self._points_limit - config.WCL_RATE_LIMIT_BUFFER
            if self._points_spent >= budget:
                reset_in = rate_data.get("pointsResetIn", 60)
                logger.warning(
                    "Approaching rate limit (%.0f/%.0f points), sleeping %.0fs",
                    self._points_spent,
                    self._points_limit,
                    reset_in,
                )
                await asyncio.sleep(float(reset_in))
        except Exception:
            logger.debug("Rate limit check failed, continuing", exc_info=True)
