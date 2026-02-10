"""Langfuse tracing initialization and helpers.

Configures the Langfuse SDK from values in config.py. Call init_langfuse()
once at app startup and flush_traces() on shutdown. The @observe decorator
from the langfuse package is used directly on functions in other modules.
"""

import logging
import os

from langfuse import get_client, observe  # noqa: F401 — re-exported

from code.shukketsu import config

logger = logging.getLogger(__name__)

_initialized = False


def init_langfuse() -> None:
    """Configure Langfuse from config values and eagerly create the client.

    Sets environment variables that the Langfuse SDK reads automatically,
    then initializes the singleton client. No-op when tracing is disabled.
    """
    global _initialized  # noqa: PLW0603

    if not config.LANGFUSE_TRACING_ENABLED:
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logger.info("Langfuse tracing disabled")
        return

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = config.LANGFUSE_HOST
    os.environ["LANGFUSE_SAMPLE_RATE"] = str(config.LANGFUSE_SAMPLE_RATE)

    get_client()
    _initialized = True
    logger.info("Langfuse tracing initialized (host=%s)", config.LANGFUSE_HOST)


def flush_traces() -> None:
    """Flush any buffered traces to Langfuse. Safe to call even if not initialized."""
    try:
        client = get_client()
        client.flush()
        logger.debug("Langfuse traces flushed")
    except Exception:
        logger.debug("Langfuse flush skipped (not initialized or unavailable)")
