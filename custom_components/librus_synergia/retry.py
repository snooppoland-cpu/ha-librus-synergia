"""Retry logic with exponential backoff for Librus Synergia API requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aiohttp
from librus_synergia.exceptions import LibrusConnectionError, LibrusServerMaintenanceError

from .security import register_credential_filter

_LOGGER = logging.getLogger(__name__)
register_credential_filter(_LOGGER)

T = TypeVar("T")

DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    LibrusConnectionError,
    LibrusServerMaintenanceError,
    TimeoutError,
    asyncio.TimeoutError,
    aiohttp.ClientError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientResponseError,
    ConnectionResetError,
    OSError,
)


async def async_retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    retryable_exceptions: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> T:
    """Execute an async operation with retry logic and exponential backoff.

    Retries transient connection, server maintenance, timeout, and network errors.
    Permanent errors (e.g. invalid credentials, captcha) fail immediately without retry.
    """
    delay = initial_delay
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as err:
            last_err = err
            if attempt >= max_retries:
                _LOGGER.warning(
                    "API request failed after %d attempts: %s",
                    max_retries,
                    err,
                )
                raise
            _LOGGER.debug(
                "API request failed (attempt %d/%d): %s. Retrying in %.2fs...",
                attempt,
                max_retries,
                err,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    assert last_err is not None
    raise last_err
