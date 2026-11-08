"""Retry logic with exponential backoff for Librus Synergia API requests."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aiohttp

from librus_synergia.exceptions import (
    LibrusConnectionError,
    LibrusServerMaintenanceError,
)

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
    max_retries = max(1, max_retries)
    delay = initial_delay
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as err:
            if isinstance(err, aiohttp.ClientResponseError):
                status = getattr(err, "status", None)
                if status is not None and status != 429 and not (500 <= status < 600):
                    raise
            last_err = err
            if attempt >= max_retries:
                _LOGGER.warning(
                    "API request failed after %d attempts: %s",
                    max_retries,
                    err,
                )
                raise
            sleep_delay = delay * random.uniform(0.8, 1.2)
            _LOGGER.debug(
                "API request failed (attempt %d/%d): %s. Retrying in %.2fs...",
                attempt,
                max_retries,
                err,
                sleep_delay,
            )
            await asyncio.sleep(sleep_delay)
            delay = min(delay * backoff_factor, max_delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unexpected end of retry loop")

