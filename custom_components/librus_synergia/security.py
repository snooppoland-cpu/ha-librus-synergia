"""Security utilities for Librus Synergia: credential protection, CSRF validation, and session locking."""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
from collections.abc import Mapping
from typing import Any

from librus_synergia.exceptions import LibrusUnexpectedResponseError

_LOGGER = logging.getLogger(__name__)

# Keys that contain sensitive authentication credentials or tokens
SENSITIVE_KEYS = frozenset(
    {
        "pass",
        "password",
        "token",
        "access_token",
        "refresh_token",
        "cookie",
        "cookies",
        "authorization",
        "client_secret",
        "secret",
        "auth",
        "csrf",
        "csrf_token",
        "_csrf",
    }
)

REDACTED = "***REDACTED***"

# Valid CSRF token pattern: base64/hex/alphanumeric characters with length 8-256
_CSRF_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-\.]{8,256}$")


def sanitize_payload(payload: Any) -> Any:
    """Sanitize sensitive credentials from data structures before logging.

    Replaces passwords, auth tokens, cookies, and secret values with REDACTED.
    """
    if isinstance(payload, Mapping):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key).lower()
            if key_str in SENSITIVE_KEYS or any(secret in key_str for secret in ("password", "secret", "token")):
                sanitized[str(key)] = REDACTED
            elif isinstance(value, (Mapping, list, tuple)):
                sanitized[str(key)] = sanitize_payload(value)
            else:
                sanitized[str(key)] = value
        return sanitized
    if isinstance(payload, (list, tuple)):
        return [sanitize_payload(item) for item in payload]
    return payload


class CredentialFilter(logging.Filter):
    """Logging filter ensuring auth payloads and credentials are never emitted in logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if isinstance(record.args, dict):
                record.args = sanitize_payload(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(sanitize_payload(arg) for arg in record.args)
        if isinstance(record.msg, str):
            for secret_pattern in (r"(password['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])", r"(pass['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])"):
                record.msg = re.sub(secret_pattern, r"\g<1>***REDACTED***\g<2>", record.msg, flags=re.IGNORECASE)
        return True


SUBMODULE_LOGGER_NAMES: tuple[str, ...] = (
    "custom_components.librus_synergia",
    "custom_components.librus_synergia.absence_budget",
    "custom_components.librus_synergia.absence_days",
    "custom_components.librus_synergia.achievements",
    "custom_components.librus_synergia.ai_summary",
    "custom_components.librus_synergia.attachment_view",
    "custom_components.librus_synergia.average_history",
    "custom_components.librus_synergia.binary_sensor",
    "custom_components.librus_synergia.button",
    "custom_components.librus_synergia.calendar",
    "custom_components.librus_synergia.calendar_sync",
    "custom_components.librus_synergia.catch_up",
    "custom_components.librus_synergia.condition",
    "custom_components.librus_synergia.config_flow",
    "custom_components.librus_synergia.coordinator",
    "custom_components.librus_synergia.diagnostics",
    "custom_components.librus_synergia.event",
    "custom_components.librus_synergia.exam_prep",
    "custom_components.librus_synergia.forecast",
    "custom_components.librus_synergia.llm_api",
    "custom_components.librus_synergia.logbook",
    "custom_components.librus_synergia.messaging",
    "custom_components.librus_synergia.other_calendars",
    "custom_components.librus_synergia.repairs",
    "custom_components.librus_synergia.retry",
    "custom_components.librus_synergia.school_day",
    "custom_components.librus_synergia.security",
    "custom_components.librus_synergia.semester_summary",
    "custom_components.librus_synergia.sensor",
    "custom_components.librus_synergia.services",
    "custom_components.librus_synergia.subject_colors",
    "custom_components.librus_synergia.subject_emoji",
    "custom_components.librus_synergia.subject_names",
    "custom_components.librus_synergia.switch",
    "custom_components.librus_synergia.system_health",
    "custom_components.librus_synergia.todo",
    "custom_components.librus_synergia.trigger",
)


def register_credential_filter(logger: logging.Logger | None = None) -> None:
    """Register CredentialFilter on the given logger or all integration sub-module loggers."""
    if logger is not None:
        if not any(isinstance(f, CredentialFilter) for f in logger.filters):
            logger.addFilter(CredentialFilter())
        return

    for name in SUBMODULE_LOGGER_NAMES:
        log = logging.getLogger(name)
        if not any(isinstance(f, CredentialFilter) for f in log.filters):
            log.addFilter(CredentialFilter())


register_credential_filter(_LOGGER)


def validate_csrf_token(
    token: str | None,
    session_secret: str | None = None,
    min_length: int = 8,
) -> str:
    """Validate that a CSRF token is present, non-empty, and structurally valid.

    If session_secret is provided, verifies that the CSRF token matches it.
    Raises LibrusUnexpectedResponseError if the token is missing, too short,
    contains illegal characters, or does not match session secret.
    """
    if token is None or not isinstance(token, str):
        raise LibrusUnexpectedResponseError("CSRF token is missing or not a string")

    token_clean = token.strip()
    if not token_clean:
        raise LibrusUnexpectedResponseError("CSRF token is empty")

    if len(token_clean) < min_length:
        raise LibrusUnexpectedResponseError(
            f"CSRF token length ({len(token_clean)}) is less than minimum ({min_length})"
        )

    if not _CSRF_TOKEN_RE.match(token_clean):
        raise LibrusUnexpectedResponseError("CSRF token contains invalid characters")

    if session_secret is not None:
        if not hmac.compare_digest(token_clean, session_secret):
            raise LibrusUnexpectedResponseError("CSRF token does not match session secret")

    return token_clean


_UNOWNED = object()


class SessionLock:
    """Re-entrant async lock manager for session management to eliminate race conditions.

    Guarantees that simultaneous requests (e.g. concurrent timetable fetches,
    message retrieval, or coordinator updates) serialize session refreshes and
    logins, preventing session collisions and cookie overwrite races.
    Re-entrant for the same asyncio task so nested session calls do not deadlock.
    Handles callers running without an active Task (asyncio.current_task() is None).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: Any = _UNOWNED
        self._count = 0

    @property
    def locked(self) -> bool:
        """Whether the lock is currently held."""
        return self._lock.locked()

    async def acquire(self) -> None:
        """Acquire the session lock, supporting re-entrancy within the same task."""
        current = asyncio.current_task()
        if current is not None and self._owner is current:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = current
        self._count = 1

    def release(self) -> None:
        """Release the session lock."""
        if not self._lock.locked():
            return
        current = asyncio.current_task()
        if self._owner is not current:
            raise RuntimeError("Cannot release lock owned by another task")
        self._count -= 1
        if self._count == 0:
            self._owner = _UNOWNED
            self._lock.release()

    async def __aenter__(self) -> SessionLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
