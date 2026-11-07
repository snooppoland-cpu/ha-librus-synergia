"""Security utilities for Librus Synergia: credential protection and session locking."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
from collections.abc import Mapping
from typing import Any

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


CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    # Authorization: Bearer in dict/JSON quotes
    (
        r"(\bAuthorization['\"]?\s*:\s*['\"]Bearer\s+)([^'\"]*?)(['\"])",
        r"\g<1>***REDACTED***\g<3>",
    ),
    # Authorization: Bearer with quoted token
    (
        r"(\bAuthorization\s*:\s*Bearer\s+)(['\"])[^'\"]*?\2",
        r"\g<1>\g<2>***REDACTED***\g<2>",
    ),
    # Authorization: Bearer unquoted
    (
        r"(\bAuthorization\s*:\s*Bearer\s+)(?!['\"])([^\s,;&'\")\]\}]+)",
        r"\g<1>***REDACTED***",
    ),
    # Key-value credential patterns (quoted values)
    (
        r"(\b(?:access_token|refresh_token|client_secret|password|pass|token|cookie|secret|auth|csrf_token|_csrf|csrf)['\"]?\s*[:=]\s*)(['\"])[^'\"]*?\2",
        r"\g<1>\g<2>***REDACTED***\g<2>",
    ),
    # Key-value credential patterns (unquoted values)
    (
        r"(\b(?:access_token|refresh_token|client_secret|password|pass|token|cookie|secret|auth|csrf_token|_csrf|csrf)['\"]?\s*[:=]\s*)(?!['\"])([^\s,;&'\")\]\}]+)",
        r"\g<1>***REDACTED***",
    ),
)


def _sanitize_string(text: str) -> str:
    """Sanitize sensitive credentials in a string using credential patterns."""
    for pattern, repl in CREDENTIAL_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _sanitize_arg(arg: Any) -> Any:
    """Sanitize an argument for logging."""
    if isinstance(arg, str):
        return _sanitize_string(arg)
    if isinstance(arg, BaseException):
        return _sanitize_string(str(arg))
    if isinstance(arg, (Mapping, list, tuple)):
        return sanitize_payload(arg)
    return arg


def sanitize_payload(payload: Any) -> Any:
    """Sanitize sensitive credentials from data structures before logging.

    Replaces passwords, auth tokens, cookies, and secret values with REDACTED.
    """
    if isinstance(payload, Mapping):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key).lower()
            if key_str in SENSITIVE_KEYS or any(
                secret in key_str for secret in ("password", "secret", "token", "cookie", "csrf")
            ):
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
            elif isinstance(record.args, tuple):
                record.args = tuple(_sanitize_arg(arg) for arg in record.args)
            elif isinstance(record.args, list):
                record.args = [_sanitize_arg(arg) for arg in record.args]
            elif isinstance(record.args, str):
                record.args = _sanitize_string(record.args)
        try:
            rendered = record.getMessage()
        except Exception:  # noqa: BLE001
            rendered = str(record.msg)
        record.msg = _sanitize_string(rendered)
        record.args = None
        if getattr(record, "exc_text", None):
            record.exc_text = _sanitize_string(str(record.exc_text))
        return True


SUBMODULE_LOGGER_NAMES: tuple[str, ...] = (
    "librus_synergia",
    "librus_synergia.client",
    "librus_synergia._dates",
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


_caller_context: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "session_lock_caller_sentinel", default=None
)
_UNOWNED = object()


class SessionLock:
    """Re-entrant async lock manager for session management to eliminate race conditions.

    Guarantees that simultaneous requests (e.g. concurrent timetable fetches,
    message retrieval, or coordinator updates) serialize session refreshes and
    logins, preventing session collisions and cookie overwrite races.
    Re-entrant for the same asyncio task so nested session calls do not deadlock.
    Handles callers running without an active Task (asyncio.current_task() is None)
    via contextvars unique sentinel per caller.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: Any = _UNOWNED
        self._count = 0

    @property
    def locked(self) -> bool:
        """Whether the lock is currently held."""
        return self._lock.locked()

    def _get_caller(self) -> Any:
        task = asyncio.current_task()
        if task is not None:
            return task
        sentinel = _caller_context.get()
        if sentinel is None:
            sentinel = object()
            _caller_context.set(sentinel)
        return sentinel

    async def acquire(self) -> None:
        """Acquire the session lock, supporting re-entrancy within the same task or caller."""
        caller = self._get_caller()
        if caller is not _UNOWNED and self._owner is caller:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = caller
        self._count = 1

    def release(self) -> None:
        """Release the session lock."""
        if not self._lock.locked():
            return
        caller = self._get_caller()
        if self._owner is not caller:
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
