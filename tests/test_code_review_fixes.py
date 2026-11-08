"""Tests for code review fixes in Librus Synergia integration."""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.data_entry_flow import FlowResultType
from librus_synergia.exceptions import (
    LibrusAccountActionRequiredError,
    LibrusConnectionError,
    LibrusInvalidCredentialsError,
    LibrusSessionExpiredError,
)
from librus_synergia.models import GradeData
from librus_synergia.parsers import parse_grades
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.librus_synergia.api_models import (
    MeResponse,
)
from custom_components.librus_synergia.const import DOMAIN
from custom_components.librus_synergia.coordinator import (
    _referenced_ids,
)
from custom_components.librus_synergia.retry import async_retry_with_backoff
from custom_components.librus_synergia.security import (
    CredentialFilter,
    SessionLock,
    _sanitize_arg,
    sanitize_payload,
)

from .conftest import build_mock_client


# 1) Credentials and Auth Payload Redaction Tests
def test_sanitize_payload_redacts_credentials() -> None:
    """Ensure sensitive credentials are redacted from payloads."""
    payload = {
        "action": "login",
        "login": "student123",
        "pass": "secret_password",
        "nested": {
            "token": "secret_token_abc",
            "safe_field": "safe_value",
            "cookies": [{"name": "DeviceCookie", "value": "xyz"}],
        },
    }
    sanitized = sanitize_payload(payload)
    assert sanitized["pass"] == "***REDACTED***"
    assert sanitized["nested"]["token"] == "***REDACTED***"
    assert sanitized["nested"]["cookies"] == "***REDACTED***"
    assert sanitized["nested"]["safe_field"] == "safe_value"
    assert sanitized["login"] == "student123"


def test_sanitize_payload_substring_keys() -> None:
    """Ensure cookie, csrf, secret keys are sanitized via substring search, but auth is exact only."""
    payload = {
        "user_cookie_header": "cookie_val",
        "custom_csrf_value": "csrf_val",
        "my_secret_key": "secret_val",
        "normal_field": "safe",
        "author": "John Doe",
        "author_id": 42,
        "auth": "secret_auth",
        "authorization": "Bearer token",
    }
    sanitized = sanitize_payload(payload)
    assert sanitized["user_cookie_header"] == "***REDACTED***"
    assert sanitized["custom_csrf_value"] == "***REDACTED***"
    assert sanitized["my_secret_key"] == "***REDACTED***"
    assert sanitized["normal_field"] == "safe"
    assert sanitized["author"] == "John Doe"
    assert sanitized["author_id"] == 42
    assert sanitized["auth"] == "***REDACTED***"
    assert sanitized["authorization"] == "***REDACTED***"


def test_credential_filter_redacts_log_records() -> None:
    """Ensure CredentialFilter strips sensitive credentials from log records and renders getMessage()."""
    cred_filter = CredentialFilter()
    # Test dictionary in args
    record1 = logging.LogRecord(
        name="test_logger",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="Login request with payload: %s",
        args=({"pass": "supersecret", "username": "admin"},),
        exc_info=None,
    )
    assert cred_filter.filter(record1) is True
    assert "***REDACTED***" in record1.getMessage()
    assert "supersecret" not in record1.getMessage()
    assert "admin" in record1.getMessage()

    # Test string args in record.args (tuple, list, and single string)
    record2 = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Request url: %s and header %s",
        args=("https://librus.pl?token=supersecret123&cookie=abc456", "password=adminpass"),
        exc_info=None,
    )
    assert cred_filter.filter(record2) is True
    assert "token=***REDACTED***" in record2.getMessage()
    assert "cookie=***REDACTED***" in record2.getMessage()
    assert "password=***REDACTED***" in record2.getMessage()
    assert "supersecret123" not in record2.getMessage()
    assert "abc456" not in record2.getMessage()
    assert "adminpass" not in record2.getMessage()

    record3 = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Single string arg: %s",
        args="secret=super_secret_val",
        exc_info=None,
    )
    assert cred_filter.filter(record3) is True
    assert record3.getMessage() == "Single string arg: secret=***REDACTED***"


def test_sanitize_arg_handles_base_exception() -> None:
    """Ensure _sanitize_arg redacts credentials from BaseException instances."""
    exc = Exception("Connection failed: token=abc12345 password=secretpass")
    sanitized = _sanitize_arg(exc)
    assert isinstance(sanitized, str)
    assert sanitized == "Connection failed: token=***REDACTED*** password=***REDACTED***"

    base_exc = BaseException("Critical error: auth: my_secret_token cookie=testcookie")
    sanitized_base = _sanitize_arg(base_exc)
    assert isinstance(sanitized_base, str)
    assert sanitized_base == "Critical error: auth: ***REDACTED*** cookie=***REDACTED***"

    # Also test through CredentialFilter with record.args having an exception
    cred_filter = CredentialFilter()
    record = logging.LogRecord(
        name="librus_synergia.client",
        level=logging.ERROR,
        pathname="client.py",
        lineno=42,
        msg="API error: %s",
        args=(exc,),
        exc_info=None,
    )
    assert cred_filter.filter(record) is True
    assert record.getMessage() == "API error: Connection failed: token=***REDACTED*** password=***REDACTED***"


def test_credential_filter_renders_and_expands_regex() -> None:
    """Ensure CredentialFilter covers access_token, refresh_token, Authorization: Bearer, client_secret."""
    cred_filter = CredentialFilter()

    # Formatted %s with Authorization: Bearer in msg and token in args
    rec1 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1, "Authorization: Bearer %s", ("secret_bearer_token",), None
    )
    cred_filter.filter(rec1)
    assert rec1.getMessage() == "Authorization: Bearer ***REDACTED***"

    # Quoted Bearer header in dict representation
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1, "Headers: {'Authorization': 'Bearer token_abc123'}", (), None
    )
    cred_filter.filter(rec2)
    assert rec2.getMessage() == "Headers: {'Authorization': 'Bearer ***REDACTED***'}"

    # access_token and refresh_token
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1, 'access_token="tok123"&refresh_token="ref456"', (), None
    )
    cred_filter.filter(rec3)
    assert rec3.getMessage() == 'access_token="***REDACTED***"&refresh_token="***REDACTED***"'

    # client_secret
    rec4 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1, "client_secret=super_secret_client_val", (), None
    )
    cred_filter.filter(rec4)
    assert rec4.getMessage() == "client_secret=***REDACTED***"


def test_credential_filter_regex_word_boundaries_and_unquoted() -> None:
    """Ensure CredentialFilter regex covers token, cookie, secret, auth, csrf and respects word boundaries."""
    cred_filter = CredentialFilter()

    # Quoted values
    rec1 = logging.LogRecord("test", logging.INFO, "t.py", 1, 'Connecting with password="supersecret"', (), None)
    cred_filter.filter(rec1)
    assert rec1.msg == 'Connecting with password="***REDACTED***"'

    # Unquoted values for password, token, cookie, secret, auth, csrf
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        "token=tok123 cookie=cook456 secret=sec789 auth: authval csrf=csrf999", (), None
    )
    cred_filter.filter(rec2)
    assert rec2.msg == "token=***REDACTED*** cookie=***REDACTED*** secret=***REDACTED*** auth: ***REDACTED*** csrf=***REDACTED***"

    # Quoted values for token, cookie, secret, auth, csrf
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'token="tok123" cookie=\'cook456\' secret="sec789" auth: "authval" csrf=\'csrf999\'', (), None
    )
    cred_filter.filter(rec3)
    assert rec3.msg == 'token="***REDACTED***" cookie=\'***REDACTED***\' secret="***REDACTED***" auth: "***REDACTED***" csrf=\'***REDACTED***\''

    # Word boundaries (should NOT redact non-credential words like compass, bypass, author, passed)
    rec4 = logging.LogRecord("test", logging.INFO, "t.py", 1, 'compass: "north", bypass=123, passed=True, author="Adam"', (), None)
    cred_filter.filter(rec4)
    assert rec4.msg == 'compass: "north", bypass=123, passed=True, author="Adam"'


def test_credential_filter_covers_csrf_token_and_underscore_csrf() -> None:
    """Ensure CredentialFilter handles csrf_token and underscore-prefixed _csrf."""
    cred_filter = CredentialFilter()

    # Quoted and unquoted csrf_token and _csrf
    rec1 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'csrf_token="secret_csrf_1" _csrf=\'secret_csrf_2\' csrf_token=secret_csrf_3 _csrf=secret_csrf_4',
        (), None,
    )
    cred_filter.filter(rec1)
    assert rec1.msg == (
        'csrf_token="***REDACTED***" _csrf=\'***REDACTED***\' '
        'csrf_token=***REDACTED*** _csrf=***REDACTED***'
    )

    # In JSON string with quotes around key and value
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        '{"_csrf": "my_secret_token", "csrf_token": "another_secret"}',
        (), None,
    )
    cred_filter.filter(rec2)
    assert rec2.msg == '{"_csrf": "***REDACTED***", "csrf_token": "***REDACTED***"}'

    # Prefixed credentials like foo_csrf and test_csrf_token are redacted
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'foo_csrf=safe and test_csrf_token="safe_val" but _csrf="secret" and csrf_token=secret',
        (), None,
    )
    cred_filter.filter(rec3)
    assert rec3.msg == 'foo_csrf=***REDACTED*** and test_csrf_token="***REDACTED***" but _csrf="***REDACTED***" and csrf_token=***REDACTED***'


def test_credential_filter_sanitizes_record_exc_text() -> None:
    """Ensure CredentialFilter sanitizes record.exc_text when present."""
    cred_filter = CredentialFilter()

    # exc_text present with sensitive credentials
    rec1 = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Error occurred during login",
        args=(),
        exc_info=None,
    )
    rec1.exc_text = (
        "Traceback (most recent call last):\n"
        "  File 'client.py', line 50, in login\n"
        "ValueError: failed with password='secret_pwd' token=secret_token _csrf=secret_csrf csrf_token='secret_csrf_tok'\n"
    )

    assert cred_filter.filter(rec1) is True
    assert rec1.exc_text is not None
    assert "secret_pwd" not in rec1.exc_text
    assert "secret_token" not in rec1.exc_text
    assert "secret_csrf" not in rec1.exc_text
    assert "secret_csrf_tok" not in rec1.exc_text
    assert "password='***REDACTED***'" in rec1.exc_text
    assert "token=***REDACTED***" in rec1.exc_text
    assert "_csrf=***REDACTED***" in rec1.exc_text
    assert "csrf_token='***REDACTED***'" in rec1.exc_text

    # exc_text is None
    rec2 = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=11,
        msg="Normal message",
        args=(),
        exc_info=None,
    )
    assert rec2.exc_text is None
    assert cred_filter.filter(rec2) is True
    assert rec2.exc_text is None

    # exc_info is present, exc_text is None initially (use Formatter.formatException)
    try:
        raise ValueError("Failed auth with password='secret_pwd' cookies='cookie_val' authorization=auth_token_val")
    except ValueError:
        rec3 = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Exception during request",
            args=(),
            exc_info=sys.exc_info(),
        )

    assert rec3.exc_text is None
    assert cred_filter.filter(rec3) is True
    assert rec3.exc_text is not None
    assert "secret_pwd" not in rec3.exc_text
    assert "cookie_val" not in rec3.exc_text
    assert "auth_token_val" not in rec3.exc_text
    assert "password='***REDACTED***'" in rec3.exc_text
    assert "cookies='***REDACTED***'" in rec3.exc_text
    assert "authorization=***REDACTED***" in rec3.exc_text

    # Formatter output also redacts traceback
    formatter = logging.Formatter()
    formatted_msg = formatter.format(rec3)
    assert "secret_pwd" not in formatted_msg
    assert "cookie_val" not in formatted_msg
    assert "auth_token_val" not in formatted_msg


def test_credential_filter_covers_cookies_and_authorization() -> None:
    """Ensure CredentialFilter handles cookies and authorization with proper word boundaries."""
    cred_filter = CredentialFilter()

    # Quoted and unquoted cookies
    rec1 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'cookies="sess_1" cookies=\'sess_2\' cookies=sess_3 cookies: sess_4',
        (), None,
    )
    cred_filter.filter(rec1)
    assert rec1.msg == (
        'cookies="***REDACTED***" cookies=\'***REDACTED***\' '
        'cookies=***REDACTED*** cookies: ***REDACTED***'
    )

    # Quoted and unquoted authorization
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'authorization="tok_1" authorization=\'tok_2\' authorization=tok_3 authorization: tok_4',
        (), None,
    )
    cred_filter.filter(rec2)
    assert rec2.msg == (
        'authorization="***REDACTED***" authorization=\'***REDACTED***\' '
        'authorization=***REDACTED*** authorization: ***REDACTED***'
    )

    # Word boundaries: authorized, cookies_jar, author should not be redacted
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'authorized=True, cookies_jar=5, author="Adam", passed=True',
        (), None,
    )
    cred_filter.filter(rec3)
    assert rec3.msg == 'authorized=True, cookies_jar=5, author="Adam", passed=True'


def test_credential_filter_unquoted_scalar_excludes_brackets_and_braces() -> None:
    """Ensure unquoted scalar regex excludes [ and { to prevent stringified JSON lists leaking nested tokens."""
    cred_filter = CredentialFilter()

    # Nested token in stringified JSON list should be properly redacted without destroying list structure
    rec1 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        '{"token": [{"id": 1, "token": "secret_nested_token"}]}',
        (), None,
    )
    cred_filter.filter(rec1)
    assert "secret_nested_token" not in rec1.msg
    assert rec1.msg == '{"token": [{"id": 1, "token": "***REDACTED***"}]}'

    # Cookies array of dicts in stringified JSON should not match cookies as unquoted scalar
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        '{"cookies": [{"name": "auth_cookie", "token": "cookie_secret"}]}',
        (), None,
    )
    cred_filter.filter(rec2)
    assert "cookie_secret" not in rec2.msg
    assert rec2.msg == '{"cookies": [{"name": "auth_cookie", "token": "***REDACTED***"}]}'

    # Unquoted scalar tokens still get redacted
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'token=unquoted_scalar_secret_val',
        (), None,
    )
    cred_filter.filter(rec3)
    assert rec3.msg == 'token=***REDACTED***'


def test_credential_filter_prefixed_credentials_and_query_strings() -> None:
    """Ensure prefixed credentials like oauth_token=, api_token=, session_token= and tokens in query strings are matched."""
    cred_filter = CredentialFilter()

    # Prefixed credentials unquoted and quoted
    rec1 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'api_token=secret_api_123 session_token="secret_sess_456" auth_token=\'secret_auth_789\' oauth_token="secret_oauth"',
        (), None,
    )
    cred_filter.filter(rec1)
    assert rec1.msg == (
        'api_token=***REDACTED*** session_token="***REDACTED***" '
        'auth_token=\'***REDACTED***\' oauth_token="***REDACTED***"'
    )

    # Query strings containing tokens and prefixed tokens
    rec2 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'GET https://api.librus.pl/v3/users?api_token=xyz123&session_token=abc456&page=1',
        (), None,
    )
    cred_filter.filter(rec2)
    assert rec2.msg == 'GET https://api.librus.pl/v3/users?api_token=***REDACTED***&session_token=***REDACTED***&page=1'

    # Plain tokens in query strings
    rec3 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'Request: https://synergia.librus.pl/login?token=login_tok_999&other=1',
        (), None,
    )
    cred_filter.filter(rec3)
    assert rec3.msg == 'Request: https://synergia.librus.pl/login?token=***REDACTED***&other=1'

    # Words like bypass and compass are not falsely redacted
    rec4 = logging.LogRecord(
        "test", logging.INFO, "t.py", 1,
        'bypass=123 compass: "north" surpass=456',
        (), None,
    )
    cred_filter.filter(rec4)
    assert rec4.msg == 'bypass=123 compass: "north" surpass=456'


# 3) Session Management Locking Tests
@pytest.mark.asyncio
async def test_session_lock_serializes_concurrent_tasks() -> None:
    """Ensure SessionLock serializes concurrent session operations without races."""
    lock = SessionLock()
    execution_order: list[str] = []

    async def task_worker(name: str, delay: float) -> None:
        async with lock:
            execution_order.append(f"{name}_start")
            await asyncio.sleep(delay)
            execution_order.append(f"{name}_end")

    task1 = asyncio.create_task(task_worker("task1", 0.02))
    task2 = asyncio.create_task(task_worker("task2", 0.01))
    await asyncio.gather(task1, task2)

    # Locking ensures task1 finishes before task2 starts or vice versa
    assert execution_order in (
        ["task1_start", "task1_end", "task2_start", "task2_end"],
        ["task2_start", "task2_end", "task1_start", "task1_end"],
    )


@pytest.mark.asyncio
async def test_session_lock_aenter_returns_self() -> None:
    """Ensure SessionLock.__aenter__ returns self."""
    lock = SessionLock()
    async with lock as ctx:
        assert ctx is lock


@pytest.mark.asyncio
async def test_session_lock_handles_none_current_task() -> None:
    """Ensure SessionLock works correctly when asyncio.current_task() is None."""
    lock = SessionLock()
    with patch("asyncio.current_task", return_value=None):
        assert not lock.locked
        async with lock:
            assert lock.locked
        assert not lock.locked

        # Release when unlocked does not raise
        lock.release()


@pytest.mark.asyncio
async def test_session_lock_none_current_task_release_conflict() -> None:
    """Ensure releasing lock owned by another task raises RuntimeError."""
    lock = SessionLock()
    await lock.acquire()  # Acquired with real task
    try:
        with (
            patch("asyncio.current_task", return_value=None),
            pytest.raises(RuntimeError, match="Cannot release lock owned by another task"),
        ):
            lock.release()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_session_lock_reentrant_same_task() -> None:
    """Ensure SessionLock supports re-entrant acquisition within the same task."""
    lock = SessionLock()
    assert not lock.locked
    async with lock:
        assert lock.locked
        assert lock._count == 1
        async with lock:
            assert lock.locked
            assert lock._count == 2
        assert lock.locked
        assert lock._count == 1
    assert not lock.locked
    assert lock._count == 0


@pytest.mark.asyncio
async def test_session_lock_reentrant_when_current_task_is_none() -> None:
    """Ensure SessionLock supports re-entrant acquisition and avoids deadlock when current_task is None."""
    lock = SessionLock()
    with patch("asyncio.current_task", return_value=None):
        await lock.acquire()
        assert lock.locked
        assert lock._count == 1
        # Re-entrant acquire must succeed without deadlock
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
        assert lock.locked
        assert lock._count == 2
        lock.release()
        assert lock.locked
        assert lock._count == 1
        lock.release()
        assert not lock.locked
        assert lock._count == 0


@pytest.mark.asyncio
async def test_session_lock_none_current_task_cross_release_prevented() -> None:
    """Ensure callers with distinct contextvars contexts cannot release each other's lock when current_task is None."""
    import contextvars

    lock = SessionLock()
    with patch("asyncio.current_task", return_value=None):
        await lock.acquire()
        assert lock.locked

        ctx2 = contextvars.Context()
        with pytest.raises(RuntimeError, match="Cannot release lock owned by another task"):
            ctx2.run(lock.release)

        assert lock.locked
        lock.release()
        assert not lock.locked


@pytest.mark.asyncio
async def test_session_lock_release_unowned_task_conflict() -> None:
    """Ensure a different task cannot release another task's lock."""
    lock = SessionLock()
    await lock.acquire()

    async def unauthorized_release() -> None:
        lock.release()

    task = asyncio.create_task(unauthorized_release())
    with pytest.raises(RuntimeError, match="Cannot release lock owned by another task"):
        await task
    lock.release()


# 4) Specific Exceptions Tests (AST inspection)
def test_no_bare_except_in_codebase() -> None:
    """Verify that custom_components/librus_synergia has no bare except handlers."""
    repo_dir = Path(__file__).resolve().parent.parent / "custom_components" / "librus_synergia"
    py_files = list(repo_dir.glob("*.py"))
    assert len(py_files) > 0

    bare_excepts: list[str] = []
    broad_excepts: list[str] = []

    for file_path in py_files:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        tree = ast.parse("\n".join(lines), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts.append(f"{file_path.name}:{node.lineno} (bare except)")
                elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                    line = lines[node.lineno - 1]
                    if "# noqa: BLE001" not in line and "noqa: BLE001" not in line:
                        broad_excepts.append(f"{file_path.name}:{node.lineno} (except {node.type.id})")

    assert not bare_excepts, f"Found bare excepts: {bare_excepts}"
    assert not broad_excepts, f"Found broad excepts: {broad_excepts}"


# 5) Retry Logic with Backoff Tests
@pytest.mark.asyncio
async def test_retry_with_backoff_success_after_transient_failure() -> None:
    """Ensure transient network errors are retried with backoff and succeed."""
    attempts = 0

    async def transient_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LibrusConnectionError("Temporary network glitch")
        return "success"

    result = await async_retry_with_backoff(
        transient_operation,
        max_retries=3,
        initial_delay=0.01,
        backoff_factor=1.5,
    )
    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_fails_non_retryable_immediately() -> None:
    """Ensure non-retryable errors (e.g. invalid credentials) fail immediately."""
    attempts = 0

    async def auth_failure_operation() -> str:
        nonlocal attempts
        attempts += 1
        raise LibrusInvalidCredentialsError("Wrong password")

    with pytest.raises(LibrusInvalidCredentialsError):
        await async_retry_with_backoff(
            auth_failure_operation,
            max_retries=3,
            initial_delay=0.01,
        )
    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_exhaustion() -> None:
    """Ensure retryable errors retry until max_retries is exhausted, then raise."""
    attempts = 0

    async def persistently_failing_operation() -> str:
        nonlocal attempts
        attempts += 1
        raise LibrusConnectionError("Persistent network outage")

    with pytest.raises(LibrusConnectionError, match="Persistent network outage"):
        await async_retry_with_backoff(
            persistently_failing_operation,
            max_retries=3,
            initial_delay=0.001,
        )
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_enforces_min_max_retries() -> None:
    """Ensure max_retries = max(1, max_retries) executes at least once even if <= 0."""
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        return "ok"

    result = await async_retry_with_backoff(op, max_retries=0)
    assert result == "ok"
    assert attempts == 1

    attempts = 0
    result_neg = await async_retry_with_backoff(op, max_retries=-5)
    assert result_neg == "ok"
    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_client_response_error_status_filtering() -> None:
    """Ensure ClientResponseError only retries 429 and 5xx, and fails immediately on 4xx."""
    from unittest.mock import MagicMock

    import aiohttp

    # 404 Not Found should fail immediately on attempt 1
    attempts_404 = 0

    async def op_404() -> str:
        nonlocal attempts_404
        attempts_404 += 1
        raise aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=404)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await async_retry_with_backoff(op_404, max_retries=3, initial_delay=0.001)
    assert exc_info.value.status == 404
    assert attempts_404 == 1

    # 401 Unauthorized should fail immediately on attempt 1
    attempts_401 = 0

    async def op_401() -> str:
        nonlocal attempts_401
        attempts_401 += 1
        raise aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=401)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await async_retry_with_backoff(op_401, max_retries=3, initial_delay=0.001)
    assert exc_info.value.status == 401
    assert attempts_401 == 1

    # 429 Too Many Requests should be retried and succeed on attempt 2
    attempts_429 = 0

    async def op_429() -> str:
        nonlocal attempts_429
        attempts_429 += 1
        if attempts_429 == 1:
            raise aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=429)
        return "rate_limited_then_ok"

    res_429 = await async_retry_with_backoff(op_429, max_retries=3, initial_delay=0.001)
    assert res_429 == "rate_limited_then_ok"
    assert attempts_429 == 2

    # 503 Service Unavailable should be retried and succeed on attempt 2
    attempts_503 = 0

    async def op_503() -> str:
        nonlocal attempts_503
        attempts_503 += 1
        if attempts_503 == 1:
            raise aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=503)
        return "server_error_then_ok"

    res_503 = await async_retry_with_backoff(op_503, max_retries=3, initial_delay=0.001)
    assert res_503 == "server_error_then_ok"
    assert attempts_503 == 2

    # Unexpected loop exit raises RuntimeError
    async def dummy_op() -> str:
        return "ok"

    with (
        patch("custom_components.librus_synergia.retry.range", return_value=[]),
        pytest.raises(RuntimeError, match="Unexpected end of retry loop"),
    ):
        await async_retry_with_backoff(dummy_op)


@pytest.mark.asyncio
async def test_retry_with_backoff_applies_random_jitter() -> None:
    """Ensure async_retry_with_backoff applies random jitter (0.8-1.2) to sleep delays."""
    attempts = 0
    slept_delays: list[float] = []

    async def failing_op() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LibrusConnectionError("Transient drop")
        return "recovered"

    with patch(
        "custom_components.librus_synergia.retry.asyncio.sleep",
        side_effect=lambda d: slept_delays.append(d),
    ):
        result = await async_retry_with_backoff(
            failing_op,
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
        )

    assert result == "recovered"
    assert attempts == 3
    assert len(slept_delays) == 2
    # First sleep base delay is 1.0, jitter between 0.8 and 1.2
    assert 0.8 <= slept_delays[0] <= 1.2
    # Second sleep base delay is 2.0, jitter between 1.6 and 2.4
    assert 1.6 <= slept_delays[1] <= 2.4


# 6) Data Models (Dataclasses) Tests
def test_data_models_dataclasses() -> None:
    """Verify dataclass response models parse correctly and safely."""
    me_dict = {
        "Me": {
            "Account": {
                "Id": 1001,
                "UserId": 2002,
                "FirstName": "Jan",
                "LastName": "Kowalski",
                "Email": "jan@example.com",
                "Login": "1234567u",
            },
            "User": {
                "FirstName": "Ola",
                "LastName": "Kowalska",
                "ClassRegisterNumber": 12,
            },
        }
    }
    me_resp = MeResponse.from_dict(me_dict)
    assert isinstance(me_resp, MeResponse)
    assert me_resp.student_name == "Ola Kowalska"
    assert me_resp.display_title == "E-dziennik Ola Kowalska"
    assert me_resp.account.id == 1001
    assert me_resp.user.class_register_number == 12

    # Test safe defaults for MeResponse
    none_instance = MeResponse.from_dict(None)
    assert isinstance(none_instance, MeResponse)
    empty_instance = MeResponse.from_dict({})
    assert isinstance(empty_instance, MeResponse)

    # Verify upstream librus_synergia.models GradeData
    grade = GradeData(
        id=501,
        value="5",
        category_id=10,
        subject_id=20,
        semester=1,
        add_date="2026-09-01",
        is_semester_proposition=False,
        is_final_proposition=False,
    )
    assert grade.id == 501
    assert grade.value == "5"
    assert grade.category_id == 10
    assert grade.subject_id == 20


# 7) Re-auth Flow Tests
@pytest.mark.asyncio
async def test_reauth_flow_account_action_required(hass) -> None:
    """Ensure account action required is surfaced in reauth flow."""
    user_input = {CONF_USERNAME: "1234567u", CONF_PASSWORD: "pw"}
    entry = MockConfigEntry(domain=DOMAIN, unique_id="1234567", data=user_input)
    entry.add_to_hass(hass)
    client = build_mock_client()
    client.async_login.side_effect = LibrusAccountActionRequiredError("Must accept terms on website")

    with patch(
        "custom_components.librus_synergia.config_flow.LibrusApiClient", return_value=client
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_REAUTH,
                "entry_id": entry.entry_id,
                "unique_id": entry.unique_id,
            },
            data=entry.data,
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PASSWORD: "new-password"}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "account_action_required"}


@pytest.mark.asyncio
async def test_coordinator_async_start_reauth(hass) -> None:
    """Ensure coordinator has async_start_reauth helper method."""
    user_input = {CONF_USERNAME: "1234567u", CONF_PASSWORD: "pw"}
    entry = MockConfigEntry(domain=DOMAIN, unique_id="1234567", data=user_input)
    entry.add_to_hass(hass)
    client = build_mock_client()

    from datetime import timedelta

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )

    coordinator = LibrusDataUpdateCoordinator(hass, entry, client, timedelta(minutes=15))
    with patch.object(entry, "async_start_reauth") as mock_start:
        coordinator.async_start_reauth()
        mock_start.assert_called_once_with(hass)


@pytest.mark.asyncio
async def test_attachment_view_requires_auth(hass, hass_client, hass_client_no_auth) -> None:
    """Ensure attachment endpoints rely on Home Assistant built-in session auth."""
    from homeassistant.helpers import device_registry as dr
    from librus_synergia.models import AttachmentFileData

    from tests.test_topics_trips_documents import setup_integration

    client = build_mock_client()
    client.async_download_message_attachment.return_value = AttachmentFileData(
        filename="test.pdf", content_type="application/pdf", content=b"%PDF"
    )
    entry = await setup_integration(hass, client)
    device_id = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0].id

    # 1. Unauthenticated client -> 401 Unauthorized (HA built-in session auth)
    http_no_auth = await hass_client_no_auth()
    resp_unauth = await http_no_auth.get(f"/api/{DOMAIN}/attachment/{device_id}/99/55")
    assert resp_unauth.status == 401

    # 2. Authenticated client -> 200 OK without any integration-level CSRF token
    http = await hass_client()
    resp_valid = await http.get(f"/api/{DOMAIN}/attachment/{device_id}/99/55")
    assert resp_valid.status == 200


@pytest.mark.asyncio
async def test_homework_attachment_view(hass, hass_client, hass_client_no_auth) -> None:
    """Ensure homework attachment view requires auth, validates ids, and streams file."""
    from homeassistant.helpers import device_registry as dr
    from librus_synergia.models import AttachmentFileData

    from tests.test_topics_trips_documents import setup_integration

    client = build_mock_client()
    client.async_download_homework_attachment.return_value = AttachmentFileData(
        filename="zadanie.pdf", content_type="application/pdf", content=b"%PDF-homework"
    )
    entry = await setup_integration(hass, client)
    device_id = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0].id

    # 1. Unauthenticated -> 401
    http_no_auth = await hass_client_no_auth()
    resp_unauth = await http_no_auth.get(f"/api/{DOMAIN}/homework_attachment/{device_id}/42")
    assert resp_unauth.status == 401

    # 2. Authenticated -> 200 with file content
    http = await hass_client()
    resp_valid = await http.get(f"/api/{DOMAIN}/homework_attachment/{device_id}/42")
    assert resp_valid.status == 200
    assert await resp_valid.read() == b"%PDF-homework"
    assert "zadanie.pdf" in resp_valid.headers["Content-Disposition"]
    client.async_download_homework_attachment.assert_awaited_once_with("42")

    # 3. Invalid ID -> 404
    resp_invalid = await http.get(f"/api/{DOMAIN}/homework_attachment/{device_id}/invalid..id")
    assert resp_invalid.status == 404


@pytest.mark.asyncio
async def test_school_file_view(hass, hass_client, hass_client_no_auth) -> None:
    """Ensure school file view requires auth, validates ids, and streams file."""
    from homeassistant.helpers import device_registry as dr
    from librus_synergia.models import AttachmentFileData

    from tests.test_topics_trips_documents import FILES, setup_integration

    client = build_mock_client(async_get_school_files=FILES)
    client.async_download_school_file.return_value = AttachmentFileData(
        filename="Regulamin.pdf", content_type="application/pdf", content=b"%PDF-school"
    )
    entry = await setup_integration(hass, client)
    device_id = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0].id

    # 1. Unauthenticated -> 401
    http_no_auth = await hass_client_no_auth()
    resp_unauth = await http_no_auth.get(f"/api/{DOMAIN}/school_file/{device_id}/17613")
    assert resp_unauth.status == 401

    # 2. Authenticated -> 200 with file content
    http = await hass_client()
    resp_valid = await http.get(f"/api/{DOMAIN}/school_file/{device_id}/17613")
    assert resp_valid.status == 200
    assert await resp_valid.read() == b"%PDF-school"
    assert "Regulamin.pdf" in resp_valid.headers["Content-Disposition"]
    client.async_download_school_file.assert_awaited_once_with("/pliki_szkoly/pobierz/1")

    # 3. Invalid ID -> 404
    resp_invalid = await http.get(f"/api/{DOMAIN}/school_file/{device_id}/invalid..id")
    assert resp_invalid.status == 404

    # 4. Unknown file ID -> 404
    resp_unknown = await http.get(f"/api/{DOMAIN}/school_file/{device_id}/999")
    assert resp_unknown.status == 404


@pytest.mark.asyncio
async def test_attachment_view_error_logging_redacts_credentials_via_filter(
    hass, hass_client, caplog
) -> None:
    """Ensure attachment view relies on CredentialFilter to redact credentials in logged errors."""
    from homeassistant.helpers import device_registry as dr

    from tests.test_topics_trips_documents import setup_integration

    client = build_mock_client()
    client.async_download_message_attachment.side_effect = LibrusConnectionError(
        "Download failed at https://synergia.librus.pl/file?cookie=secret_session_cookie&token=secret_dl_token&_csrf=secret_csrf"
    )
    entry = await setup_integration(hass, client)
    device_id = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0].id
    http = await hass_client()

    with caplog.at_level(logging.WARNING):
        resp = await http.get(f"/api/{DOMAIN}/attachment/{device_id}/99/55")
        assert resp.status == 502

    log_text = caplog.text
    assert "Attachment download failed" in log_text
    assert "secret_session_cookie" not in log_text
    assert "secret_dl_token" not in log_text
    assert "secret_csrf" not in log_text
    assert "cookie=***REDACTED***" in log_text
    assert "token=***REDACTED***" in log_text
    assert "_csrf=***REDACTED***" in log_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise",
    [
        "client_error",
        "timeout_error",
        "os_error",
    ],
)
async def test_attachment_view_catches_network_exceptions(
    hass, hass_client, exception_to_raise: str
) -> None:
    """Ensure attachment view catches aiohttp.ClientError, TimeoutError, and OSError returning 502."""
    import aiohttp
    from homeassistant.helpers import device_registry as dr

    from tests.test_topics_trips_documents import setup_integration

    err_map = {
        "client_error": aiohttp.ClientError("network failure"),
        "timeout_error": TimeoutError("timeout exceeded"),
        "os_error": OSError("network unreachable"),
    }

    client = build_mock_client()
    client.async_download_message_attachment.side_effect = err_map[exception_to_raise]
    entry = await setup_integration(hass, client)
    device_id = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0].id
    http = await hass_client()

    resp = await http.get(f"/api/{DOMAIN}/attachment/{device_id}/99/55")
    assert resp.status == 502
    assert "download failed" in (await resp.text()).lower()


def test_coordinator_exception_tuples_include_cancelled_error() -> None:
    """Verify that coordinator message send catches BaseException (including asyncio.CancelledError)."""
    import inspect

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )

    source = inspect.getsource(LibrusDataUpdateCoordinator.async_send_message)
    assert "BaseException" in source or "asyncio.CancelledError" in source


def test_coordinator_grades_reference_ids_use_upstream_grade_data() -> None:
    """Verify coordinator _referenced_ids uses upstream librus_synergia.models GradeData."""
    grades_payload = {
        "Grades": [
            {"Id": 1, "Grade": "5", "Weight": 2.0, "Category": {"Id": 10}, "Subject": {"Id": 20}},
            {"Id": 2, "Grade": "4", "Weight": 1.0, "Category": {"Id": 11}, "Subject": {"Id": 21}},
        ]
    }
    parsed = parse_grades(grades_payload)
    assert len(parsed) == 2
    assert isinstance(parsed[0], GradeData)
    assert parsed[0].id == 1
    assert parsed[0].category_id == 10
    assert parsed[0].subject_id == 20

    cat_ids = _referenced_ids(grades_payload, "Grades", "Category")
    subj_ids = _referenced_ids(grades_payload, "Grades", "Subject")
    assert cat_ids == {10, 11}
    assert subj_ids == {20, 21}


@pytest.mark.asyncio
async def test_coordinator_retry_wired(hass) -> None:
    """Verify coordinator relogin uses retry with backoff."""
    from datetime import timedelta

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )

    user_input = {CONF_USERNAME: "1234567u", CONF_PASSWORD: "pw"}
    entry = MockConfigEntry(domain=DOMAIN, unique_id="1234567", data=user_input)
    entry.add_to_hass(hass)
    client = build_mock_client()

    coordinator = LibrusDataUpdateCoordinator(hass, entry, client, timedelta(minutes=15))
    with patch(
        "custom_components.librus_synergia.coordinator.async_retry_with_backoff",
        wraps=async_retry_with_backoff,
    ) as mock_retry:
        await coordinator._async_relogin(seen=0)
        assert mock_retry.called


@pytest.mark.asyncio
async def test_coordinator_concurrent_expiration_mailbox_bootstrap(hass) -> None:
    """Verify concurrent session expirations only bootstrap mailbox once and downloads run outside lock."""
    user_input = {CONF_USERNAME: "1234567u", CONF_PASSWORD: "pw"}
    entry = MockConfigEntry(domain=DOMAIN, unique_id="1234567", data=user_input)
    entry.add_to_hass(hass)
    client = build_mock_client()

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )
    coordinator = LibrusDataUpdateCoordinator(hass, entry, client, timedelta(minutes=15))

    bootstrap_count = 0
    async def mock_bootstrap():
        nonlocal bootstrap_count
        bootstrap_count += 1
        return True

    client.async_bootstrap_messages.side_effect = mock_bootstrap

    lock_held_during_fetch: list[bool] = []
    async def mock_get_message(mailbox, msg_id):
        await asyncio.sleep(0.01)
        if coordinator._login_count == 0:
            raise LibrusSessionExpiredError("Expired")
        lock_held_during_fetch.append(coordinator.session_lock.locked)
        return {"id": msg_id, "body": "content"}

    client.async_get_message.side_effect = mock_get_message

    res1, res2 = await asyncio.gather(
        coordinator.async_fetch_message("inbox", "msg1"),
        coordinator.async_fetch_message("inbox", "msg2"),
    )

    assert res1 == {"id": "msg1", "body": "content"}
    assert res2 == {"id": "msg2", "body": "content"}
    assert bootstrap_count == 1
    assert all(locked is False for locked in lock_held_during_fetch)


@pytest.mark.asyncio
async def test_coordinator_download_attachment_concurrent_expiration(hass) -> None:
    """Verify concurrent attachment downloads only bootstrap mailbox once and download runs outside lock."""
    user_input = {CONF_USERNAME: "1234567u", CONF_PASSWORD: "pw"}
    entry = MockConfigEntry(domain=DOMAIN, unique_id="1234567", data=user_input)
    entry.add_to_hass(hass)
    client = build_mock_client()

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )
    coordinator = LibrusDataUpdateCoordinator(hass, entry, client, timedelta(minutes=15))

    bootstrap_count = 0
    async def mock_bootstrap():
        nonlocal bootstrap_count
        bootstrap_count += 1
        return True

    client.async_bootstrap_messages.side_effect = mock_bootstrap

    lock_held_during_download: list[bool] = []
    async def mock_download(att_id, msg_id):
        await asyncio.sleep(0.01)
        if coordinator._login_count == 0:
            raise LibrusSessionExpiredError("Expired")
        lock_held_during_download.append(coordinator.session_lock.locked)
        return b"filecontent"

    client.async_download_message_attachment.side_effect = mock_download

    res1, res2 = await asyncio.gather(
        coordinator.async_download_attachment("att1", "msg1"),
        coordinator.async_download_attachment("att2", "msg2"),
    )

    assert res1 == b"filecontent"
    assert res2 == b"filecontent"
    assert bootstrap_count == 1
    assert all(locked is False for locked in lock_held_during_download)


def test_credential_filter_registered_on_all_submodule_loggers() -> None:
    """Verify CredentialFilter is registered on all sub-module loggers."""
    from custom_components.librus_synergia import (
        ai_summary,
        attachment_view,
        average_history,
        calendar,
        calendar_sync,
        config_flow,
        coordinator,
        llm_api,
        other_calendars,
        retry,
        security,
        services,
        trigger,
    )

    submodules = [
        coordinator,
        config_flow,
        services,
        attachment_view,
        ai_summary,
        average_history,
        calendar,
        calendar_sync,
        llm_api,
        other_calendars,
        retry,
        security,
        trigger,
    ]

    for mod in submodules:
        logger = mod._LOGGER
        assert any(
            isinstance(f, CredentialFilter) for f in logger.filters
        ), f"CredentialFilter not registered on {mod.__name__}._LOGGER"

    assert "librus_synergia" in security.SUBMODULE_LOGGER_NAMES
    assert "librus_synergia.client" in security.SUBMODULE_LOGGER_NAMES
    assert "librus_synergia._dates" in security.SUBMODULE_LOGGER_NAMES

    for name in security.SUBMODULE_LOGGER_NAMES:
        log = logging.getLogger(name)
        assert any(
            isinstance(f, CredentialFilter) for f in log.filters
        ), f"CredentialFilter not registered on {name}"


# 7) Schema Rejection Fallback Tests
def test_is_schema_error() -> None:
    """Verify _is_schema_error detects ServiceValidationError and Invalid class names."""
    from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

    from custom_components.librus_synergia.ai_summary import _is_schema_error

    class Invalid(Exception):
        pass

    class SubInvalid(Invalid):
        pass

    assert _is_schema_error(ServiceValidationError("Schema validation error")) is True
    assert _is_schema_error(Invalid("Invalid structure schema")) is True
    assert _is_schema_error(SubInvalid("Invalid sub-schema")) is True
    assert _is_schema_error(ValueError("Invalid argument")) is False
    assert _is_schema_error(TypeError("Bad type")) is False
    assert _is_schema_error(HomeAssistantError("Service failed")) is False


@pytest.mark.asyncio
async def test_weekly_summary_schema_rejection_fallback(hass) -> None:
    """Verify weekly summary retries without structure when service raises ServiceValidationError."""
    from homeassistant.core import SupportsResponse
    from homeassistant.exceptions import ServiceValidationError

    from custom_components.librus_synergia.const import (
        CONF_AI_AUDIENCE,
        CONF_AI_TASK_ENTITY,
        CONF_AI_TIME,
        CONF_AI_WEEKDAY,
    )
    from tests.test_topics_trips_documents import setup_integration

    ai_options = {
        CONF_AI_TASK_ENTITY: "ai_task.mock",
        CONF_AI_AUDIENCE: "parent",
        CONF_AI_WEEKDAY: "7",
        CONF_AI_TIME: "18:00:00",
    }
    calls = []

    async def _handler(call):
        calls.append(dict(call.data))
        if "structure" in call.data:
            raise ServiceValidationError("Extra keys not allowed @ data['structure']")
        return {
            "data": {
                "headline": "Spokojny tydzień",
                "status": "ok",
                "grades": "Brak nowych ocen.",
                "grades_status": "ok",
                "summary": "Wszystko w porządku.",
            }
        }

    hass.services.async_register(
        "ai_task", "generate_data", _handler, supports_response=SupportsResponse.ONLY
    )

    client = build_mock_client()
    entry = await setup_integration(hass, client, options=ai_options)
    coordinator = entry.runtime_data

    await coordinator.weekly_summary.async_generate(wait=True)

    assert len(calls) == 2
    assert "structure" in calls[0]
    assert "structure" not in calls[1]
    assert coordinator.weekly_summary.results["parent"]["headline"] == "Spokojny tydzień"


@pytest.mark.asyncio
async def test_weekly_summary_non_schema_error_does_not_fallback(hass) -> None:
    """Verify weekly summary does not retry when service raises non-schema errors."""
    from homeassistant.core import SupportsResponse
    from homeassistant.exceptions import HomeAssistantError

    from custom_components.librus_synergia.const import (
        CONF_AI_AUDIENCE,
        CONF_AI_TASK_ENTITY,
        CONF_AI_TIME,
        CONF_AI_WEEKDAY,
    )
    from tests.test_topics_trips_documents import setup_integration

    ai_options = {
        CONF_AI_TASK_ENTITY: "ai_task.mock",
        CONF_AI_AUDIENCE: "parent",
        CONF_AI_WEEKDAY: "7",
        CONF_AI_TIME: "18:00:00",
    }
    calls = []

    async def _handler(call):
        calls.append(dict(call.data))
        raise HomeAssistantError("Model backend unreachable")

    hass.services.async_register(
        "ai_task", "generate_data", _handler, supports_response=SupportsResponse.ONLY
    )

    client = build_mock_client()
    entry = await setup_integration(hass, client, options=ai_options)
    coordinator = entry.runtime_data

    with pytest.raises(HomeAssistantError, match="Model backend unreachable"):
        await coordinator.weekly_summary.async_generate(wait=True)

    assert len(calls) == 1
    assert "structure" in calls[0]


@pytest.mark.asyncio
async def test_weekly_summary_catches_aiohttp_client_error(hass) -> None:
    """Verify weekly summary catches aiohttp.ClientError and raises HomeAssistantError."""
    import aiohttp
    from homeassistant.core import SupportsResponse
    from homeassistant.exceptions import HomeAssistantError

    from custom_components.librus_synergia.const import (
        CONF_AI_AUDIENCE,
        CONF_AI_TASK_ENTITY,
        CONF_AI_TIME,
        CONF_AI_WEEKDAY,
    )
    from tests.test_topics_trips_documents import setup_integration

    ai_options = {
        CONF_AI_TASK_ENTITY: "ai_task.mock",
        CONF_AI_AUDIENCE: "parent",
        CONF_AI_WEEKDAY: "7",
        CONF_AI_TIME: "18:00:00",
    }

    async def _handler(call):
        raise aiohttp.ClientError("Cannot connect to AI service host")

    hass.services.async_register(
        "ai_task", "generate_data", _handler, supports_response=SupportsResponse.ONLY
    )

    client = build_mock_client()
    entry = await setup_integration(hass, client, options=ai_options)
    coordinator = entry.runtime_data

    with pytest.raises(HomeAssistantError, match="Weekly summary failed: Cannot connect to AI service host"):
        await coordinator.weekly_summary.async_generate(wait=True)

    assert coordinator.weekly_summary.last_error == "Cannot connect to AI service host"


def test_classvar_annotations() -> None:
    """Verify ClassVar typing annotations on forecast._ALL."""
    from custom_components.librus_synergia.forecast import DataMemo

    assert "ClassVar" in str(DataMemo.__annotations__["_ALL"])


def test_attachment_view_file_response_csp() -> None:
    """Verify Content-Security-Policy header in _file_response."""
    from unittest.mock import MagicMock

    from custom_components.librus_synergia.attachment_view import _file_response

    mock_file = MagicMock()
    mock_file.filename = "test.pdf"
    mock_file.content = b"content"
    mock_file.content_type = "application/pdf"

    resp = _file_response(mock_file, "fallback.pdf")
    assert resp.headers.get("Content-Security-Policy") == "default-src 'none'; sandbox"


@pytest.mark.asyncio
async def test_coordinator_read_receipts_handles_non_dict_payload(hass) -> None:
    """Verify _async_refresh_read_receipts defensively ignores non-dict payloads."""
    from librus_synergia.models import MessageData

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )
    from tests.test_topics_trips_documents import setup_integration

    client = build_mock_client()
    entry = await setup_integration(hass, client)
    coordinator: LibrusDataUpdateCoordinator = entry.runtime_data

    sent_msg = MessageData(
        "msg_non_dict", "Sender", "Topic", "Content", "2026-11-01T10:00:00", None, False
    )
    client.async_get_message.return_value = "invalid-payload-not-a-dict"

    await coordinator._async_refresh_read_receipts([sent_msg])
    assert "msg_non_dict" not in coordinator.read_receipts



