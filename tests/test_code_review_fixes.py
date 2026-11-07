"""Tests for code review fixes in Librus Synergia integration."""

from __future__ import annotations

import ast
import asyncio
from datetime import timedelta
import logging
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
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.librus_synergia.api_models import (
    GradeResponse,
    MeResponse,
    TeacherResponse,
)
from custom_components.librus_synergia.const import DOMAIN
from custom_components.librus_synergia.coordinator import (
    parse_grade_responses,
    parse_teacher_responses,
)
from custom_components.librus_synergia.retry import async_retry_with_backoff
from custom_components.librus_synergia.security import (
    CredentialFilter,
    SessionLock,
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
    """Ensure cookie, auth, csrf keys are sanitized via substring search."""
    payload = {
        "user_cookie_header": "cookie_val",
        "client_auth_code": "auth_val",
        "custom_csrf_value": "csrf_val",
        "normal_field": "safe",
    }
    sanitized = sanitize_payload(payload)
    assert sanitized["user_cookie_header"] == "***REDACTED***"
    assert sanitized["client_auth_code"] == "***REDACTED***"
    assert sanitized["custom_csrf_value"] == "***REDACTED***"
    assert sanitized["normal_field"] == "safe"


def test_credential_filter_redacts_log_records() -> None:
    """Ensure CredentialFilter strips sensitive credentials from log records."""
    cred_filter = CredentialFilter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="Login request with payload: %s",
        args=({"pass": "supersecret", "username": "admin"},),
        exc_info=None,
    )
    assert cred_filter.filter(record) is True
    args = record.args if isinstance(record.args, dict) else record.args[0]
    assert args["pass"] == "***REDACTED***"
    assert args["username"] == "admin"


def test_credential_filter_regex_word_boundaries_and_unquoted() -> None:
    """Ensure CredentialFilter regex respects word boundaries and scrubs unquoted values."""
    cred_filter = CredentialFilter()

    # Quoted values
    rec1 = logging.LogRecord("test", logging.INFO, "t.py", 1, 'Connecting with password="supersecret"', (), None)
    cred_filter.filter(rec1)
    assert rec1.msg == 'Connecting with password="***REDACTED***"'

    # Unquoted values
    rec2 = logging.LogRecord("test", logging.INFO, "t.py", 1, "Connecting with password=supersecret status=ok", (), None)
    cred_filter.filter(rec2)
    assert rec2.msg == "Connecting with password=***REDACTED*** status=ok"

    rec3 = logging.LogRecord("test", logging.INFO, "t.py", 1, "pass: supersecret, user: admin", (), None)
    cred_filter.filter(rec3)
    assert rec3.msg == "pass: ***REDACTED***, user: admin"

    # Word boundaries (should NOT redact non-password/pass words)
    rec4 = logging.LogRecord("test", logging.INFO, "t.py", 1, 'compass: "north", bypass=123, passed=True', (), None)
    cred_filter.filter(rec4)
    assert rec4.msg == 'compass: "north", bypass=123, passed=True'


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
async def test_session_lock_reentrant_none_current_task() -> None:
    """Ensure SessionLock supports re-entrant acquisition when current_task is None without deadlocking."""
    lock = SessionLock()
    with patch("asyncio.current_task", return_value=None):
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

    # Test GradeResponse
    grade_dict = {
        "Id": 501,
        "Grade": "5",
        "Weight": 2.5,
        "IsConstituent": True,
        "Category": {"Id": 10},
        "Subject": {"Id": 20},
    }
    grade = GradeResponse.from_dict(grade_dict)
    assert grade.id == 501
    assert grade.grade == "5"
    assert grade.weight == 2.5
    assert grade.is_constituent is True
    assert grade.category_id == 10
    assert grade.subject_id == 20

    # Test TeacherResponse
    teacher_dict = {
        "Id": 501,
        "FirstName": "Anna",
        "LastName": "Nowak",
    }
    teacher = TeacherResponse.from_dict(teacher_dict)
    assert teacher.id == 501
    assert teacher.first_name == "Anna"
    assert teacher.last_name == "Nowak"
    assert teacher.full_name == "Anna Nowak"

    # Test from_dict(None) and from_dict({}) safe defaults
    for model_cls in (GradeResponse, TeacherResponse):
        none_instance = model_cls.from_dict(None)
        assert isinstance(none_instance, model_cls)
        assert none_instance.id == ""

        empty_instance = model_cls.from_dict({})
        assert isinstance(empty_instance, model_cls)
        assert empty_instance.id == ""


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
    device = dr.async_get(hass).async_get_device_by_identifier((DOMAIN, entry.entry_id), entry.entry_id)

    # 1. Unauthenticated client -> 401 Unauthorized (HA built-in session auth)
    http_no_auth = await hass_client_no_auth()
    resp_unauth = await http_no_auth.get(f"/api/{DOMAIN}/attachment/{device.id}/99/55")
    assert resp_unauth.status == 401

    # 2. Authenticated client -> 200 OK without any integration-level CSRF token
    http = await hass_client()
    resp_valid = await http.get(f"/api/{DOMAIN}/attachment/{device.id}/99/55")
    assert resp_valid.status == 200


def test_coordinator_exception_tuples_include_cancelled_error() -> None:
    """Verify that coordinator message send catches BaseException (including asyncio.CancelledError)."""
    import inspect

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )

    source = inspect.getsource(LibrusDataUpdateCoordinator.async_send_message)
    assert "BaseException" in source or "asyncio.CancelledError" in source


def test_coordinator_parsers_use_dataclass_models() -> None:
    """Verify coordinator parser helpers use api_models.py dataclasses."""
    grades_payload = {
        "Grades": [
            {"Id": 1, "Grade": "5", "Weight": 2.0, "Category": {"Id": 10}, "Subject": {"Id": 20}}
        ]
    }
    parsed_grades = parse_grade_responses(grades_payload)
    assert len(parsed_grades) == 1
    assert isinstance(parsed_grades[0], GradeResponse)
    assert parsed_grades[0].grade == "5"

    teachers_payload = {
        "Teachers": [{"Id": 301, "FirstName": "Anna", "LastName": "Nowak"}]
    }
    parsed_teachers = parse_teacher_responses(teachers_payload)
    assert len(parsed_teachers) == 1
    assert isinstance(parsed_teachers[0], TeacherResponse)
    assert parsed_teachers[0].full_name == "Anna Nowak"


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

    from custom_components.librus_synergia.coordinator import LibrusDataUpdateCoordinator
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

    from custom_components.librus_synergia.coordinator import LibrusDataUpdateCoordinator
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

    for name in security.SUBMODULE_LOGGER_NAMES:
        log = logging.getLogger(name)
        assert any(
            isinstance(f, CredentialFilter) for f in log.filters
        ), f"CredentialFilter not registered on {name}"

