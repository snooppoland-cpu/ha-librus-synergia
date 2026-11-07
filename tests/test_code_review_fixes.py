"""Tests for code review fixes in Librus Synergia integration."""

from __future__ import annotations

import ast
import asyncio
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
    LibrusUnexpectedResponseError,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.librus_synergia.const import DOMAIN
from custom_components.librus_synergia.coordinator import (
    parse_attendance_responses,
    parse_grade_responses,
    parse_lesson_responses,
    parse_teacher_responses,
    wrap_api_response,
)
from custom_components.librus_synergia.api_models import (
    ApiResponse,
    AttendanceResponse,
    GradeResponse,
    LessonResponse,
    MeResponse,
    TeacherResponse,
)
from custom_components.librus_synergia.retry import async_retry_with_backoff
from custom_components.librus_synergia.security import (
    CredentialFilter,
    SessionLock,
    sanitize_payload,
    validate_csrf_token,
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


# 2) CSRF Token Validation Tests
def test_validate_csrf_token_success() -> None:
    """Ensure valid CSRF tokens pass validation."""
    valid_token = "aB3_cD9-eF012345"
    assert validate_csrf_token(valid_token) == valid_token


@pytest.mark.parametrize(
    "invalid_token",
    [
        None,
        "",
        "   ",
        "short",  # < 8 chars
        "invalid!token@#$",  # invalid characters
    ],
)
def test_validate_csrf_token_failure(invalid_token: str | None) -> None:
    """Ensure invalid CSRF tokens raise LibrusUnexpectedResponseError."""
    with pytest.raises(LibrusUnexpectedResponseError):
        validate_csrf_token(invalid_token)


def test_validate_csrf_token_with_session_secret() -> None:
    """Ensure CSRF tokens are matched against session secret."""
    secret = "aB3_cD9-eF012345"
    assert validate_csrf_token(secret, session_secret=secret) == secret

    with pytest.raises(LibrusUnexpectedResponseError, match="CSRF token does not match session secret"):
        validate_csrf_token(secret, session_secret="different_secret_999")


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


# 4) Specific Exceptions Tests (AST inspection)
def test_no_bare_except_in_codebase() -> None:
    """Verify that custom_components/librus_synergia has no bare except handlers."""
    repo_dir = Path(__file__).resolve().parent.parent / "custom_components" / "librus_synergia"
    py_files = list(repo_dir.glob("*.py"))
    assert len(py_files) > 0

    bare_excepts: list[str] = []
    broad_excepts: list[str] = []

    for file_path in py_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts.append(f"{file_path.name}:{node.lineno} (bare except)")
                elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
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

    # Test LessonResponse
    lesson_dict = {
        "Id": 301,
        "LessonNo": 1,
        "Date": "2026-11-06",
        "Subject": {"Id": 20},
        "Teacher": {"Id": 40},
        "Classroom": {"Id": 15},
    }
    lesson = LessonResponse.from_dict(lesson_dict)
    assert lesson.id == 301
    assert lesson.lesson_no == 1
    assert lesson.subject_id == 20
    assert lesson.teacher_id == 40
    assert lesson.classroom_id == 15

    # Test ApiResponse container
    api_resp = ApiResponse(status="ok", data=grade)
    assert api_resp.status == "ok"
    assert api_resp.data == grade


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
    """Verify that coordinator message send catches asyncio.CancelledError."""
    import inspect

    from custom_components.librus_synergia.coordinator import (
        LibrusDataUpdateCoordinator,
    )

    source = inspect.getsource(LibrusDataUpdateCoordinator.async_send_message)
    assert "asyncio.CancelledError" in source


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

    timetable_payload = {
        "Timetable": {
            "2026-11-06": [{"Id": 101, "LessonNo": 1, "Subject": {"Id": 20}}]
        }
    }
    parsed_lessons = parse_lesson_responses(timetable_payload)
    assert len(parsed_lessons) == 1
    assert isinstance(parsed_lessons[0], LessonResponse)
    assert parsed_lessons[0].id == 101

    attendances_payload = {
        "Attendances": [{"Id": 201, "LessonNo": 1, "Type": {"Id": 2}}]
    }
    parsed_attendances = parse_attendance_responses(attendances_payload)
    assert len(parsed_attendances) == 1
    assert isinstance(parsed_attendances[0], AttendanceResponse)
    assert parsed_attendances[0].id == 201

    teachers_payload = {
        "Teachers": [{"Id": 301, "FirstName": "Anna", "LastName": "Nowak"}]
    }
    parsed_teachers = parse_teacher_responses(teachers_payload)
    assert len(parsed_teachers) == 1
    assert isinstance(parsed_teachers[0], TeacherResponse)
    assert parsed_teachers[0].full_name == "Anna Nowak"

    wrapped = wrap_api_response(parsed_grades)
    assert isinstance(wrapped, ApiResponse)
    assert wrapped.data == parsed_grades


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
        logger = getattr(mod, "_LOGGER")
        assert any(
            isinstance(f, CredentialFilter) for f in logger.filters
        ), f"CredentialFilter not registered on {mod.__name__}._LOGGER"

    for name in security.SUBMODULE_LOGGER_NAMES:
        log = logging.getLogger(name)
        assert any(
            isinstance(f, CredentialFilter) for f in log.filters
        ), f"CredentialFilter not registered on {name}"

