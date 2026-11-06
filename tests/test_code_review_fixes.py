"""Tests for code review fixes in Librus Synergia integration."""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.librus_synergia.const import DOMAIN
from custom_components.librus_synergia.models import (
    AccountInfo,
    ApiResponse,
    AttendanceResponse,
    GradeResponse,
    LessonResponse,
    MeResponse,
    TeacherResponse,
    UserInfo,
)
from custom_components.librus_synergia.retry import async_retry_with_backoff
from custom_components.librus_synergia.security import (
    CredentialFilter,
    SessionLock,
    sanitize_payload,
    validate_csrf_token,
)
from librus_synergia.client import LibrusSessionData
from librus_synergia.exceptions import (
    LibrusAccountActionRequiredError,
    LibrusCaptchaRequiredError,
    LibrusConnectionError,
    LibrusInvalidCredentialsError,
    LibrusUnexpectedResponseError,
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

    from custom_components.librus_synergia.coordinator import LibrusDataUpdateCoordinator
    from datetime import timedelta

    coordinator = LibrusDataUpdateCoordinator(hass, entry, client, timedelta(minutes=15))
    with patch.object(entry, "async_start_reauth") as mock_start:
        coordinator.async_start_reauth()
        mock_start.assert_called_once_with(hass)
