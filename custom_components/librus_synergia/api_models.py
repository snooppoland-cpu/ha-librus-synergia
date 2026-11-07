"""Data models (dataclasses) for Librus Synergia API responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AccountInfo:
    """User account details from the Librus /Me endpoint."""

    id: int | str | None = None
    user_id: int | str | None = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    group_id: int | str | None = None
    login: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> AccountInfo:
        """Create an AccountInfo safely from a dictionary."""
        if not data or not isinstance(data, Mapping):
            return cls()
        return cls(
            id=data.get("Id"),
            user_id=data.get("UserId"),
            first_name=str(data.get("FirstName") or ""),
            last_name=str(data.get("LastName") or ""),
            email=str(data.get("Email") or ""),
            group_id=data.get("GroupId"),
            login=str(data.get("Login") or ""),
        )


@dataclass(slots=True)
class UserInfo:
    """Student/User details from the Librus /Me endpoint."""

    first_name: str = ""
    last_name: str = ""
    class_register_number: int | str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> UserInfo:
        """Create a UserInfo safely from a dictionary."""
        if not data or not isinstance(data, Mapping):
            return cls()
        return cls(
            first_name=str(data.get("FirstName") or ""),
            last_name=str(data.get("LastName") or ""),
            class_register_number=data.get("ClassRegisterNumber"),
        )


@dataclass(slots=True)
class MeResponse:
    """Parsed response data model for the /Me endpoint."""

    account: AccountInfo = field(default_factory=AccountInfo)
    user: UserInfo = field(default_factory=UserInfo)
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def student_name(self) -> str:
        """Full name of the student or account holder."""
        name = f"{self.user.first_name} {self.user.last_name}".strip()
        if name:
            return name
        return f"{self.account.first_name} {self.account.last_name}".strip()

    @property
    def display_title(self) -> str:
        """Friendly display title for Home Assistant integration."""
        name = self.student_name
        return f"E-dziennik {name}" if name else ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> MeResponse:
        """Parse raw /Me payload into typed MeResponse dataclass."""
        if not data or not isinstance(data, Mapping):
            return cls()
        me = data.get("Me", {}) if isinstance(data, Mapping) else {}
        if not isinstance(me, Mapping):
            me = {}
        account = AccountInfo.from_dict(me.get("Account"))
        user = UserInfo.from_dict(me.get("User"))
        return cls(
            account=account,
            user=user,
            raw_data=dict(data) if isinstance(data, Mapping) else {},
        )
