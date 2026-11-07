"""Data models (dataclasses) for Librus Synergia API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

T = TypeVar("T")


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


@dataclass(slots=True)
class GradeResponse:
    """Data model representing a grade returned by the API."""

    id: int | str
    lesson_id: int | str | None = None
    subject_id: int | str | None = None
    grade: str = ""
    category_id: int | str | None = None
    date: str = ""
    add_date: str = ""
    weight: float = 1.0
    is_constituent: bool = False
    is_semester: bool = False
    is_final: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GradeResponse:
        """Create GradeResponse from API dictionary."""
        lesson = data.get("Lesson")
        lesson_id = lesson.get("Id") if isinstance(lesson, Mapping) else lesson

        subject = data.get("Subject")
        subject_id = subject.get("Id") if isinstance(subject, Mapping) else subject

        category = data.get("Category")
        category_id = category.get("Id") if isinstance(category, Mapping) else category

        raw_weight = data.get("Weight", 1.0)
        try:
            weight = float(raw_weight) if raw_weight is not None else 1.0
        except (ValueError, TypeError):
            weight = 1.0

        return cls(
            id=data.get("Id", ""),
            lesson_id=lesson_id,
            subject_id=subject_id,
            grade=str(data.get("Grade", "")),
            category_id=category_id,
            date=str(data.get("Date", "")),
            add_date=str(data.get("AddDate", "")),
            weight=weight,
            is_constituent=bool(data.get("IsConstituent", False)),
            is_semester=bool(data.get("IsSemester", False)),
            is_final=bool(data.get("IsFinal", False)),
        )


@dataclass(slots=True)
class LessonResponse:
    """Data model representing a timetable lesson returned by the API."""

    id: int | str
    lesson_no: int | None = None
    date: str = ""
    subject_id: int | str | None = None
    teacher_id: int | str | None = None
    classroom_id: int | str | None = None
    is_canceled: bool = False
    is_substitution: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LessonResponse:
        """Create LessonResponse from API dictionary."""
        subject = data.get("Subject")
        subject_id = subject.get("Id") if isinstance(subject, Mapping) else subject

        teacher = data.get("Teacher")
        teacher_id = teacher.get("Id") if isinstance(teacher, Mapping) else teacher

        classroom = data.get("Classroom")
        classroom_id = classroom.get("Id") if isinstance(classroom, Mapping) else classroom

        return cls(
            id=data.get("Id", ""),
            lesson_no=data.get("LessonNo"),
            date=str(data.get("Date", "")),
            subject_id=subject_id,
            teacher_id=teacher_id,
            classroom_id=classroom_id,
            is_canceled=bool(data.get("IsCanceled", False)),
            is_substitution=bool(data.get("IsSubstitution", False)),
        )


@dataclass(slots=True)
class AttendanceResponse:
    """Data model representing attendance record returned by the API."""

    id: int | str
    lesson_no: int | None = None
    date: str = ""
    type_id: int | str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttendanceResponse:
        """Create AttendanceResponse from API dictionary."""
        type_obj = data.get("Type")
        type_id = type_obj.get("Id") if isinstance(type_obj, Mapping) else type_obj
        return cls(
            id=data.get("Id", ""),
            lesson_no=data.get("LessonNo"),
            date=str(data.get("Date", "")),
            type_id=type_id,
        )


@dataclass(slots=True)
class TeacherResponse:
    """Data model representing a teacher returned by the API."""

    id: int | str
    first_name: str = ""
    last_name: str = ""

    @property
    def full_name(self) -> str:
        """Full name of the teacher."""
        return f"{self.first_name} {self.last_name}".strip()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TeacherResponse:
        """Create TeacherResponse from API dictionary."""
        return cls(
            id=data.get("Id", ""),
            first_name=str(data.get("FirstName") or ""),
            last_name=str(data.get("LastName") or ""),
        )


@dataclass(slots=True)
class ApiResponse(Generic[T]):
    """Generic wrapper for typed API responses."""

    status: str = "ok"
    data: T | None = None
    message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
