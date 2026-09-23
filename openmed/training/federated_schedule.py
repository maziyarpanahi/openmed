"""Deterministic UTC scheduling windows for federated training rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Final, Mapping

FEDERATED_SCHEDULE_SCHEMA_VERSION = "openmed.training.federated_schedule.v1"
MAX_FEDERATED_PHASE_DURATION_SECONDS = 365 * 24 * 60 * 60


class FederatedSchedulePhase(str, Enum):
    """A bounded participation phase in chronological order."""

    ENROLLMENT = "enrollment"
    UPDATE_SUBMISSION = "update_submission"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"

    def __str__(self) -> str:
        return self.value


FEDERATED_SCHEDULE_PHASES: Final[tuple[FederatedSchedulePhase, ...]] = tuple(
    FederatedSchedulePhase
)


class FederatedScheduleError(ValueError):
    """Raised when a federated schedule is malformed or unsupported."""


@dataclass(frozen=True)
class FederatedRoundSchedule:
    """Immutable phase boundaries with no participant or network metadata."""

    enrollment_starts_at: datetime
    update_submission_starts_at: datetime
    aggregation_starts_at: datetime
    evaluation_starts_at: datetime
    finishes_at: datetime
    max_enrollment_duration_seconds: int | None = None
    max_update_submission_duration_seconds: int | None = None
    max_aggregation_duration_seconds: int | None = None
    max_evaluation_duration_seconds: int | None = None
    schema_version: str = FEDERATED_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_SCHEDULE_SCHEMA_VERSION
        ):
            raise FederatedScheduleError("unsupported federated schedule schema")

        boundary_names = (
            "enrollment_starts_at",
            "update_submission_starts_at",
            "aggregation_starts_at",
            "evaluation_starts_at",
            "finishes_at",
        )
        for name in boundary_names:
            object.__setattr__(self, name, _require_utc(getattr(self, name), name))

        for previous_name, current_name in zip(boundary_names, boundary_names[1:]):
            if getattr(self, current_name) <= getattr(self, previous_name):
                raise FederatedScheduleError(
                    f"{current_name} must be later than {previous_name}"
                )

        boundaries = self._boundaries()
        duration_limits = self._duration_limits()
        for (phase, start, end), limit in zip(boundaries, duration_limits):
            field = _maximum_field(phase)
            _require_optional_duration(limit, field)
            duration = end - start
            if duration > timedelta(seconds=MAX_FEDERATED_PHASE_DURATION_SECONDS):
                raise FederatedScheduleError(
                    f"{_end_field(phase)} exceeds the supported phase duration"
                )
            if limit is not None and duration > timedelta(seconds=limit):
                raise FederatedScheduleError(f"{_end_field(phase)} exceeds {field}")

    def active_phase_at(self, timestamp: datetime) -> FederatedSchedulePhase | None:
        """Return the active half-open phase at a caller-supplied UTC timestamp."""

        moment = _require_utc(timestamp, "timestamp")
        for phase, start, end in self._boundaries():
            if start <= moment < end:
                return phase
        return None

    def next_phase_at(self, timestamp: datetime) -> FederatedSchedulePhase | None:
        """Return the next phase that has not started at the supplied timestamp."""

        moment = _require_utc(timestamp, "timestamp")
        for phase, start, _ in self._boundaries():
            if moment < start:
                return phase
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned metadata-only schedule representation."""

        return {
            "boundaries": {
                "aggregation_starts_at": _format_utc(self.aggregation_starts_at),
                "enrollment_starts_at": _format_utc(self.enrollment_starts_at),
                "evaluation_starts_at": _format_utc(self.evaluation_starts_at),
                "finishes_at": _format_utc(self.finishes_at),
                "update_submission_starts_at": _format_utc(
                    self.update_submission_starts_at
                ),
            },
            "maximum_duration_seconds": {
                "aggregation": self.max_aggregation_duration_seconds,
                "enrollment": self.max_enrollment_duration_seconds,
                "evaluation": self.max_evaluation_duration_seconds,
                "update_submission": self.max_update_submission_duration_seconds,
            },
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with canonical UTC timestamps."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FederatedRoundSchedule:
        """Parse a strict schedule payload without retaining unknown metadata."""

        if not isinstance(payload, Mapping):
            raise FederatedScheduleError("invalid federated schedule payload")
        if _keys(payload) != {
            "boundaries",
            "maximum_duration_seconds",
            "schema_version",
        }:
            raise FederatedScheduleError("invalid federated schedule payload")
        if (
            type(payload["schema_version"]) is not str
            or payload["schema_version"] != FEDERATED_SCHEDULE_SCHEMA_VERSION
        ):
            raise FederatedScheduleError("unsupported federated schedule schema")

        boundaries = payload["boundaries"]
        maximums = payload["maximum_duration_seconds"]
        if not isinstance(boundaries, Mapping) or _keys(boundaries) != {
            "aggregation_starts_at",
            "enrollment_starts_at",
            "evaluation_starts_at",
            "finishes_at",
            "update_submission_starts_at",
        }:
            raise FederatedScheduleError("invalid federated schedule boundaries")
        if not isinstance(maximums, Mapping) or _keys(maximums) != {
            "aggregation",
            "enrollment",
            "evaluation",
            "update_submission",
        }:
            raise FederatedScheduleError("invalid federated schedule durations")

        return cls(
            enrollment_starts_at=_parse_utc(
                boundaries["enrollment_starts_at"], "enrollment_starts_at"
            ),
            update_submission_starts_at=_parse_utc(
                boundaries["update_submission_starts_at"],
                "update_submission_starts_at",
            ),
            aggregation_starts_at=_parse_utc(
                boundaries["aggregation_starts_at"], "aggregation_starts_at"
            ),
            evaluation_starts_at=_parse_utc(
                boundaries["evaluation_starts_at"], "evaluation_starts_at"
            ),
            finishes_at=_parse_utc(boundaries["finishes_at"], "finishes_at"),
            max_enrollment_duration_seconds=maximums["enrollment"],
            max_update_submission_duration_seconds=maximums["update_submission"],
            max_aggregation_duration_seconds=maximums["aggregation"],
            max_evaluation_duration_seconds=maximums["evaluation"],
        )

    @classmethod
    def from_json(cls, payload: str) -> FederatedRoundSchedule:
        """Parse schedule JSON while replacing parser details with a safe error."""

        if type(payload) is not str:
            raise FederatedScheduleError("invalid federated schedule JSON")
        try:
            decoded = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, FederatedScheduleError, TypeError):
            raise FederatedScheduleError("invalid federated schedule JSON") from None
        if not isinstance(decoded, Mapping):
            raise FederatedScheduleError("invalid federated schedule payload")
        return cls.from_dict(decoded)

    def _boundaries(
        self,
    ) -> tuple[tuple[FederatedSchedulePhase, datetime, datetime], ...]:
        return (
            (
                FederatedSchedulePhase.ENROLLMENT,
                self.enrollment_starts_at,
                self.update_submission_starts_at,
            ),
            (
                FederatedSchedulePhase.UPDATE_SUBMISSION,
                self.update_submission_starts_at,
                self.aggregation_starts_at,
            ),
            (
                FederatedSchedulePhase.AGGREGATION,
                self.aggregation_starts_at,
                self.evaluation_starts_at,
            ),
            (
                FederatedSchedulePhase.EVALUATION,
                self.evaluation_starts_at,
                self.finishes_at,
            ),
        )

    def _duration_limits(self) -> tuple[int | None, ...]:
        return (
            self.max_enrollment_duration_seconds,
            self.max_update_submission_duration_seconds,
            self.max_aggregation_duration_seconds,
            self.max_evaluation_duration_seconds,
        )


def _require_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise FederatedScheduleError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise FederatedScheduleError(f"{field} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise FederatedScheduleError(f"{field} must use UTC")
    return value.astimezone(timezone.utc)


def _require_optional_duration(value: object, field: str) -> None:
    if value is None:
        return
    if (
        type(value) is not int
        or value < 1
        or value > MAX_FEDERATED_PHASE_DURATION_SECONDS
    ):
        raise FederatedScheduleError(f"{field} must be a supported positive integer")


def _format_utc(value: datetime) -> str:
    canonical = _require_utc(value, "timestamp")
    timespec = "microseconds" if canonical.microsecond else "seconds"
    return canonical.isoformat(timespec=timespec).replace("+00:00", "Z")


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise FederatedScheduleError(f"{field} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise FederatedScheduleError(
            f"{field} must be a canonical UTC timestamp"
        ) from None
    if _format_utc(parsed) != value:
        raise FederatedScheduleError(f"{field} must be a canonical UTC timestamp")
    return parsed


def _keys(value: Mapping[Any, Any]) -> set[Any]:
    try:
        return set(value)
    except Exception:
        raise FederatedScheduleError("invalid federated schedule payload") from None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FederatedScheduleError("invalid federated schedule JSON")
        result[key] = value
    return result


def _end_field(phase: FederatedSchedulePhase) -> str:
    if phase is FederatedSchedulePhase.ENROLLMENT:
        return "update_submission_starts_at"
    if phase is FederatedSchedulePhase.UPDATE_SUBMISSION:
        return "aggregation_starts_at"
    if phase is FederatedSchedulePhase.AGGREGATION:
        return "evaluation_starts_at"
    return "finishes_at"


def _maximum_field(phase: FederatedSchedulePhase) -> str:
    return f"max_{phase.value}_duration_seconds"


__all__ = [
    "FEDERATED_SCHEDULE_PHASES",
    "FEDERATED_SCHEDULE_SCHEMA_VERSION",
    "MAX_FEDERATED_PHASE_DURATION_SECONDS",
    "FederatedRoundSchedule",
    "FederatedScheduleError",
    "FederatedSchedulePhase",
]
