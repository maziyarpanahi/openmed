"""Deterministic UTC scheduling windows for federated training rounds."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

FEDERATED_SCHEDULE_SCHEMA_VERSION = "openmed.training.federated_schedule.v1"
FEDERATED_SCHEDULE_V2_SCHEMA_VERSION = "openmed.training.federated_schedule.v2"
FEDERATED_SCHEDULE_SCHEMA_VERSIONS: Final[tuple[str, ...]] = (
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    FEDERATED_SCHEDULE_V2_SCHEMA_VERSION,
)
MAX_FEDERATED_PHASE_DURATION_SECONDS = 365 * 24 * 60 * 60
# A larger bound would hide an unsynchronized site rather than tolerate drift.
MAX_FEDERATED_CLOCK_SKEW_SECONDS = 15 * 60

_V1_PAYLOAD_KEYS: Final = frozenset(
    {"boundaries", "maximum_duration_seconds", "schema_version"}
)
_V2_PAYLOAD_KEYS: Final = _V1_PAYLOAD_KEYS | {"max_clock_skew_seconds"}


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


class FederatedClockBasis(str, Enum):
    """The clock reading that produced an admission decision."""

    SITE_ADVISORY = "site_advisory"
    COORDINATOR_RECEIPT = "coordinator_receipt"

    def __str__(self) -> str:
        return self.value


class FederatedAdmissionReasonCode(str, Enum):
    """Stable, categorical reasons for a refused phase admission."""

    LOCAL_CLOCK_BEFORE_WINDOW = "local_clock_before_window"
    LOCAL_CLOCK_AFTER_WINDOW = "local_clock_after_window"
    RECEIVED_BEFORE_WINDOW = "received_before_window"
    RECEIVED_AFTER_WINDOW = "received_after_window"

    def __str__(self) -> str:
        return self.value


_REASON_CODES_BY_BASIS: Final[
    Mapping[FederatedClockBasis, frozenset[FederatedAdmissionReasonCode]]
] = MappingProxyType(
    {
        FederatedClockBasis.SITE_ADVISORY: frozenset(
            {
                FederatedAdmissionReasonCode.LOCAL_CLOCK_BEFORE_WINDOW,
                FederatedAdmissionReasonCode.LOCAL_CLOCK_AFTER_WINDOW,
            }
        ),
        FederatedClockBasis.COORDINATOR_RECEIPT: frozenset(
            {
                FederatedAdmissionReasonCode.RECEIVED_BEFORE_WINDOW,
                FederatedAdmissionReasonCode.RECEIVED_AFTER_WINDOW,
            }
        ),
    }
)


@dataclass(frozen=True)
class FederatedPhaseAdmission:
    """A shareable admission decision without timestamps or clock offsets.

    Attributes:
        phase: The phase the site asked to start or submit to.
        clock_basis: ``site_advisory`` for a site's local start check, or
            ``coordinator_receipt`` for the authoritative coordinator check.
        reason_code: ``None`` when admitted, otherwise a stable refusal code.
        coordinator_sequence: The coordinator-assigned receipt sequence for a
            coordinator decision; always ``None`` for a site decision.
    """

    phase: FederatedSchedulePhase
    clock_basis: FederatedClockBasis
    reason_code: FederatedAdmissionReasonCode | None = None
    coordinator_sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, FederatedSchedulePhase):
            raise FederatedScheduleError("phase must be a federated schedule phase")
        if not isinstance(self.clock_basis, FederatedClockBasis):
            raise FederatedScheduleError("invalid admission clock basis")
        if self.reason_code is not None and (
            not isinstance(self.reason_code, FederatedAdmissionReasonCode)
            or self.reason_code not in _REASON_CODES_BY_BASIS[self.clock_basis]
        ):
            raise FederatedScheduleError("invalid admission reason code")
        if self.clock_basis is FederatedClockBasis.COORDINATOR_RECEIPT:
            _require_sequence(self.coordinator_sequence)
        elif self.coordinator_sequence is not None:
            raise FederatedScheduleError(
                "site admission cannot carry a coordinator sequence"
            )

    @property
    def admitted(self) -> bool:
        """Return whether the phase may be started or the arrival accepted."""

        return self.reason_code is None

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-only record with no site timestamp or offset."""

        return {
            "admitted": self.admitted,
            "clock_basis": self.clock_basis.value,
            "coordinator_sequence": self.coordinator_sequence,
            "phase": self.phase.value,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class FederatedRoundSchedule:
    """Immutable phase boundaries with no participant or network metadata.

    Schema v1 resolves phases from exact half-open boundaries. Schema v2 also
    declares ``max_clock_skew_seconds``, the largest tolerated difference
    between a site clock and the coordinator clock. Site clocks are advisory:
    a site starts a phase only inside the window shrunk by that bound, while
    the coordinator admits arrivals by its own receipt time alone.
    """

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
    max_clock_skew_seconds: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version not in FEDERATED_SCHEDULE_SCHEMA_VERSIONS
        ):
            raise FederatedScheduleError("unsupported federated schedule schema")
        if self.schema_version == FEDERATED_SCHEDULE_SCHEMA_VERSION:
            if self.max_clock_skew_seconds is not None:
                raise FederatedScheduleError(
                    "max_clock_skew_seconds is not supported by this schedule schema"
                )
        else:
            _require_clock_skew(self.max_clock_skew_seconds)

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
            if self.max_clock_skew_seconds is not None and duration <= timedelta(
                seconds=2 * self.max_clock_skew_seconds
            ):
                raise FederatedScheduleError(
                    f"{_end_field(phase)} leaves no client window within "
                    "max_clock_skew_seconds"
                )

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

    def client_admission_at(
        self, phase: FederatedSchedulePhase, local_time: datetime
    ) -> FederatedPhaseAdmission:
        """Decide whether a site may start ``phase`` at its local UTC time.

        The site clock is advisory. A site starts a phase only when its local
        reading lies inside the phase window shrunk by the declared skew on
        both sides, so any clock within the bound agrees the phase is open.
        The decision never retains the local reading.

        Args:
            phase: The phase the site intends to start.
            local_time: A reading from the site's injected UTC clock.

        Returns:
            A ``site_advisory`` decision. A refusal carries
            ``local_clock_before_window`` or ``local_clock_after_window``.

        Raises:
            FederatedScheduleError: If the schedule declares no skew bound,
                the phase is unknown, or the time is not an aware UTC value.
        """

        if self.max_clock_skew_seconds is None:
            raise FederatedScheduleError(
                "client admission requires max_clock_skew_seconds"
            )
        start, end = self._window(phase)
        moment = _require_utc(local_time, "local_time")
        tolerance = timedelta(seconds=self.max_clock_skew_seconds)
        reason = None
        if moment < start + tolerance:
            reason = FederatedAdmissionReasonCode.LOCAL_CLOCK_BEFORE_WINDOW
        elif moment >= end - tolerance:
            reason = FederatedAdmissionReasonCode.LOCAL_CLOCK_AFTER_WINDOW
        return FederatedPhaseAdmission(
            phase=phase,
            clock_basis=FederatedClockBasis.SITE_ADVISORY,
            reason_code=reason,
        )

    def coordinator_admission_at(
        self,
        phase: FederatedSchedulePhase,
        received_at: datetime,
        *,
        coordinator_sequence: int,
    ) -> FederatedPhaseAdmission:
        """Decide whether the coordinator accepts an arrival for ``phase``.

        The coordinator receipt time is authoritative and uses the exact
        half-open phase window. The skew bound and any site-reported time play
        no part, so a late arrival is refused by receipt time only.

        Args:
            phase: The phase the arrival belongs to.
            received_at: The coordinator's injected UTC clock at receipt.
            coordinator_sequence: The positive receipt sequence the coordinator
                assigned to the arrival; evidence is ordered by it.

        Returns:
            A ``coordinator_receipt`` decision. A refusal carries
            ``received_before_window`` or ``received_after_window``.

        Raises:
            FederatedScheduleError: If the phase is unknown, the time is not an
                aware UTC value, or the sequence is not a positive integer.
        """

        start, end = self._window(phase)
        moment = _require_utc(received_at, "received_at")
        _require_sequence(coordinator_sequence)
        reason = None
        if moment < start:
            reason = FederatedAdmissionReasonCode.RECEIVED_BEFORE_WINDOW
        elif moment >= end:
            reason = FederatedAdmissionReasonCode.RECEIVED_AFTER_WINDOW
        return FederatedPhaseAdmission(
            phase=phase,
            clock_basis=FederatedClockBasis.COORDINATOR_RECEIPT,
            reason_code=reason,
            coordinator_sequence=coordinator_sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned metadata-only schedule representation."""

        payload: dict[str, Any] = {
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
        if self.schema_version != FEDERATED_SCHEDULE_SCHEMA_VERSION:
            payload["max_clock_skew_seconds"] = self.max_clock_skew_seconds
        return payload

    def to_json(self) -> str:
        """Return byte-stable JSON with canonical UTC timestamps."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FederatedRoundSchedule:
        """Parse a strict schedule payload without retaining unknown metadata."""

        if not isinstance(payload, Mapping):
            raise FederatedScheduleError("invalid federated schedule payload")
        keys = _keys(payload)
        if keys != _V1_PAYLOAD_KEYS and keys != _V2_PAYLOAD_KEYS:
            raise FederatedScheduleError("invalid federated schedule payload")
        schema_version = payload["schema_version"]
        if (
            type(schema_version) is not str
            or schema_version not in FEDERATED_SCHEDULE_SCHEMA_VERSIONS
        ):
            raise FederatedScheduleError("unsupported federated schedule schema")
        if (schema_version == FEDERATED_SCHEDULE_SCHEMA_VERSION) != (
            keys == _V1_PAYLOAD_KEYS
        ):
            raise FederatedScheduleError("invalid federated schedule payload")

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
            schema_version=schema_version,
            max_clock_skew_seconds=payload.get("max_clock_skew_seconds"),
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

    def _window(self, phase: object) -> tuple[datetime, datetime]:
        if not isinstance(phase, FederatedSchedulePhase):
            raise FederatedScheduleError("phase must be a federated schedule phase")
        windows = {item: (start, end) for item, start, end in self._boundaries()}
        return windows[phase]

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


def order_admission_evidence(
    records: Iterable[FederatedPhaseAdmission],
) -> tuple[FederatedPhaseAdmission, ...]:
    """Order coordinator admission evidence by coordinator sequence.

    Site clocks and site-reported times are advisory, so evidence is ordered
    only by the sequence the coordinator assigned at receipt.

    Args:
        records: Coordinator decisions from ``coordinator_admission_at``.

    Returns:
        The decisions sorted by ascending coordinator sequence.

    Raises:
        FederatedScheduleError: If an item is not a decision, a decision has no
            coordinator sequence, or two decisions share a sequence.
    """

    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
        raise FederatedScheduleError("invalid admission evidence")
    collected = tuple(records)
    if not all(isinstance(item, FederatedPhaseAdmission) for item in collected):
        raise FederatedScheduleError("invalid admission evidence")
    if any(item.coordinator_sequence is None for item in collected):
        raise FederatedScheduleError(
            "admission evidence requires a coordinator sequence"
        )
    sequences = {item.coordinator_sequence for item in collected}
    if len(sequences) != len(collected):
        raise FederatedScheduleError("duplicate coordinator sequence")
    return tuple(sorted(collected, key=lambda item: item.coordinator_sequence))


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


def _require_clock_skew(value: object) -> None:
    if type(value) is not int or value < 0 or value > MAX_FEDERATED_CLOCK_SKEW_SECONDS:
        raise FederatedScheduleError(
            "max_clock_skew_seconds must be a supported non-negative integer"
        )


def _require_sequence(value: object) -> None:
    if type(value) is not int or value < 1:
        raise FederatedScheduleError("coordinator_sequence must be a positive integer")


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
    "FEDERATED_SCHEDULE_SCHEMA_VERSIONS",
    "FEDERATED_SCHEDULE_V2_SCHEMA_VERSION",
    "MAX_FEDERATED_CLOCK_SKEW_SECONDS",
    "MAX_FEDERATED_PHASE_DURATION_SECONDS",
    "FederatedAdmissionReasonCode",
    "FederatedClockBasis",
    "FederatedPhaseAdmission",
    "FederatedRoundSchedule",
    "FederatedScheduleError",
    "FederatedSchedulePhase",
    "order_admission_evidence",
]
