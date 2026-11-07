"""Deterministic lifetime and audience checks for agent capability grants.

A grant is described by integer epoch seconds and two identifiers. This module
decides whether it is usable at a given instant; it does not issue grants,
verify signatures, evaluate authorization policy, or execute anything. Grant
payloads, tool arguments, and clinical text are never accepted, so a report
cannot carry them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .identifiers import CapabilityId, GovernanceIdError

CAPABILITY_VALIDITY_SCHEMA_VERSION: Final[str] = "openmed.agent.capability_validity.v1"
MAX_CLOCK_SKEW_SECONDS: Final[int] = 300
MAX_GRANT_LIFETIME_SECONDS: Final[int] = 86_400
MAX_EPOCH_SECONDS: Final[int] = 4_102_444_800
DEFAULT_CLOCK_SKEW_SECONDS: Final[int] = 60

GRANT_REASON_CODES: Final[tuple[str, ...]] = (
    "audience_mismatch",
    "clock_skew_exceeded",
    "lifetime_exceeded",
    "grant_not_yet_valid",
    "grant_expired",
)

_AUDIENCE_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_GRANT_FIELDS = (
    "capability_id",
    "audience",
    "issued_at",
    "not_before",
    "expires_at",
)
_REPORT_FIELDS = (
    "schema_version",
    "capability_id",
    "audience",
    "status",
    "reason_codes",
    "evaluated_at",
    "seconds_until_expiry",
)


class GrantStatus(str, Enum):
    """Closed verdict vocabulary for one grant evaluation.

    Values:
        VALID: The grant is usable at the evaluated instant.
        NOT_YET_VALID: The grant starts later; it may become usable.
        EXPIRED: The grant's lifetime has elapsed.
        REJECTED: The grant is unusable regardless of when it is evaluated.
    """

    VALID = "valid"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    REJECTED = "rejected"


_STATUS_RANK: Final[dict[GrantStatus, int]] = {
    GrantStatus.VALID: 0,
    GrantStatus.NOT_YET_VALID: 1,
    GrantStatus.EXPIRED: 2,
    GrantStatus.REJECTED: 3,
}


class CapabilityValidityError(ValueError):
    """Raised when grant metadata fails closed structural validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional fixed public field associated with the failure.

    Messages and attributes carry controlled diagnostic metadata only; a
    rejected value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CapabilityGrant:
    """Lifetime and audience metadata for one capability grant.

    Attributes:
        capability_id: Canonical ``capability:`` governance identifier.
        audience: Local service name the grant was minted for.
        issued_at: Issue time in whole epoch seconds.
        expires_at: Expiry time in whole epoch seconds, after ``issued_at``.
        not_before: Optional activation time inside the issue/expiry window.
    """

    capability_id: str
    audience: str
    issued_at: int
    expires_at: int
    not_before: int | None = None

    def __post_init__(self) -> None:
        _validate_capability_id(self.capability_id)
        _validate_audience(self.audience, "audience")
        _validate_epoch(self.issued_at, "issued_at")
        _validate_epoch(self.expires_at, "expires_at")
        if self.expires_at <= self.issued_at:
            raise CapabilityValidityError("expiry_not_after_issue", "expires_at")
        if self.not_before is not None:
            _validate_epoch(self.not_before, "not_before")
            if not self.issued_at <= self.not_before <= self.expires_at:
                raise CapabilityValidityError("not_before_out_of_window", "not_before")

    @property
    def effective_not_before(self) -> int:
        """Return ``not_before`` when present, otherwise ``issued_at``."""

        return self.issued_at if self.not_before is None else self.not_before

    @property
    def lifetime_seconds(self) -> int:
        """Return the declared lifetime in whole seconds."""

        return self.expires_at - self.issued_at

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "capability_id": self.capability_id,
            "audience": self.audience,
            "issued_at": self.issued_at,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
        }
        return {field: values[field] for field in _GRANT_FIELDS}


@dataclass(frozen=True, slots=True)
class GrantValidityReport:
    """Deterministic verdict for one grant at one instant."""

    capability_id: str
    audience: str
    status: GrantStatus
    reason_codes: tuple[str, ...]
    evaluated_at: int
    seconds_until_expiry: int
    schema_version: str = CAPABILITY_VALIDITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != CAPABILITY_VALIDITY_SCHEMA_VERSION
        ):
            raise CapabilityValidityError("invalid_schema_version", "schema_version")

    @property
    def is_valid(self) -> bool:
        """Return whether the grant is usable at the evaluated instant."""

        return self.status is GrantStatus.VALID

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "audience": self.audience,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "evaluated_at": self.evaluated_at,
            "seconds_until_expiry": self.seconds_until_expiry,
        }
        return {field: values[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def check_capability_validity(
    grant: CapabilityGrant,
    *,
    now: int,
    expected_audience: str,
    max_clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    max_lifetime_seconds: int | None = None,
) -> GrantValidityReport:
    """Decide whether a capability grant is usable at ``now``.

    The caller supplies the instant, so the check is offline, reproducible,
    and independent of the host clock. All findings are collected in one pass
    and the worst status wins, ranked
    ``valid < not_yet_valid < expired < rejected``.

    Args:
        grant: The grant to evaluate.
        now: Evaluation instant in whole epoch seconds.
        expected_audience: Local service name that intends to use the grant.
        max_clock_skew_seconds: Tolerance applied to both window edges and to
            a grant issued slightly in the future.
        max_lifetime_seconds: Optional ceiling on the declared lifetime.

    Returns:
        A :class:`GrantValidityReport` whose reason codes follow the fixed
        order in :data:`GRANT_REASON_CODES`.

    Raises:
        CapabilityValidityError: If any argument is structurally invalid.
    """

    if not isinstance(grant, CapabilityGrant):
        raise CapabilityValidityError("invalid_grant_type", "grant")
    _validate_epoch(now, "now")
    _validate_audience(expected_audience, "expected_audience")
    _validate_bounded_int(
        max_clock_skew_seconds, "max_clock_skew_seconds", MAX_CLOCK_SKEW_SECONDS
    )
    if max_lifetime_seconds is not None:
        _validate_bounded_int(
            max_lifetime_seconds, "max_lifetime_seconds", MAX_GRANT_LIFETIME_SECONDS
        )
        if max_lifetime_seconds == 0:
            raise CapabilityValidityError(
                "lifetime_bound_invalid", "max_lifetime_seconds"
            )

    findings: list[tuple[str, GrantStatus]] = []
    if grant.audience != expected_audience:
        findings.append(("audience_mismatch", GrantStatus.REJECTED))
    if grant.issued_at > now + max_clock_skew_seconds:
        findings.append(("clock_skew_exceeded", GrantStatus.REJECTED))
    if (
        max_lifetime_seconds is not None
        and grant.lifetime_seconds > max_lifetime_seconds
    ):
        findings.append(("lifetime_exceeded", GrantStatus.REJECTED))
    if now < grant.effective_not_before - max_clock_skew_seconds:
        findings.append(("grant_not_yet_valid", GrantStatus.NOT_YET_VALID))
    if now > grant.expires_at + max_clock_skew_seconds:
        findings.append(("grant_expired", GrantStatus.EXPIRED))

    status = GrantStatus.VALID
    for _, candidate in findings:
        if _STATUS_RANK[candidate] > _STATUS_RANK[status]:
            status = candidate
    reason_codes = tuple(
        sorted((code for code, _ in findings), key=GRANT_REASON_CODES.index)
    )
    return GrantValidityReport(
        capability_id=grant.capability_id,
        audience=grant.audience,
        status=status,
        reason_codes=reason_codes,
        evaluated_at=now,
        seconds_until_expiry=grant.expires_at - now,
    )


def _validate_capability_id(value: Any) -> None:
    try:
        CapabilityId.parse(value)
    except GovernanceIdError:
        pass
    else:
        return
    # Raised outside the handler so the rejected identifier cannot be chained.
    raise CapabilityValidityError("invalid_capability_id", "capability_id")


def _validate_audience(value: Any, field_name: str) -> None:
    if type(value) is not str or _AUDIENCE_RE.fullmatch(value) is None:
        raise CapabilityValidityError("invalid_audience", field_name)


def _validate_epoch(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise CapabilityValidityError("invalid_timestamp", field_name)
    if value < 0 or value > MAX_EPOCH_SECONDS:
        raise CapabilityValidityError("timestamp_out_of_range", field_name)


def _validate_bounded_int(value: Any, field_name: str, maximum: int) -> None:
    if type(value) is not int:
        raise CapabilityValidityError("invalid_bound", field_name)
    if value < 0 or value > maximum:
        raise CapabilityValidityError("bound_out_of_range", field_name)


__all__ = [
    "CAPABILITY_VALIDITY_SCHEMA_VERSION",
    "DEFAULT_CLOCK_SKEW_SECONDS",
    "GRANT_REASON_CODES",
    "MAX_CLOCK_SKEW_SECONDS",
    "MAX_EPOCH_SECONDS",
    "MAX_GRANT_LIFETIME_SECONDS",
    "CapabilityGrant",
    "CapabilityValidityError",
    "GrantStatus",
    "GrantValidityReport",
    "check_capability_validity",
]
