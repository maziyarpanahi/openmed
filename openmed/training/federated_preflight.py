"""Deterministic fail-closed preflight reporting for federated rounds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from openmed.core.repro_hash import compute_environment_lock_digest

from .federated_metrics import (
    FederatedMetricEnvelope,
    FederatedMetricError,
)
from .federated_schedule import FederatedRoundSchedule
from .federated_update_metadata import (
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

FEDERATED_PREFLIGHT_SCHEMA_VERSION = "openmed.training.federated_preflight.v1"
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

class FederatedPreflightStatus(str, Enum):
    """Overall status of a federated preflight report."""

    ELIGIBLE = "eligible"
    REVIEW_REQUIRED = "review-required"
    BLOCKED = "blocked"

_STATUS_PRECEDENCE = {
    FederatedPreflightStatus.ELIGIBLE: 0,
    FederatedPreflightStatus.REVIEW_REQUIRED: 1,
    FederatedPreflightStatus.BLOCKED: 2,
}

@dataclass(frozen=True, slots=True)
class FederatedPreflightFinding:
    """One independent finding produced by a federated preflight check."""

    check: str
    status: FederatedPreflightStatus
    reason_code: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.check:
            raise ValueError("check must not be empty")

        if not self.reason_code:
            raise ValueError("reason_code must not be empty")

        if not isinstance(self.status, FederatedPreflightStatus):
            raise TypeError("status must be a FederatedPreflightStatus")

        if self.reference == "":
            raise ValueError("reference must be None or non-empty")

    def to_dict(self) -> dict[str, str | None]:
        """Return the canonical dictionary representation."""
        return {
            "check": self.check,
            "reason_code": self.reason_code,
            "reference": self.reference,
            "status": self.status.value,
        }

@dataclass(frozen=True, slots=True)
class FederatedPreflightReport:
    """Immutable, deterministic result of federated round preflight."""

    status: FederatedPreflightStatus
    findings: tuple[FederatedPreflightFinding, ...]
    digest_refs: tuple[str, ...] = ()
    schema_version: str = FEDERATED_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, FederatedPreflightStatus):
            raise TypeError("status must be a FederatedPreflightStatus")

        if self.schema_version != FEDERATED_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported preflight schema version: {self.schema_version!r}"
            )

        for finding in self.findings:
            if not isinstance(finding, FederatedPreflightFinding):
                raise TypeError(
                    "findings must contain FederatedPreflightFinding values"
                )

        expected_status = self._status_from_findings(self.findings)
        if self.status != expected_status:
            raise ValueError(
                "report status must be derived from finding statuses"
            )
        if len(set(self.digest_refs)) != len(self.digest_refs):
            raise ValueError("digest_refs must not contain duplicates")

        for digest_ref in self.digest_refs:
            if not isinstance(digest_ref, str):
                raise TypeError("digest_refs must contain strings")
            if _SHA256_DIGEST.fullmatch(digest_ref) is None:
                raise ValueError("digest_refs must be sha256 digests")

        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    self.findings,
                    key=lambda finding: (
                        finding.check,
                        finding.reason_code,
                        finding.status.value,
                        finding.reference or "",
                    ),
                )
            ),
        )

        object.__setattr__(
            self,
            "digest_refs",
            tuple(sorted(self.digest_refs)),
        )

    @staticmethod
    def _status_from_findings(
        findings: tuple[FederatedPreflightFinding, ...],
    ) -> FederatedPreflightStatus:
        """Derive status using precedence, never aggregate scoring."""
        if not findings:
            return FederatedPreflightStatus.ELIGIBLE

        return max(
            (finding.status for finding in findings),
            key=_STATUS_PRECEDENCE.__getitem__,
        )

    @classmethod
    def from_findings(
        cls,
        findings: (
            tuple[FederatedPreflightFinding, ...]
            | list[FederatedPreflightFinding]
        ),
        *,
        digest_refs: tuple[str, ...] | list[str] = (),
    ) -> "FederatedPreflightReport":
        """Build a report while deriving status from independent findings."""
        normalized_findings = tuple(findings)

        return cls(
            status=cls._status_from_findings(normalized_findings),
            findings=normalized_findings,
            digest_refs=tuple(digest_refs),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical dictionary representation."""
        return {
            "digest_refs": list(self.digest_refs),
            "findings": [finding.to_dict() for finding in self.findings],
            "schema_version": self.schema_version,
            "status": self.status.value,
        }

    def to_json(self) -> str:
        """Return deterministic canonical JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

def check_federated_schedule(
    schedule: FederatedRoundSchedule,
    *,
    timestamp: datetime,
    reference: str | None = None,
) -> FederatedPreflightFinding:
    """Check whether federated enrollment is currently allowed."""

    phase = schedule.active_phase_at(timestamp)

    if phase is None:
        return FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="SCHEDULE_EXPIRED",
            reference=reference,
        )

    if phase.value != "enrollment":
        return FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="ENROLLMENT_WINDOW_CLOSED",
            reference=reference,
        )

    return FederatedPreflightFinding(
        check="schedule",
        status=FederatedPreflightStatus.ELIGIBLE,
        reason_code="SCHEDULE_VALID",
        reference=reference,
    )

def run_federated_preflight(
    findings: (
        tuple[FederatedPreflightFinding, ...]
        | list[FederatedPreflightFinding]
    ),
    *,
    mandatory_checks: tuple[str, ...] | list[str] = (),
    digest_refs: tuple[str, ...] | list[str] = (),
) -> FederatedPreflightReport:
    """Build a fail-closed report from independent preflight findings.

    This function only orchestrates already-computed findings. It does not
    implement capability, manifest, schedule, privacy, environment, or
    update-schema validation.

    Any mandatory check absent from ``findings`` produces a blocked finding.
    """

    normalized_findings = tuple(findings)

    for finding in normalized_findings:
        if not isinstance(finding, FederatedPreflightFinding):
            raise TypeError(
                "findings must contain FederatedPreflightFinding values"
            )

    present_checks = {finding.check for finding in normalized_findings}

    for mandatory_check in mandatory_checks:
        if not mandatory_check:
            raise ValueError("mandatory check names must not be empty")

        if mandatory_check not in present_checks:
            normalized_findings += (
                FederatedPreflightFinding(
                    check=mandatory_check,
                    status=FederatedPreflightStatus.BLOCKED,
                    reason_code="MANDATORY_CHECK_MISSING",
                ),
            )

    return FederatedPreflightReport.from_findings(
        normalized_findings,
        digest_refs=digest_refs,
    )

def check_federated_environment(
    declared_digest: str | None,
    *,
    lock_path: str = "uv.lock",
    reference: str | None = None,
) -> FederatedPreflightFinding:
    """Check that the declared environment lock matches the repository lock."""

    if declared_digest is None or not declared_digest.strip():
        return FederatedPreflightFinding(
            check="environment",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="ENVIRONMENT_LOCK_MISSING",
            reference=reference,
        )

    expected_digest = compute_environment_lock_digest(lock_path)

    if declared_digest.strip() != expected_digest:
        return FederatedPreflightFinding(
            check="environment",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="ENVIRONMENT_LOCK_MISMATCH",
            reference=reference,
        )

    return FederatedPreflightFinding(
        check="environment",
        status=FederatedPreflightStatus.ELIGIBLE,
        reason_code="ENVIRONMENT_LOCK_VALID",
        reference=reference,
    )

def check_federated_update_schema(
    payload: object,
    *,
    policy: FederatedUpdatePolicy,
    reference: str | None = None,
) -> FederatedPreflightFinding:
    """Validate update metadata using the existing coordinator policy."""

    try:
        FederatedUpdateMetadata.from_dict(payload, policy=policy)
    except FederatedUpdateMetadataError:
        return FederatedPreflightFinding(
            check="update-schema",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="UPDATE_SCHEMA_INVALID",
            reference=reference,
        )

    return FederatedPreflightFinding(
        check="update-schema",
        status=FederatedPreflightStatus.ELIGIBLE,
        reason_code="UPDATE_SCHEMA_VALID",
        reference=reference,
    )

def check_federated_metric_schema(
    payload: object,
    *,
    reference: str | None = None,
) -> FederatedPreflightFinding:
    """Validate an aggregate metric envelope using its existing schema."""

    try:
        FederatedMetricEnvelope.from_dict(payload)
    except FederatedMetricError:
        return FederatedPreflightFinding(
            check="metric-schema",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="METRIC_SCHEMA_INVALID",
            reference=reference,
        )

    return FederatedPreflightFinding(
        check="metric-schema",
        status=FederatedPreflightStatus.ELIGIBLE,
        reason_code="METRIC_SCHEMA_VALID",
        reference=reference,
    )
