"""Purpose-bound demographic minimization before summary generation.

The minimizer retains values only for attribute classes explicitly allowed by
the caller's purpose policy. Audit output contains opaque policy identifiers,
counts, and categorical reasons, never demographic values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

SUMMARY_DEMOGRAPHIC_MINIMIZER_SCHEMA_VERSION: Final[int] = 1
NOT_ALLOWED_FOR_PURPOSE: Final[str] = "not_allowed_for_purpose"

_OPAQUE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DemographicMinimizationError(ValueError):
    """Raised when demographic evidence or a purpose policy is invalid."""


@dataclass(frozen=True, slots=True)
class DemographicEvidence:
    """One demographic attribute offered to the generation pipeline.

    ``value`` is intentionally excluded from representations. It is retained
    only when its opaque attribute class is explicitly allowed.
    """

    attribute_class_id: str
    value: Any = field(repr=False)

    def __post_init__(self) -> None:
        _validate_opaque_id(self.attribute_class_id, "attribute class")


@dataclass(frozen=True, slots=True)
class DemographicPurposePolicy:
    """Explicit attribute-class allowlist for one summary purpose."""

    purpose_id: str
    allowed_attribute_class_ids: frozenset[str]

    def __post_init__(self) -> None:
        _validate_opaque_id(self.purpose_id, "purpose")
        if not isinstance(self.allowed_attribute_class_ids, (set, frozenset)):
            raise DemographicMinimizationError(
                "allowed attribute classes must be a set of identifiers"
            )
        normalized = frozenset(self.allowed_attribute_class_ids)
        for class_id in normalized:
            _validate_opaque_id(class_id, "attribute class")
        object.__setattr__(self, "allowed_attribute_class_ids", normalized)


@dataclass(frozen=True, slots=True)
class RemovedDemographicClass:
    """Value-free removal count and policy reason for one class."""

    attribute_class_id: str
    removed_count: int
    reason_code: str = NOT_ALLOWED_FOR_PURPOSE

    def __post_init__(self) -> None:
        _validate_opaque_id(self.attribute_class_id, "attribute class")
        if type(self.removed_count) is not int or self.removed_count <= 0:
            raise DemographicMinimizationError(
                "removed count must be a positive integer"
            )
        if self.reason_code != NOT_ALLOWED_FOR_PURPOSE:
            raise DemographicMinimizationError("unsupported removal reason")

    def to_dict(self) -> dict[str, object]:
        """Return the closed audit representation for this class."""

        return {
            "attribute_class_id": self.attribute_class_id,
            "reason_code": self.reason_code,
            "removed_count": self.removed_count,
        }


@dataclass(frozen=True, slots=True)
class DemographicMinimizationReport:
    """Value-free report for one purpose-specific filtering operation."""

    purpose_id: str
    input_count: int
    allowed_count: int
    removed_count: int
    removed_classes: tuple[RemovedDemographicClass, ...]
    schema_version: int = SUMMARY_DEMOGRAPHIC_MINIMIZER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic report with no demographic values."""

        return {
            "allowed_count": self.allowed_count,
            "input_count": self.input_count,
            "purpose_id": self.purpose_id,
            "removed_classes": [item.to_dict() for item in self.removed_classes],
            "removed_count": self.removed_count,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize the report with stable ordering and separators."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True)
class DemographicMinimizationResult:
    """Allowed evidence for generation plus a separate value-free report."""

    allowed_evidence: tuple[DemographicEvidence, ...] = field(repr=False)
    report: DemographicMinimizationReport

    def to_dict(self) -> dict[str, object]:
        """Return only the value-free audit report for safe logging."""

        return self.report.to_dict()


def minimize_demographic_evidence(
    evidence: Iterable[DemographicEvidence],
    policy: DemographicPurposePolicy,
) -> DemographicMinimizationResult:
    """Filter demographic evidence through an explicit purpose allowlist.

    The returned ``allowed_evidence`` tuple is suitable for handing to a local
    generator. Removed values are not retained in the result. The separate
    report aggregates removals by opaque class identifier and fixed policy
    reason.

    Args:
        evidence: Demographic records proposed for summary generation.
        policy: Purpose-specific allowlist applied before generation.

    Returns:
        Allowed evidence and a deterministic, value-free audit report.

    Raises:
        DemographicMinimizationError: If the collection or policy is invalid.
            Exceptions do not echo caller-provided values or identifiers.
    """

    if not isinstance(policy, DemographicPurposePolicy):
        raise DemographicMinimizationError("invalid demographic purpose policy")
    records = _materialize_evidence(evidence)
    allowed: list[DemographicEvidence] = []
    removed_by_class: dict[str, int] = {}
    for record in records:
        if record.attribute_class_id in policy.allowed_attribute_class_ids:
            allowed.append(record)
        else:
            removed_by_class[record.attribute_class_id] = (
                removed_by_class.get(record.attribute_class_id, 0) + 1
            )

    removed_classes = tuple(
        RemovedDemographicClass(
            attribute_class_id=class_id,
            removed_count=removed_by_class[class_id],
        )
        for class_id in sorted(removed_by_class)
    )
    report = DemographicMinimizationReport(
        purpose_id=policy.purpose_id,
        input_count=len(records),
        allowed_count=len(allowed),
        removed_count=len(records) - len(allowed),
        removed_classes=removed_classes,
    )
    return DemographicMinimizationResult(tuple(allowed), report)


def _materialize_evidence(
    evidence: Iterable[DemographicEvidence],
) -> tuple[DemographicEvidence, ...]:
    if isinstance(evidence, (str, bytes, bytearray)):
        raise DemographicMinimizationError("invalid demographic evidence collection")
    try:
        records = tuple(evidence)
    except Exception:
        raise DemographicMinimizationError(
            "invalid demographic evidence collection"
        ) from None
    if any(not isinstance(record, DemographicEvidence) for record in records):
        raise DemographicMinimizationError("invalid demographic evidence collection")
    return records


def _validate_opaque_id(value: object, kind: str) -> None:
    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise DemographicMinimizationError(f"invalid {kind} identifier")


__all__ = [
    "NOT_ALLOWED_FOR_PURPOSE",
    "SUMMARY_DEMOGRAPHIC_MINIMIZER_SCHEMA_VERSION",
    "DemographicEvidence",
    "DemographicMinimizationError",
    "DemographicMinimizationReport",
    "DemographicMinimizationResult",
    "DemographicPurposePolicy",
    "RemovedDemographicClass",
    "minimize_demographic_evidence",
]
