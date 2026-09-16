"""Deterministic numeric contradiction checks for clinical NLI pairs.

The precheck consumes structured values that a caller has already paired.  It
compares exact magnitudes, unit dimensions, and explicit reference intervals
before model inference.  It does not decide whether a measurement is clinically
normal, abnormal, safe, or actionable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from openmed.clinical.units import parse_measurement

NLI_NUMERIC_PRECHECK_SCHEMA_VERSION: Final[int] = 1
_REL_TOLERANCE: Final[float] = 1e-12
_ABS_TOLERANCE: Final[float] = 1e-12


class NumericPrecheckError(ValueError):
    """Raised when a numeric claim violates the value-free input contract."""


class NumericPrecheckStatus(str, Enum):
    """Outcome of the pre-model numeric check."""

    COMPATIBLE = "compatible"
    CONTRADICTION = "contradiction"
    REVIEW_REQUIRED = "review_required"
    NOT_APPLICABLE = "not_applicable"


class NumericPrecheckReason(str, Enum):
    """Value-free reason emitted by the numeric precheck."""

    VALUE_MISMATCH = "value_mismatch"
    UNIT_DIMENSION_MISMATCH = "unit_dimension_mismatch"
    REFERENCE_INTERVAL_MISMATCH = "reference_interval_mismatch"
    MISSING_UNIT_CONTEXT = "missing_unit_context"
    UNRESOLVED_UNIT = "unresolved_unit"
    INCOMPARABLE_NUMERIC_SHAPES = "incomparable_numeric_shapes"


@dataclass(frozen=True, repr=False)
class NumericClaim:
    """Structured numeric evidence for one side of an NLI pair.

    Raw values, units, and identifiers are deliberately excluded from
    ``repr``.  Precheck reports retain only fingerprints, dimensions, offsets,
    and reason codes.
    """

    value: object | None = field(default=None, repr=False)
    unit: str | None = field(default=None, repr=False)
    reference_low: object | None = field(default=None, repr=False)
    reference_high: object | None = field(default=None, repr=False)
    reference_unit: str | None = field(default=None, repr=False)
    low_inclusive: bool = True
    high_inclusive: bool = True
    measurement_key: str | None = field(default=None, repr=False)
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        if (
            self.value is None
            and self.reference_low is None
            and self.reference_high is None
        ):
            raise NumericPrecheckError(
                "numeric claim requires value or interval evidence"
            )
        for unit in (self.unit, self.reference_unit):
            if unit is not None and (type(unit) is not str or not unit.strip()):
                raise NumericPrecheckError("invalid numeric claim unit")
        if self.measurement_key is not None and (
            type(self.measurement_key) is not str or not self.measurement_key.strip()
        ):
            raise NumericPrecheckError("invalid numeric claim identity")
        if (
            type(self.low_inclusive) is not bool
            or type(self.high_inclusive) is not bool
        ):
            raise NumericPrecheckError("invalid reference interval boundary")
        if (self.source_start is None) != (self.source_end is None):
            raise NumericPrecheckError("invalid numeric claim source span")
        if self.source_start is not None and (
            type(self.source_start) is not int
            or type(self.source_end) is not int
            or self.source_start < 0
            or self.source_end <= self.source_start
        ):
            raise NumericPrecheckError("invalid numeric claim source span")

    def __repr__(self) -> str:
        """Return a value-free representation safe for routine diagnostics."""

        return (
            "NumericClaim("
            f"has_value={self.value is not None}, "
            f"has_interval={self.reference_low is not None or self.reference_high is not None}, "
            f"has_unit={self.unit is not None}, "
            f"source_span={self.source_span!r})"
        )

    @property
    def source_span(self) -> tuple[int, int] | None:
        """Return the half-open source span when one was supplied."""

        if self.source_start is None or self.source_end is None:
            return None
        return self.source_start, self.source_end

    @classmethod
    def from_obj(cls, value: NumericClaim | Mapping[str, Any]) -> NumericClaim:
        """Coerce a structured claim without retaining source text."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise NumericPrecheckError("numeric claim must be a mapping")

        interval = value.get("reference_interval", value.get("reference_range"))
        low = value.get("reference_low", value.get("low"))
        high = value.get("reference_high", value.get("high"))
        interval_unit = value.get("reference_unit")
        low_inclusive = value.get("low_inclusive", True)
        high_inclusive = value.get("high_inclusive", True)
        if interval is not None:
            if isinstance(interval, Mapping):
                low = interval.get("low", interval.get("minimum", low))
                high = interval.get("high", interval.get("maximum", high))
                interval_unit = interval.get("unit", interval_unit)
                low_inclusive = interval.get("low_inclusive", low_inclusive)
                high_inclusive = interval.get("high_inclusive", high_inclusive)
            elif (
                isinstance(interval, Sequence)
                and not isinstance(interval, (str, bytes))
                and len(interval) == 2
            ):
                low, high = interval
            else:
                raise NumericPrecheckError("invalid numeric reference interval")

        start, end = _coerce_span(value)
        return cls(
            value=_first_present(value, ("value", "magnitude", "numeric_value")),
            unit=_first_present(value, ("unit", "value_unit")),
            reference_low=low,
            reference_high=high,
            reference_unit=interval_unit,
            low_inclusive=low_inclusive,
            high_inclusive=high_inclusive,
            measurement_key=_first_present(
                value,
                ("measurement_key", "concept_id", "code", "measurement_id"),
            ),
            source_start=start,
            source_end=end,
        )


@dataclass(frozen=True)
class NumericContradictionEvidence:
    """Privacy-safe structured evidence for one precheck finding."""

    reason: NumericPrecheckReason
    field: str
    premise_fingerprint: str
    hypothesis_fingerprint: str
    premise_span: tuple[int, int] | None
    hypothesis_span: tuple[int, int] | None
    dimension: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return evidence without raw values, units, identifiers, or text."""

        return {
            "reason": self.reason.value,
            "field": self.field,
            "premise_fingerprint": self.premise_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "premise_span": list(self.premise_span) if self.premise_span else None,
            "hypothesis_span": list(self.hypothesis_span)
            if self.hypothesis_span
            else None,
            "dimension": dict(self.dimension),
        }


@dataclass(frozen=True)
class NumericPrecheckResult:
    """Fail-closed result returned before local NLI model inference."""

    status: NumericPrecheckStatus
    evidence: tuple[NumericContradictionEvidence, ...]
    schema_version: int = NLI_NUMERIC_PRECHECK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NLI_NUMERIC_PRECHECK_SCHEMA_VERSION:
            raise NumericPrecheckError("unsupported numeric-precheck schema")
        if not isinstance(self.status, NumericPrecheckStatus):
            raise NumericPrecheckError("invalid numeric-precheck status")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, NumericContradictionEvidence) for item in self.evidence
        ):
            raise NumericPrecheckError("invalid numeric-precheck evidence")
        if (
            self.status
            in {
                NumericPrecheckStatus.CONTRADICTION,
                NumericPrecheckStatus.REVIEW_REQUIRED,
            }
            and not self.evidence
        ):
            raise NumericPrecheckError("numeric-precheck finding requires evidence")
        if (
            self.status
            in {
                NumericPrecheckStatus.COMPATIBLE,
                NumericPrecheckStatus.NOT_APPLICABLE,
            }
            and self.evidence
        ):
            raise NumericPrecheckError("numeric-precheck status forbids evidence")

    @property
    def contradiction(self) -> bool:
        """Return whether exact structured evidence proves a contradiction."""

        return self.status is NumericPrecheckStatus.CONTRADICTION

    @property
    def requires_review(self) -> bool:
        """Return whether ambiguity must be escalated instead of inferred."""

        return self.status is NumericPrecheckStatus.REVIEW_REQUIRED

    @property
    def inference_allowed(self) -> bool:
        """Return whether a local NLI model may evaluate the pair."""

        return self.status in {
            NumericPrecheckStatus.COMPATIBLE,
            NumericPrecheckStatus.NOT_APPLICABLE,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a value-free, JSON-safe precheck report."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "contradiction": self.contradiction,
            "requires_review": self.requires_review,
            "inference_allowed": self.inference_allowed,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class _NormalizedNumber:
    status: str
    magnitude: float | None = None
    dimension: tuple[tuple[str, int], ...] = ()


def numeric_contradiction_precheck(
    premise: NumericClaim | Mapping[str, Any],
    hypothesis: NumericClaim | Mapping[str, Any],
) -> NumericPrecheckResult:
    """Compare structured numeric evidence before invoking a clinical NLI model.

    The caller is responsible for pairing claims that refer to the same
    measurement.  When both claims supply different identity keys, the result
    is ``not_applicable``.  Unresolved units or incomparable value/interval
    shapes fail closed to ``review_required``.
    """

    premise_claim = NumericClaim.from_obj(premise)
    hypothesis_claim = NumericClaim.from_obj(hypothesis)
    if (
        premise_claim.measurement_key is not None
        and hypothesis_claim.measurement_key is not None
        and _normalize_key(premise_claim.measurement_key)
        != _normalize_key(hypothesis_claim.measurement_key)
    ):
        return NumericPrecheckResult(
            status=NumericPrecheckStatus.NOT_APPLICABLE,
            evidence=(),
        )

    findings: list[NumericContradictionEvidence] = []
    review_findings: list[NumericContradictionEvidence] = []
    comparable = False

    if premise_claim.value is not None and hypothesis_claim.value is not None:
        comparable = True
        finding, review = _compare_values(premise_claim, hypothesis_claim)
        if finding is not None:
            findings.append(finding)
        if review is not None:
            review_findings.append(review)

    premise_has_interval = _has_interval(premise_claim)
    hypothesis_has_interval = _has_interval(hypothesis_claim)
    if premise_has_interval and hypothesis_has_interval:
        comparable = True
        finding, review = _compare_intervals(premise_claim, hypothesis_claim)
        if finding is not None:
            findings.append(finding)
        if review is not None:
            review_findings.append(review)

    if findings:
        return NumericPrecheckResult(
            status=NumericPrecheckStatus.CONTRADICTION,
            evidence=tuple(findings),
        )
    if review_findings:
        return NumericPrecheckResult(
            status=NumericPrecheckStatus.REVIEW_REQUIRED,
            evidence=tuple(review_findings),
        )
    if not comparable:
        evidence = _evidence(
            NumericPrecheckReason.INCOMPARABLE_NUMERIC_SHAPES,
            "numeric_shape",
            premise_claim,
            hypothesis_claim,
        )
        return NumericPrecheckResult(
            status=NumericPrecheckStatus.REVIEW_REQUIRED,
            evidence=(evidence,),
        )
    return NumericPrecheckResult(
        status=NumericPrecheckStatus.COMPATIBLE,
        evidence=(),
    )


def check_numeric_contradiction(
    premise: NumericClaim | Mapping[str, Any],
    hypothesis: NumericClaim | Mapping[str, Any],
) -> NumericPrecheckResult:
    """Alias for :func:`numeric_contradiction_precheck`."""

    return numeric_contradiction_precheck(premise, hypothesis)


def _compare_values(
    premise: NumericClaim,
    hypothesis: NumericClaim,
) -> tuple[
    NumericContradictionEvidence | None,
    NumericContradictionEvidence | None,
]:
    if (premise.unit is None) != (hypothesis.unit is None):
        return None, _evidence(
            NumericPrecheckReason.MISSING_UNIT_CONTEXT,
            "unit",
            premise,
            hypothesis,
        )

    premise_value = _normalize_number(premise.value, premise.unit)
    hypothesis_value = _normalize_number(hypothesis.value, hypothesis.unit)
    unresolved = _unresolved_reason(premise_value, hypothesis_value)
    if unresolved is not None:
        return None, _evidence(unresolved, "unit", premise, hypothesis)
    if premise_value.dimension != hypothesis_value.dimension:
        return _evidence(
            NumericPrecheckReason.UNIT_DIMENSION_MISMATCH,
            "unit",
            premise,
            hypothesis,
        ), None
    if not _same_number(premise_value.magnitude, hypothesis_value.magnitude):
        return _evidence(
            NumericPrecheckReason.VALUE_MISMATCH,
            "value",
            premise,
            hypothesis,
            premise_value.dimension,
        ), None
    return None, None


def _compare_intervals(
    premise: NumericClaim,
    hypothesis: NumericClaim,
) -> tuple[
    NumericContradictionEvidence | None,
    NumericContradictionEvidence | None,
]:
    premise_unit = premise.reference_unit or premise.unit
    hypothesis_unit = hypothesis.reference_unit or hypothesis.unit
    if (premise_unit is None) != (hypothesis_unit is None):
        return None, _evidence(
            NumericPrecheckReason.MISSING_UNIT_CONTEXT,
            "reference_interval",
            premise,
            hypothesis,
        )

    premise_low = _normalize_number(premise.reference_low, premise_unit)
    premise_high = _normalize_number(premise.reference_high, premise_unit)
    hypothesis_low = _normalize_number(hypothesis.reference_low, hypothesis_unit)
    hypothesis_high = _normalize_number(hypothesis.reference_high, hypothesis_unit)
    endpoints = (premise_low, premise_high, hypothesis_low, hypothesis_high)
    resolved = tuple(item for item in endpoints if item.status != "missing")
    if not resolved:
        return None, _evidence(
            NumericPrecheckReason.INCOMPARABLE_NUMERIC_SHAPES,
            "reference_interval",
            premise,
            hypothesis,
        )
    if any(item.status != "ok" for item in resolved):
        return None, _evidence(
            NumericPrecheckReason.UNRESOLVED_UNIT,
            "reference_interval",
            premise,
            hypothesis,
        )
    dimensions = {item.dimension for item in resolved}
    if len(dimensions) != 1:
        return _evidence(
            NumericPrecheckReason.UNIT_DIMENSION_MISMATCH,
            "reference_interval",
            premise,
            hypothesis,
        ), None

    if _interval_is_reversed(premise_low.magnitude, premise_high.magnitude) or (
        _interval_is_reversed(hypothesis_low.magnitude, hypothesis_high.magnitude)
    ):
        return None, _evidence(
            NumericPrecheckReason.INCOMPARABLE_NUMERIC_SHAPES,
            "reference_interval",
            premise,
            hypothesis,
        )

    if _intervals_disjoint(
        premise_low.magnitude,
        premise_high.magnitude,
        premise.low_inclusive,
        premise.high_inclusive,
        hypothesis_low.magnitude,
        hypothesis_high.magnitude,
        hypothesis.low_inclusive,
        hypothesis.high_inclusive,
    ):
        return _evidence(
            NumericPrecheckReason.REFERENCE_INTERVAL_MISMATCH,
            "reference_interval",
            premise,
            hypothesis,
            next(iter(dimensions)),
        ), None
    return None, None


def _normalize_number(value: object | None, unit: str | None) -> _NormalizedNumber:
    if value is None:
        return _NormalizedNumber(status="missing")
    if unit is None:
        numeric = _finite_float(value)
        if numeric is None:
            return _NormalizedNumber(status="unknown")
        return _NormalizedNumber(status="ok", magnitude=numeric)

    result = parse_measurement(value, unit)
    if result["status"] != "ok":
        return _NormalizedNumber(status=str(result["status"]))
    magnitude = result.get("canonical_magnitude")
    dimension = result.get("dimension")
    numeric_magnitude = _finite_float(magnitude)
    if not isinstance(dimension, Mapping) or numeric_magnitude is None:
        return _NormalizedNumber(status="unknown")
    return _NormalizedNumber(
        status="ok",
        magnitude=numeric_magnitude,
        dimension=tuple(
            sorted((str(key), int(power)) for key, power in dimension.items())
        ),
    )


def _unresolved_reason(
    premise: _NormalizedNumber,
    hypothesis: _NormalizedNumber,
) -> NumericPrecheckReason | None:
    if premise.status == "ok" and hypothesis.status == "ok":
        return None
    return NumericPrecheckReason.UNRESOLVED_UNIT


def _same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(
        left,
        right,
        rel_tol=_REL_TOLERANCE,
        abs_tol=_ABS_TOLERANCE,
    )


def _intervals_disjoint(
    left_low: float | None,
    left_high: float | None,
    left_low_inclusive: bool,
    left_high_inclusive: bool,
    right_low: float | None,
    right_high: float | None,
    right_low_inclusive: bool,
    right_high_inclusive: bool,
) -> bool:
    if left_high is not None and right_low is not None:
        if _same_number(left_high, right_low):
            if not (left_high_inclusive and right_low_inclusive):
                return True
        elif left_high < right_low:
            return True
    if right_high is not None and left_low is not None:
        if _same_number(right_high, left_low):
            if not (right_high_inclusive and left_low_inclusive):
                return True
        elif right_high < left_low:
            return True
    return False


def _interval_is_reversed(low: float | None, high: float | None) -> bool:
    return (
        low is not None
        and high is not None
        and not _same_number(low, high)
        and low > high
    )


def _evidence(
    reason: NumericPrecheckReason,
    field_name: str,
    premise: NumericClaim,
    hypothesis: NumericClaim,
    dimension: tuple[tuple[str, int], ...] = (),
) -> NumericContradictionEvidence:
    return NumericContradictionEvidence(
        reason=reason,
        field=field_name,
        premise_fingerprint=_fingerprint(premise),
        hypothesis_fingerprint=_fingerprint(hypothesis),
        premise_span=premise.source_span,
        hypothesis_span=hypothesis.source_span,
        dimension=dimension,
    )


def _fingerprint(claim: NumericClaim) -> str:
    payload = {
        "value": _stable_value(claim.value),
        "unit": claim.unit,
        "reference_low": _stable_value(claim.reference_low),
        "reference_high": _stable_value(claim.reference_high),
        "reference_unit": claim.reference_unit,
        "low_inclusive": claim.low_inclusive,
        "high_inclusive": claim.high_inclusive,
        "measurement_key": claim.measurement_key,
        "source_span": claim.source_span,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"openmed:nli-numeric-precheck:v1\0" + encoded).hexdigest()


def _stable_value(value: object | None) -> object:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    return str(value)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _has_interval(claim: NumericClaim) -> bool:
    return claim.reference_low is not None or claim.reference_high is not None


def _normalize_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_span(mapping: Mapping[str, Any]) -> tuple[int | None, int | None]:
    raw_span = mapping.get(
        "source_span", mapping.get("source_offset", mapping.get("offset"))
    )
    if raw_span is not None:
        if (
            isinstance(raw_span, Sequence)
            and not isinstance(raw_span, (str, bytes))
            and len(raw_span) == 2
        ):
            return raw_span[0], raw_span[1]
        raise NumericPrecheckError("invalid numeric claim source span")
    return mapping.get("source_start", mapping.get("start")), mapping.get(
        "source_end", mapping.get("end")
    )


__all__ = [
    "NLI_NUMERIC_PRECHECK_SCHEMA_VERSION",
    "NumericClaim",
    "NumericContradictionEvidence",
    "NumericPrecheckError",
    "NumericPrecheckReason",
    "NumericPrecheckResult",
    "NumericPrecheckStatus",
    "check_numeric_contradiction",
    "numeric_contradiction_precheck",
]
