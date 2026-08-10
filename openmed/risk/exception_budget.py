"""Deterministic, PHI-free gating for waived privacy exceptions.

Privacy exceptions are release metadata, not a second place to retain a
finding.  This module accepts only bounded metadata (severity, scope, expiry,
policy fingerprint, and an optional count) and emits aggregate counts and
one-way fingerprints.  Unknown or unbounded metadata fails closed.  Any input
field such as ``finding_text`` is deliberately ignored and is never retained,
logged, or returned by a report.

The evaluator has no clock or network dependency.  Callers that need expiry
status or an expiry-duration limit provide an explicit ``as_of`` date, which
keeps the result reproducible in local and CI runs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from openmed.core.audit import stable_hash

__all__ = [
    "ExceptionBudget",
    "ExceptionBudgetExceeded",
    "ExceptionBudgetGate",
    "ExceptionBudgetVerdict",
    "ExceptionBudgetViolation",
    "PrivacyException",
    "PrivacyExceptionBudget",
    "PrivacyExceptionBudgetExceeded",
    "check_exception_budget",
    "check_privacy_exception_budget",
    "evaluate_exception_budget",
    "evaluate_privacy_exception_budget",
    "fingerprint_policy",
    "scope_fingerprint",
]

_FINGERPRINT_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_KNOWN_SEVERITIES = frozenset(
    {
        "blocker",
        "critical",
        "error",
        "high",
        "info",
        "low",
        "medium",
        "moderate",
        "negligible",
        "notice",
        "warning",
    }
)
_UNBOUNDED_MARKERS = frozenset({"*", "all", "unbounded"})
_EXPIRY_BOUNDED = "bounded"
_EXPIRY_ACTIVE = "active"
_EXPIRY_EXPIRED = "expired"
_EXPIRY_UNBOUNDED = "unbounded"
_SCHEMA = "openmed.privacy-exception-budget.v1"


def _text(value: Any) -> tuple[str | None, bool]:
    """Return a trimmed string and whether a non-string value was supplied."""

    if value is None:
        return None, False
    if not isinstance(value, str):
        return None, True
    normalized = value.strip()
    return (normalized or None), False


def _is_unbounded_marker(value: str | None) -> bool:
    return value is not None and value.casefold() in _UNBOUNDED_MARKERS


def _safe_label(value: Any) -> tuple[str | None, bool, bool]:
    """Normalize a non-sensitive label without preserving malformed input."""

    normalized, invalid_type = _text(value)
    if normalized is None:
        return None, invalid_type, False
    lowered = normalized.casefold()
    if lowered in _UNBOUNDED_MARKERS:
        return None, False, True
    if not _SAFE_LABEL_RE.fullmatch(lowered):
        return None, True, False
    return lowered, False, False


def _coerce_date(value: Any) -> tuple[date | None, bool]:
    """Parse an ISO date while returning only a safe validity bit."""

    if value is None:
        return None, False
    if isinstance(value, datetime):
        return value.date(), False
    if isinstance(value, date):
        return value, False
    if not isinstance(value, str):
        return None, True
    normalized = value.strip()
    if not normalized:
        return None, True
    try:
        return date.fromisoformat(normalized), False
    except ValueError:
        try:
            iso_datetime = normalized.replace("Z", "+00:00")
            return datetime.fromisoformat(iso_datetime).date(), False
        except ValueError:
            return None, True


def _safe_severity(value: Any) -> tuple[str | None, bool, bool]:
    """Normalize one of the finite, non-sensitive severity labels."""

    normalized, invalid, unbounded = _safe_label(value)
    if normalized is not None and normalized not in _KNOWN_SEVERITIES:
        return None, True, False
    return normalized, invalid, unbounded


def _validate_limit(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _canonical_fingerprint(
    value: Any, *, namespace: str
) -> tuple[str | None, bool, bool]:
    """Return a safe fingerprint, invalidity, and explicit unboundedness."""

    normalized, invalid_type = _text(value)
    if normalized is None:
        return None, invalid_type, False
    if _is_unbounded_marker(normalized):
        return None, False, True
    lowered = normalized.casefold()
    if _FINGERPRINT_RE.fullmatch(lowered):
        return lowered, False, False
    if _HEX_DIGEST_RE.fullmatch(lowered):
        return f"sha256:{lowered}", False, False
    return (
        stable_hash({"schema": f"{_SCHEMA}.{namespace}.v1", "value": normalized}),
        False,
        False,
    )


def scope_fingerprint(scope: str) -> str:
    """Return a deterministic fingerprint for a release scope.

    Scope values are never returned verbatim because a caller may derive a
    scope from a sensitive identifier.  Passing an existing SHA-256 or
    HMAC-SHA-256 fingerprint preserves that safe value.

    Args:
        scope: A non-empty scope label or an existing fingerprint.

    Raises:
        TypeError: If ``scope`` is not a string.
        ValueError: If ``scope`` is empty or explicitly unbounded.
    """

    fingerprint, invalid, unbounded = _canonical_fingerprint(
        scope,
        namespace="scope",
    )
    if invalid:
        raise TypeError("scope must be a non-empty string")
    if unbounded or fingerprint is None:
        raise ValueError("scope must be bounded")
    return fingerprint


def fingerprint_policy(policy: str) -> str:
    """Return a deterministic, safe policy fingerprint.

    Existing ``sha256:``/``hmac-sha256:`` values are retained; a plain policy
    label is hashed before it can enter a report or exception object.
    """

    fingerprint, invalid, unbounded = _canonical_fingerprint(
        policy,
        namespace="policy",
    )
    if invalid:
        raise TypeError("policy must be a non-empty string")
    if unbounded or fingerprint is None:
        raise ValueError("policy fingerprint must be bounded")
    return fingerprint


@dataclass(frozen=True, init=False)
class PrivacyException:
    """Bounded metadata describing one waived privacy finding.

    The constructor accepts a scope label only long enough to fingerprint it;
    the raw scope is not stored.  ``None`` expiry, scope, policy fingerprint,
    or severity represents an unbounded exception and is rejected by the gate.
    ``count`` lets synthetic fixtures represent repeated exceptions without
    carrying finding-level records.

    Args:
        severity: A stable severity label such as ``"high"``.
        scope: A release scope label. It is stored only as a fingerprint.
        expires_on: Inclusive ISO expiry date, ``date``, or ``datetime``.
        policy_fingerprint: A policy fingerprint or a label that will be
            hashed before storage.
        count: Number of equivalent exceptions represented by this record.
        expires_at: Alias for ``expires_on``.
        expiry: Alias for ``expires_on``.
        unbounded: Explicitly mark the exception as unbounded.
    """

    severity: str | None
    scope_fingerprint: str | None
    expires_on: date | None
    policy_fingerprint: str | None
    count: int
    _invalid: bool = field(default=False, repr=False, compare=False)
    _explicitly_unbounded: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        severity: str | None,
        scope: str | None,
        expires_on: date | datetime | str | None = None,
        policy_fingerprint: str | None = None,
        count: int = 1,
        *,
        expires_at: date | datetime | str | None = None,
        expiry: date | datetime | str | None = None,
        unbounded: bool = False,
    ) -> None:
        expiry_values = [
            value for value in (expires_on, expires_at, expiry) if value is not None
        ]
        if len(expiry_values) > 1:
            raise ValueError("only one expiry value may be supplied")
        selected_expiry = expiry_values[0] if expiry_values else None

        normalized_severity, severity_invalid, severity_unbounded = _safe_severity(
            severity
        )
        normalized_scope, scope_invalid, scope_unbounded = _canonical_fingerprint(
            scope,
            namespace="scope",
        )
        normalized_expiry, expiry_invalid = _coerce_date(selected_expiry)
        normalized_policy, policy_invalid, policy_unbounded = _canonical_fingerprint(
            policy_fingerprint,
            namespace="policy",
        )
        count_invalid = isinstance(count, bool) or not isinstance(count, int)
        if count_invalid or count < 0:
            raise ValueError("count must be a non-negative integer")

        object.__setattr__(self, "severity", normalized_severity)
        object.__setattr__(self, "scope_fingerprint", normalized_scope)
        object.__setattr__(self, "expires_on", normalized_expiry)
        object.__setattr__(self, "policy_fingerprint", normalized_policy)
        object.__setattr__(self, "count", int(count))
        object.__setattr__(
            self,
            "_invalid",
            severity_invalid or scope_invalid or expiry_invalid or policy_invalid,
        )
        object.__setattr__(
            self,
            "_explicitly_unbounded",
            bool(unbounded)
            or severity_unbounded
            or scope_unbounded
            or policy_unbounded,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrivacyException":
        """Build a privacy exception from metadata without retaining the mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("exception must be a mapping")
        if "expires_on" in payload:
            expiry = payload.get("expires_on")
        elif "expires_at" in payload:
            expiry = payload.get("expires_at")
        else:
            expiry = payload.get("expiry")
        scope = payload.get("scope")
        if "scope_fingerprint" in payload and scope is None:
            scope = payload.get("scope_fingerprint")
        policy = payload.get("policy_fingerprint")
        if policy is None:
            policy = payload.get("policy")
        return cls(
            payload.get("severity"),
            scope,
            expiry,
            policy,
            payload.get("count", payload.get("exception_count", 1)),
            unbounded=payload.get("unbounded", False) is True,
        )

    @property
    def unbounded(self) -> bool:
        """Return whether any required exception bound is absent or invalid."""

        return (
            self._invalid
            or self._explicitly_unbounded
            or self.severity is None
            or self.scope_fingerprint is None
            or self.expires_on is None
            or self.policy_fingerprint is None
        )

    @property
    def expires_at(self) -> date | None:
        """Return the normalized expiry date under the common alias."""

        return self.expires_on

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without raw scope or finding content."""

        return {
            "severity": self.severity,
            "scope_fingerprint": self.scope_fingerprint,
            "expires_on": self.expires_on.isoformat() if self.expires_on else None,
            "policy_fingerprint": self.policy_fingerprint,
            "count": self.count,
            "bounded": not self.unbounded,
        }


@dataclass(frozen=True)
class ExceptionBudget:
    """Finite aggregate limits for one privacy-exception review.

    ``max_total`` is required and always finite.  Per-severity, per-scope, and
    per-policy limits are optional refinements; omitted keys are still bounded
    by ``max_total``.  ``max_expiry_days`` requires an explicit ``as_of`` date
    during evaluation so the duration check remains deterministic.
    """

    max_total: int
    max_by_severity: Mapping[str, int] = field(default_factory=dict)
    max_by_scope: Mapping[str, int] = field(default_factory=dict)
    max_by_policy_fingerprint: Mapping[str, int] = field(default_factory=dict)
    max_expiry_days: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_total", _validate_limit(self.max_total, name="max_total")
        )
        object.__setattr__(
            self,
            "max_by_severity",
            MappingProxyType(
                _normalize_limit_mapping(
                    self.max_by_severity,
                    dimension="severity",
                )
            ),
        )
        object.__setattr__(
            self,
            "max_by_scope",
            MappingProxyType(
                _normalize_limit_mapping(self.max_by_scope, dimension="scope")
            ),
        )
        object.__setattr__(
            self,
            "max_by_policy_fingerprint",
            MappingProxyType(
                _normalize_limit_mapping(
                    self.max_by_policy_fingerprint,
                    dimension="policy_fingerprint",
                )
            ),
        )
        if self.max_expiry_days is not None:
            object.__setattr__(
                self,
                "max_expiry_days",
                _validate_limit(self.max_expiry_days, name="max_expiry_days"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExceptionBudget":
        """Build a budget from a JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("budget must be an ExceptionBudget or mapping")
        return cls(
            max_total=payload.get("max_total", payload.get("total", 0)),
            max_by_severity=payload.get("max_by_severity", {}),
            max_by_scope=payload.get("max_by_scope", {}),
            max_by_policy_fingerprint=payload.get(
                "max_by_policy_fingerprint",
                payload.get("max_by_policy", {}),
            ),
            max_expiry_days=payload.get("max_expiry_days"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, PHI-free budget metadata."""

        return {
            "max_total": self.max_total,
            "max_by_severity": dict(sorted(self.max_by_severity.items())),
            "max_by_scope": dict(sorted(self.max_by_scope.items())),
            "max_by_policy_fingerprint": dict(
                sorted(self.max_by_policy_fingerprint.items())
            ),
            "max_expiry_days": self.max_expiry_days,
        }


PrivacyExceptionBudget = ExceptionBudget


def _normalize_limit_mapping(
    value: Mapping[str, int],
    *,
    dimension: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{dimension} limits must be a mapping")
    normalized: dict[str, int] = {}
    for key, limit in value.items():
        if dimension == "severity":
            safe_key, invalid, unbounded = _safe_severity(key)
            if invalid or unbounded or safe_key is None:
                raise ValueError("severity limit keys must be safe labels")
        elif dimension == "scope":
            safe_key, invalid, unbounded = _canonical_fingerprint(
                key,
                namespace="scope",
            )
            if invalid or unbounded or safe_key is None:
                raise ValueError("scope limit keys must be bounded strings")
        else:
            safe_key, invalid, unbounded = _canonical_fingerprint(
                key,
                namespace="policy",
            )
            if invalid or unbounded or safe_key is None:
                raise ValueError("policy limit keys must be bounded strings")
        if safe_key in normalized:
            raise ValueError(f"duplicate normalized {dimension} limit key")
        normalized[safe_key] = _validate_limit(limit, name=f"{dimension} limit")
    return normalized


@dataclass(frozen=True)
class ExceptionBudgetViolation:
    """One aggregate or boundedness failure in an exception-budget verdict."""

    metric: str
    consumed: int
    limit: int | None
    comparison: str
    key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic violation without finding details."""

        return {
            "metric": self.metric,
            "key": self.key,
            "consumed": self.consumed,
            "limit": self.limit,
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class ExceptionBudgetVerdict:
    """Aggregate, deterministic result of a privacy-exception budget check."""

    allowed: bool
    budget: Mapping[str, Any]
    total_count: int
    counts_by_severity: Mapping[str, int]
    counts_by_scope: Mapping[str, int]
    counts_by_expiry: Mapping[str, int]
    counts_by_policy_fingerprint: Mapping[str, int]
    unbounded_count: int
    expired_count: int
    evaluation_date: str | None
    violations: tuple[ExceptionBudgetViolation, ...]

    @property
    def within_budget(self) -> bool:
        """Return the verdict under the naming used by other risk gates."""

        return self.allowed

    @property
    def breakdown(self) -> dict[str, Any]:
        """Return the aggregate dimensions in a compact report shape."""

        return {
            "total": self.total_count,
            "severity": dict(sorted(self.counts_by_severity.items())),
            "scope": dict(sorted(self.counts_by_scope.items())),
            "expiry": dict(sorted(self.counts_by_expiry.items())),
            "policy_fingerprint": dict(
                sorted(self.counts_by_policy_fingerprint.items())
            ),
            "unbounded": self.unbounded_count,
            "expired": self.expired_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible, PHI-free report data."""

        return {
            "allowed": self.allowed,
            "within_budget": self.within_budget,
            "budget": dict(self.budget),
            "total_count": self.total_count,
            "counts_by_severity": dict(sorted(self.counts_by_severity.items())),
            "counts_by_scope": dict(sorted(self.counts_by_scope.items())),
            "counts_by_expiry": dict(sorted(self.counts_by_expiry.items())),
            "counts_by_policy_fingerprint": dict(
                sorted(self.counts_by_policy_fingerprint.items())
            ),
            "unbounded_count": self.unbounded_count,
            "expired_count": self.expired_count,
            "evaluation_date": self.evaluation_date,
            "violations": [violation.to_dict() for violation in self.violations],
        }


class ExceptionBudgetExceeded(ValueError):
    """Raised by strict checking when a privacy exception budget is denied."""

    def __init__(self, verdict: ExceptionBudgetVerdict) -> None:
        self.verdict = verdict
        metrics = (
            ", ".join(sorted({violation.metric for violation in verdict.violations}))
            or "exception_budget"
        )
        super().__init__(f"Privacy exception budget denied: {metrics}")


PrivacyExceptionBudgetExceeded = ExceptionBudgetExceeded


@dataclass(frozen=True)
class ExceptionBudgetGate:
    """Reusable evaluator for one explicit privacy-exception budget."""

    budget: ExceptionBudget
    as_of: date | datetime | str | None = None

    def evaluate(
        self,
        exceptions: Iterable[PrivacyException | Mapping[str, Any]],
    ) -> ExceptionBudgetVerdict:
        """Evaluate exceptions against this gate without raising."""

        return evaluate_exception_budget(
            exceptions,
            self.budget,
            as_of=self.as_of,
        )

    def check(
        self,
        exceptions: Iterable[PrivacyException | Mapping[str, Any]],
    ) -> ExceptionBudgetVerdict:
        """Evaluate exceptions and raise when the gate is not allowed."""

        return check_exception_budget(
            exceptions,
            self.budget,
            as_of=self.as_of,
        )


def evaluate_exception_budget(
    exceptions: Iterable[PrivacyException | Mapping[str, Any]]
    | PrivacyException
    | Mapping[str, Any],
    budget: ExceptionBudget | Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
    strict: bool = False,
) -> ExceptionBudgetVerdict:
    """Evaluate synthetic privacy exceptions against finite aggregate limits.

    Args:
        exceptions: Metadata records or a single metadata record. Mapping
            fields outside severity, scope, expiry, policy fingerprint, count,
            and ``unbounded`` are ignored.
        budget: An :class:`ExceptionBudget` or JSON-compatible budget mapping.
        as_of: Explicit date used for expired and expiry-duration checks. When
            omitted, bounded records are counted as ``"bounded"`` without
            consulting the system clock.
        strict: Raise :class:`ExceptionBudgetExceeded` when the result is not
            allowed.

    Returns:
        A deterministic aggregate report containing no raw exception values.
    """

    selected_budget = _coerce_budget(budget)
    evaluation_date, invalid_as_of = _coerce_date(as_of)
    if invalid_as_of:
        raise ValueError("as_of must be an ISO date, date, datetime, or None")

    severity_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    expiry_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    total_count = 0
    unbounded_count = 0
    expired_count = 0
    max_expiry_distance: int | None = None

    for exception in _iter_exceptions(exceptions):
        count = exception.count
        if count == 0:
            continue
        total_count += count
        severity_counts[exception.severity or "unknown"] += count
        scope_counts[exception.scope_fingerprint or _EXPIRY_UNBOUNDED] += count
        policy_counts[exception.policy_fingerprint or _EXPIRY_UNBOUNDED] += count

        if exception.unbounded:
            unbounded_count += count
            expiry_counts[_EXPIRY_UNBOUNDED] += count
            continue

        if evaluation_date is None:
            expiry_counts[_EXPIRY_BOUNDED] += count
        elif exception.expires_on < evaluation_date:
            expiry_counts[_EXPIRY_EXPIRED] += count
            expired_count += count
        else:
            expiry_counts[_EXPIRY_ACTIVE] += count

        if (
            selected_budget.max_expiry_days is not None
            and evaluation_date is not None
            and exception.expires_on is not None
        ):
            expiry_distance = (exception.expires_on - evaluation_date).days
            if expiry_distance > selected_budget.max_expiry_days and (
                max_expiry_distance is None or expiry_distance > max_expiry_distance
            ):
                max_expiry_distance = expiry_distance

    violations: list[ExceptionBudgetViolation] = []
    if total_count > selected_budget.max_total:
        violations.append(
            ExceptionBudgetViolation(
                metric="total",
                consumed=total_count,
                limit=selected_budget.max_total,
                comparison="max",
            )
        )
    _append_dimension_violations(
        violations,
        metric="severity",
        counts=severity_counts,
        limits=selected_budget.max_by_severity,
    )
    _append_dimension_violations(
        violations,
        metric="scope",
        counts=scope_counts,
        limits=selected_budget.max_by_scope,
    )
    _append_dimension_violations(
        violations,
        metric="policy_fingerprint",
        counts=policy_counts,
        limits=selected_budget.max_by_policy_fingerprint,
    )
    if unbounded_count:
        violations.append(
            ExceptionBudgetViolation(
                metric="unbounded_exception",
                consumed=unbounded_count,
                limit=0,
                comparison="must_be_zero",
            )
        )
    if expired_count:
        violations.append(
            ExceptionBudgetViolation(
                metric="expired_exception",
                consumed=expired_count,
                limit=0,
                comparison="must_be_zero",
            )
        )
    if selected_budget.max_expiry_days is not None:
        if evaluation_date is None and total_count:
            violations.append(
                ExceptionBudgetViolation(
                    metric="expiry_evaluation_date",
                    consumed=total_count,
                    limit=0,
                    comparison="required",
                )
            )
        elif max_expiry_distance is not None:
            violations.append(
                ExceptionBudgetViolation(
                    metric="expiry_window",
                    consumed=max_expiry_distance,
                    limit=selected_budget.max_expiry_days,
                    comparison="max_days",
                )
            )

    verdict = ExceptionBudgetVerdict(
        allowed=not violations,
        budget=selected_budget.to_dict(),
        total_count=total_count,
        counts_by_severity=MappingProxyType(dict(sorted(severity_counts.items()))),
        counts_by_scope=MappingProxyType(dict(sorted(scope_counts.items()))),
        counts_by_expiry=MappingProxyType(dict(sorted(expiry_counts.items()))),
        counts_by_policy_fingerprint=MappingProxyType(
            dict(sorted(policy_counts.items()))
        ),
        unbounded_count=unbounded_count,
        expired_count=expired_count,
        evaluation_date=evaluation_date.isoformat() if evaluation_date else None,
        violations=tuple(violations),
    )
    if strict and violations:
        raise ExceptionBudgetExceeded(verdict)
    return verdict


def check_exception_budget(
    exceptions: Iterable[PrivacyException | Mapping[str, Any]]
    | PrivacyException
    | Mapping[str, Any],
    budget: ExceptionBudget | Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
) -> ExceptionBudgetVerdict:
    """Evaluate exceptions and raise when the privacy budget is denied."""

    return evaluate_exception_budget(exceptions, budget, as_of=as_of, strict=True)


def _coerce_budget(
    budget: ExceptionBudget | Mapping[str, Any],
) -> ExceptionBudget:
    if isinstance(budget, ExceptionBudget):
        return budget
    if isinstance(budget, Mapping):
        return ExceptionBudget.from_mapping(budget)
    raise TypeError("budget must be an ExceptionBudget or mapping")


def _iter_exceptions(
    exceptions: Iterable[PrivacyException | Mapping[str, Any]]
    | PrivacyException
    | Mapping[str, Any],
) -> Iterable[PrivacyException]:
    if isinstance(exceptions, PrivacyException):
        return (exceptions,)
    if isinstance(exceptions, Mapping):
        candidates: Iterable[Any] = (exceptions,)
    elif isinstance(exceptions, str) or not isinstance(exceptions, Iterable):
        candidates = (None,)
    else:
        candidates = exceptions

    normalized: list[PrivacyException] = []
    for candidate in candidates:
        if isinstance(candidate, PrivacyException):
            normalized.append(candidate)
            continue
        if isinstance(candidate, Mapping):
            try:
                normalized.append(PrivacyException.from_mapping(candidate))
            except (TypeError, ValueError):
                normalized.append(
                    PrivacyException(
                        None,
                        None,
                        None,
                        None,
                        count=1,
                        unbounded=True,
                    )
                )
            continue
        normalized.append(
            PrivacyException(None, None, None, None, count=1, unbounded=True)
        )
    return tuple(normalized)


def _append_dimension_violations(
    violations: list[ExceptionBudgetViolation],
    *,
    metric: str,
    counts: Mapping[str, int],
    limits: Mapping[str, int],
) -> None:
    for key in sorted(limits):
        consumed = counts.get(key, 0)
        limit = limits[key]
        if consumed > limit:
            violations.append(
                ExceptionBudgetViolation(
                    metric=metric,
                    key=key,
                    consumed=consumed,
                    limit=limit,
                    comparison="max",
                )
            )


evaluate_privacy_exception_budget = evaluate_exception_budget
check_privacy_exception_budget = check_exception_budget
