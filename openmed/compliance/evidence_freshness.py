"""Deterministic, aggregate-only freshness checks for privacy evidence.

Evidence freshness is a local release control.  The gate accepts typed
``datetime`` values, policy versions, and opaque supersession references; it
never fetches evidence or derives a clock from the host.  A caller must inject
the evaluation time so that the result can be replayed exactly.

Only aggregate counts leave the evaluator.  Evidence references and unknown
mapping fields are used for comparison but are never copied into reports,
exceptions, or serialized diagnostics.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Final

_UTC: Final = timezone.utc
_SAFE_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_WILDCARD_LIMIT: Final = "*"

MISSING_EVIDENCE: Final = "missing_evidence"
MISSING_EVIDENCE_ID: Final = "missing_evidence_id"
INVALID_EVIDENCE_ID: Final = "invalid_evidence_id"
DUPLICATE_EVIDENCE_ID: Final = "duplicate_evidence_id"
MISSING_EVIDENCE_TYPE: Final = "missing_evidence_type"
INVALID_EVIDENCE_TYPE: Final = "invalid_evidence_type"
UNKNOWN_EVIDENCE_TYPE: Final = "unknown_evidence_type"
MISSING_TIMESTAMP: Final = "missing_timestamp"
INVALID_TIMESTAMP: Final = "invalid_timestamp"
FUTURE_TIMESTAMP: Final = "future_timestamp"
EXPIRED_EVIDENCE: Final = "expired_evidence"
MISSING_POLICY_VERSION: Final = "missing_policy_version"
INVALID_POLICY_VERSION: Final = "invalid_policy_version"
POLICY_MISMATCH: Final = "policy_mismatch"
INVALID_SUPERSESSION_LINK: Final = "invalid_supersession_link"
SUPERSEDED_EVIDENCE: Final = "superseded_evidence"

_REASON_CODES: Final = frozenset(
    {
        MISSING_EVIDENCE,
        MISSING_EVIDENCE_ID,
        INVALID_EVIDENCE_ID,
        DUPLICATE_EVIDENCE_ID,
        MISSING_EVIDENCE_TYPE,
        INVALID_EVIDENCE_TYPE,
        UNKNOWN_EVIDENCE_TYPE,
        MISSING_TIMESTAMP,
        INVALID_TIMESTAMP,
        FUTURE_TIMESTAMP,
        EXPIRED_EVIDENCE,
        MISSING_POLICY_VERSION,
        INVALID_POLICY_VERSION,
        POLICY_MISMATCH,
        INVALID_SUPERSESSION_LINK,
        SUPERSEDED_EVIDENCE,
    }
)

__all__ = [
    "DUPLICATE_EVIDENCE_ID",
    "EvidenceAgePolicy",
    "EXPIRED_EVIDENCE",
    "EvidenceFreshnessError",
    "EvidenceFreshnessPolicy",
    "EvidenceFreshnessReport",
    "EvidenceRecord",
    "FUTURE_TIMESTAMP",
    "INVALID_EVIDENCE_ID",
    "INVALID_EVIDENCE_TYPE",
    "INVALID_POLICY_VERSION",
    "INVALID_SUPERSESSION_LINK",
    "INVALID_TIMESTAMP",
    "MISSING_EVIDENCE",
    "MISSING_EVIDENCE_ID",
    "MISSING_EVIDENCE_TYPE",
    "MISSING_POLICY_VERSION",
    "MISSING_TIMESTAMP",
    "POLICY_MISMATCH",
    "PrivacyEvidence",
    "SUPERSEDED_EVIDENCE",
    "UNKNOWN_EVIDENCE_TYPE",
    "assert_evidence_freshness",
    "check_evidence_freshness",
    "evaluate_evidence_freshness",
]


def _safe_token(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_TOKEN_RE.fullmatch(value) is not None


def _require_token(value: Any, *, name: str) -> str:
    if not _safe_token(value):
        raise ValueError(f"{name} must be a non-empty opaque token")
    return value


def _require_duration(value: Any, *, name: str) -> timedelta:
    if type(value) is not timedelta:
        raise TypeError(f"{name} must be a datetime.timedelta")
    if value < timedelta(0):
        raise ValueError(f"{name} must not be negative")
    return value


def _read_first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _parse_datetime(value: Any) -> datetime | None:
    """Parse one aware UTC datetime without assuming a local timezone."""

    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            candidate = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if candidate.tzinfo is None or candidate.utcoffset() is None:
        return None
    return candidate.astimezone(_UTC)


@dataclass(frozen=True, init=False)
class EvidenceFreshnessPolicy:
    """Typed age limits for each privacy-evidence kind.

    ``age_limits`` maps an opaque evidence kind to a non-negative
    :class:`datetime.timedelta`.  The optional ``"*"`` entry is a typed
    fallback for kinds that share one limit.  A specific kind always wins over
    the fallback.  Policy versions are compared exactly; no network lookup or
    version discovery is performed.
    """

    policy_version: str
    age_limits: Mapping[str, timedelta]

    def __init__(
        self,
        policy_version: str,
        age_limits: Mapping[str, timedelta] | None = None,
        *,
        max_age_by_type: Mapping[str, timedelta] | None = None,
    ) -> None:
        if age_limits is not None and max_age_by_type is not None:
            raise TypeError("provide only one age-limit mapping")
        limits = age_limits if age_limits is not None else max_age_by_type
        if limits is None:
            limits = {}
        if not isinstance(limits, Mapping):
            raise TypeError("age_limits must be a mapping")

        version = _require_token(policy_version, name="policy_version")
        normalized: dict[str, timedelta] = {}
        for evidence_type, max_age in limits.items():
            if evidence_type != _WILDCARD_LIMIT:
                _require_token(evidence_type, name="evidence_type")
            normalized[evidence_type] = _require_duration(
                max_age,
                name="age limit",
            )

        object.__setattr__(self, "policy_version", version)
        object.__setattr__(
            self,
            "age_limits",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceFreshnessPolicy":
        """Build a policy from an allow-listed mapping.

        Age-limit values remain typed: callers must provide
        :class:`datetime.timedelta` instances rather than untyped day counts
        or strings.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("policy must be a mapping")
        return cls(
            _read_first(payload, "policy_version", "version"),
            _read_first(payload, "age_limits", "max_age_by_type", "limits"),
        )

    def max_age_for(self, evidence_type: str) -> timedelta | None:
        """Return the exact or wildcard age limit for one evidence kind."""

        return self.age_limits.get(
            evidence_type,
            self.age_limits.get(_WILDCARD_LIMIT),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic policy description without input payloads."""

        return {
            "policy_version": self.policy_version,
            "age_limits_seconds": {
                evidence_type: max_age.total_seconds()
                for evidence_type, max_age in self.age_limits.items()
            },
        }


# A concise alias for callers that use the age-policy terminology from the
# release specification.
EvidenceAgePolicy = EvidenceFreshnessPolicy


@dataclass(frozen=True, init=False)
class EvidenceRecord:
    """One privacy-evidence descriptor used by the freshness gate.

    ``evidence_id`` is an opaque local reference, not a patient or document
    identifier.  The evaluator accepts common serialized aliases such as
    ``timestamp`` and ``kind`` but ignores all other mapping fields.
    """

    evidence_id: str | None
    evidence_type: str | None
    generated_at: datetime | str | None
    policy_version: str | None
    supersedes: str | None
    superseded_by: str | None

    def __init__(
        self,
        evidence_id: str | None = None,
        evidence_type: str | None = None,
        generated_at: datetime | str | None = None,
        policy_version: str | None = None,
        supersedes: str | None = None,
        *,
        timestamp: datetime | str | None = None,
        observed_at: datetime | str | None = None,
        kind: str | None = None,
        record_id: str | None = None,
        policy: str | None = None,
        superseded_by: str | None = None,
        supersession_link: str | Mapping[str, Any] | None = None,
    ) -> None:
        if generated_at is not None and (
            timestamp is not None or observed_at is not None
        ):
            raise TypeError("provide only one evidence timestamp")
        if evidence_type is not None and kind is not None:
            raise TypeError("provide only one evidence type")
        if evidence_id is not None and record_id is not None:
            raise TypeError("provide only one evidence reference")
        if policy_version is not None and policy is not None:
            raise TypeError("provide only one policy version")
        if supersession_link is not None:
            if supersedes is not None or superseded_by is not None:
                raise TypeError("provide only one supersession link")
            if isinstance(supersession_link, Mapping):
                supersedes = _read_first(
                    supersession_link,
                    "supersedes",
                    "prior",
                )
                superseded_by = _read_first(
                    supersession_link,
                    "superseded_by",
                    "replacement",
                )
            else:
                superseded_by = supersession_link

        resolved_timestamp = generated_at
        if timestamp is not None:
            resolved_timestamp = timestamp
        elif observed_at is not None:
            resolved_timestamp = observed_at

        object.__setattr__(
            self,
            "evidence_id",
            evidence_id if evidence_id is not None else record_id,
        )
        object.__setattr__(
            self,
            "evidence_type",
            evidence_type if evidence_type is not None else kind,
        )
        object.__setattr__(self, "generated_at", resolved_timestamp)
        object.__setattr__(
            self,
            "policy_version",
            policy_version if policy_version is not None else policy,
        )
        object.__setattr__(self, "supersedes", supersedes)
        object.__setattr__(self, "superseded_by", superseded_by)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        """Read only the safe, typed descriptor fields from a mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("evidence must be a mapping")

        supersedes = _read_first(payload, "supersedes", "supersedes_id")
        superseded_by = _read_first(
            payload,
            "superseded_by",
            "superseded_by_id",
        )
        link = _read_first(payload, "supersession_link", "supersession")
        if isinstance(link, Mapping):
            supersedes = supersedes or _read_first(link, "supersedes", "prior")
            superseded_by = superseded_by or _read_first(
                link,
                "superseded_by",
                "replacement",
            )
        elif link is not None and superseded_by is None:
            superseded_by = link

        return cls(
            evidence_id=_read_first(payload, "evidence_id", "id", "record_id"),
            evidence_type=_read_first(
                payload,
                "evidence_type",
                "type",
                "kind",
            ),
            generated_at=_read_first(
                payload,
                "generated_at",
                "timestamp",
                "observed_at",
                "created_at",
            ),
            policy_version=_read_first(
                payload,
                "policy_version",
                "policy",
            ),
            supersedes=supersedes,
            superseded_by=superseded_by,
        )

    @property
    def timestamp(self) -> datetime | str | None:
        """Return the canonical timestamp under the common serialized name."""

        return self.generated_at

    @property
    def kind(self) -> str | None:
        """Return the evidence kind under the common serialized name."""

        return self.evidence_type

    def to_dict(self) -> dict[str, Any]:
        """Return only the typed descriptor fields.

        This method is intended for local persistence of an already validated
        descriptor.  Freshness reports never include this record-level data.
        """

        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "generated_at": self.generated_at,
            "policy_version": self.policy_version,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
        }


# A descriptive alias for callers that use privacy-evidence terminology.
PrivacyEvidence = EvidenceRecord


@dataclass(frozen=True)
class EvidenceFreshnessReport:
    """Aggregate result of a freshness evaluation.

    Each input record contributes at most one primary reason.  This keeps the
    diagnostics counts-only and makes totals stable even when one record has
    more than one malformed field.
    """

    policy_version: str
    total_count: int
    current_count: int
    rejected_count: int
    reason_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _safe_token(self.policy_version):
            raise ValueError("policy_version must be a non-empty opaque token")
        for name, value in (
            ("total_count", self.total_count),
            ("current_count", self.current_count),
            ("rejected_count", self.rejected_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.current_count + self.rejected_count != self.total_count:
            raise ValueError("freshness counts do not add up")
        if not isinstance(self.reason_counts, Mapping):
            raise TypeError("reason_counts must be a mapping")
        normalized: dict[str, int] = {}
        for reason, count in self.reason_counts.items():
            if reason not in _REASON_CODES:
                raise ValueError("reason_counts contains an unsupported reason")
            if type(count) is not int or count < 0:
                raise ValueError("reason counts must be non-negative integers")
            if count:
                normalized[reason] = count
        object.__setattr__(
            self,
            "reason_counts",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @property
    def passed(self) -> bool:
        """Whether all supplied evidence is current and at least one exists."""

        return self.total_count > 0 and self.rejected_count == 0

    @property
    def fresh(self) -> bool:
        """Alias for :attr:`passed` for gate-oriented call sites."""

        return self.passed

    @property
    def accepted_count(self) -> int:
        """Return the number of records that passed the gate."""

        return self.current_count

    @property
    def diagnostic_counts(self) -> Mapping[str, int]:
        """Return the immutable counts-only diagnostic mapping."""

        return self.reason_counts

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report with no record-level fields."""

        return {
            "policy_version": self.policy_version,
            "total_count": self.total_count,
            "current_count": self.current_count,
            "rejected_count": self.rejected_count,
            "reason_counts": dict(self.reason_counts),
            "passed": self.passed,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize only aggregate freshness diagnostics."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def failure_message(self) -> str:
        """Return a safe exception message containing counts only."""

        reasons = ", ".join(
            f"{reason}={count}" for reason, count in self.reason_counts.items()
        )
        return (
            "privacy evidence freshness gate failed: "
            f"rejected={self.rejected_count}, reasons={reasons or 'none'}"
        )


class EvidenceFreshnessError(ValueError):
    """Raised by :func:`assert_evidence_freshness` with counts-only details."""

    def __init__(self, report: EvidenceFreshnessReport) -> None:
        if not isinstance(report, EvidenceFreshnessReport):
            raise TypeError("report must be an EvidenceFreshnessReport")
        self.report = report
        super().__init__(report.failure_message())


def _coerce_policy(
    policy: EvidenceFreshnessPolicy | Mapping[str, Any],
) -> EvidenceFreshnessPolicy:
    if isinstance(policy, EvidenceFreshnessPolicy):
        return policy
    if isinstance(policy, Mapping):
        return EvidenceFreshnessPolicy.from_mapping(policy)
    raise TypeError("policy must be an EvidenceFreshnessPolicy")


def _coerce_record(record: EvidenceRecord | Mapping[str, Any]) -> EvidenceRecord:
    if isinstance(record, EvidenceRecord):
        return record
    if isinstance(record, Mapping):
        return EvidenceRecord.from_mapping(record)
    raise TypeError("evidence records must be EvidenceRecord values or mappings")


def _coerce_records(
    records: Iterable[EvidenceRecord | Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[EvidenceRecord]:
    if records is None:
        return []
    if isinstance(records, Mapping):
        return [_coerce_record(records)]
    if isinstance(records, (str, bytes)):
        raise TypeError("evidence records must be iterable records")
    return [_coerce_record(record) for record in records]


def _resolve_evaluation_time(
    *,
    as_of: datetime | str | None,
    now: datetime | str | None,
    clock: Callable[[], datetime | str] | Any | None,
) -> datetime:
    supplied = sum(value is not None for value in (as_of, now, clock))
    if supplied != 1:
        raise ValueError("provide exactly one injected as_of, now, or clock")

    value: Any
    if clock is not None:
        if callable(clock):
            value = clock()
        else:
            clock_now = getattr(clock, "now", None)
            if not callable(clock_now):
                raise TypeError("clock must be callable or expose now()")
            value = clock_now()
    else:
        value = as_of if as_of is not None else now

    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError("injected evaluation time must be an aware datetime")
    return parsed


def _link_target(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if not _safe_token(value):
        return None, True
    return value, False


def _primary_reason(
    record: EvidenceRecord,
    *,
    policy: EvidenceFreshnessPolicy,
    as_of: datetime,
    duplicate_ids: set[str],
    superseded_ids: set[str],
    invalid_link_indices: set[int],
    index: int,
) -> str | None:
    evidence_id = record.evidence_id
    if evidence_id is None:
        return MISSING_EVIDENCE_ID
    if not _safe_token(evidence_id):
        return INVALID_EVIDENCE_ID
    if evidence_id in duplicate_ids:
        return DUPLICATE_EVIDENCE_ID

    evidence_type = record.evidence_type
    if evidence_type is None:
        return MISSING_EVIDENCE_TYPE
    if not _safe_token(evidence_type):
        return INVALID_EVIDENCE_TYPE
    max_age = policy.max_age_for(evidence_type)
    if max_age is None:
        return UNKNOWN_EVIDENCE_TYPE

    if record.generated_at is None:
        return MISSING_TIMESTAMP
    generated_at = _parse_datetime(record.generated_at)
    if generated_at is None:
        return INVALID_TIMESTAMP
    if generated_at > as_of:
        return FUTURE_TIMESTAMP
    if as_of - generated_at > max_age:
        return EXPIRED_EVIDENCE

    policy_version = record.policy_version
    if policy_version is None:
        return MISSING_POLICY_VERSION
    if not _safe_token(policy_version):
        return INVALID_POLICY_VERSION
    if policy_version != policy.policy_version:
        return POLICY_MISMATCH

    if index in invalid_link_indices:
        return INVALID_SUPERSESSION_LINK
    if evidence_id in superseded_ids:
        return SUPERSEDED_EVIDENCE
    return None


def evaluate_evidence_freshness(
    records: Iterable[EvidenceRecord | Mapping[str, Any]] | Mapping[str, Any] | None,
    policy: EvidenceFreshnessPolicy | Mapping[str, Any],
    *,
    as_of: datetime | str | None = None,
    now: datetime | str | None = None,
    clock: Callable[[], datetime | str] | Any | None = None,
) -> EvidenceFreshnessReport:
    """Evaluate evidence age, policy binding, and supersession state.

    The evaluation time is deliberately injected through exactly one of
    ``as_of``, ``now``, or ``clock``.  Omitting it is an error rather than an
    implicit read of wall-clock time, which keeps release decisions
    deterministic and replayable.

    ``superseded_by`` marks a record as no longer eligible.  ``supersedes``
    marks an earlier record when that referenced record is present in the same
    input; the replacement remains eligible when the earlier artifact is kept
    in an external archive.  All link values are treated as opaque references.

    Args:
        records: Typed evidence descriptors or mappings containing only their
            descriptor fields.  Unknown mapping fields are ignored.
        policy: The expected policy version and typed ``timedelta`` age limits.
        as_of: Explicit aware evaluation time.
        now: Alias for ``as_of``.
        clock: Callable or object exposing ``now()`` that returns an aware
            datetime.  It is invoked once.

    Returns:
        A deterministic report with aggregate counts and reason codes only.
    """

    active_policy = _coerce_policy(policy)
    evaluation_time = _resolve_evaluation_time(
        as_of=as_of,
        now=now,
        clock=clock,
    )
    evidence = _coerce_records(records)
    if not evidence:
        return EvidenceFreshnessReport(
            policy_version=active_policy.policy_version,
            total_count=0,
            current_count=0,
            rejected_count=0,
            reason_counts={MISSING_EVIDENCE: 1},
        )

    id_to_indices: dict[str, list[int]] = {}
    for index, record in enumerate(evidence):
        if _safe_token(record.evidence_id):
            id_to_indices.setdefault(record.evidence_id, []).append(index)
    duplicate_ids = {
        evidence_id
        for evidence_id, indices in id_to_indices.items()
        if len(indices) > 1
    }

    superseded_ids: set[str] = set()
    invalid_link_indices: set[int] = set()
    for index, record in enumerate(evidence):
        supersedes, invalid_supersedes = _link_target(record.supersedes)
        superseded_by, invalid_superseded_by = _link_target(record.superseded_by)
        if invalid_supersedes or invalid_superseded_by:
            invalid_link_indices.add(index)
        if supersedes is not None and supersedes in id_to_indices:
            superseded_ids.add(supersedes)
        if superseded_by is not None:
            superseded_ids.add(record.evidence_id or "")

    reasons: Counter[str] = Counter()
    current_count = 0
    for index, record in enumerate(evidence):
        reason = _primary_reason(
            record,
            policy=active_policy,
            as_of=evaluation_time,
            duplicate_ids=duplicate_ids,
            superseded_ids=superseded_ids,
            invalid_link_indices=invalid_link_indices,
            index=index,
        )
        if reason is None:
            current_count += 1
        else:
            reasons[reason] += 1

    return EvidenceFreshnessReport(
        policy_version=active_policy.policy_version,
        total_count=len(evidence),
        current_count=current_count,
        rejected_count=len(evidence) - current_count,
        reason_counts=dict(reasons),
    )


def assert_evidence_freshness(
    records: Iterable[EvidenceRecord | Mapping[str, Any]] | Mapping[str, Any] | None,
    policy: EvidenceFreshnessPolicy | Mapping[str, Any],
    *,
    as_of: datetime | str | None = None,
    now: datetime | str | None = None,
    clock: Callable[[], datetime | str] | Any | None = None,
) -> EvidenceFreshnessReport:
    """Return a passing report or raise a counts-only freshness error."""

    report = evaluate_evidence_freshness(
        records,
        policy,
        as_of=as_of,
        now=now,
        clock=clock,
    )
    if not report.passed:
        raise EvidenceFreshnessError(report)
    return report


# The check name reads naturally in release-gate call sites while retaining a
# single implementation and identical counts-only behavior.
check_evidence_freshness = evaluate_evidence_freshness
