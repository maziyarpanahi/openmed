"""Medication-status contradiction checks for clinical NLI pairs.

The precheck compares caller-paired medication records before model inference.
It recognizes explicit medication states and absolute event times, but never
infers a recommended state or resolves an ambiguous regimen transition.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

from openmed.clinical.med_reconciliation import normalize_medication_timestamp

NLI_MEDICATION_STATUS_SCHEMA_VERSION: Final[int] = 1


class MedicationStatusPrecheckError(ValueError):
    """Raised when medication precheck input violates the safe contract."""


class MedicationClaimStatus(str, Enum):
    """Controlled medication states accepted by the NLI precheck."""

    STARTED = "started"
    CONTINUED = "continued"
    HELD = "held"
    CHANGED = "changed"
    STOPPED = "stopped"
    HISTORICAL = "historical"
    UNCERTAIN = "uncertain"


class MedicationStatusPrecheckStatus(str, Enum):
    """Outcome of the pre-model medication-status check."""

    COMPATIBLE = "compatible"
    CONTRADICTION = "contradiction"
    REVIEW_REQUIRED = "review_required"
    NOT_APPLICABLE = "not_applicable"


class MedicationStatusReason(str, Enum):
    """Value-free reason for a contradiction or review escalation."""

    STATUS_CONTRADICTION = "status_contradiction"
    UNCERTAIN_STATUS = "uncertain_status"
    AMBIGUOUS_REGIMEN_TRANSITION = "ambiguous_regimen_transition"
    EVENT_TIME_MISMATCH = "event_time_mismatch"
    MISSING_EVENT_TIME = "missing_event_time"


class EventTimeRelation(str, Enum):
    """Privacy-safe relationship between two normalized event times."""

    SAME = "same"
    PREMISE_BEFORE = "premise_before"
    PREMISE_AFTER = "premise_after"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


_STATUS_ALIASES: dict[str, MedicationClaimStatus] = {
    "start": MedicationClaimStatus.STARTED,
    "started": MedicationClaimStatus.STARTED,
    "starting": MedicationClaimStatus.STARTED,
    "initiate": MedicationClaimStatus.STARTED,
    "initiated": MedicationClaimStatus.STARTED,
    "begin": MedicationClaimStatus.STARTED,
    "began": MedicationClaimStatus.STARTED,
    "continue": MedicationClaimStatus.CONTINUED,
    "continued": MedicationClaimStatus.CONTINUED,
    "continuing": MedicationClaimStatus.CONTINUED,
    "active": MedicationClaimStatus.CONTINUED,
    "ongoing": MedicationClaimStatus.CONTINUED,
    "resume": MedicationClaimStatus.CONTINUED,
    "resumed": MedicationClaimStatus.CONTINUED,
    "hold": MedicationClaimStatus.HELD,
    "held": MedicationClaimStatus.HELD,
    "withheld": MedicationClaimStatus.HELD,
    "on hold": MedicationClaimStatus.HELD,
    "change": MedicationClaimStatus.CHANGED,
    "changed": MedicationClaimStatus.CHANGED,
    "increase": MedicationClaimStatus.CHANGED,
    "increased": MedicationClaimStatus.CHANGED,
    "decrease": MedicationClaimStatus.CHANGED,
    "decreased": MedicationClaimStatus.CHANGED,
    "titrated": MedicationClaimStatus.CHANGED,
    "stop": MedicationClaimStatus.STOPPED,
    "stopped": MedicationClaimStatus.STOPPED,
    "discontinue": MedicationClaimStatus.STOPPED,
    "discontinued": MedicationClaimStatus.STOPPED,
    "inactive": MedicationClaimStatus.STOPPED,
    "completed": MedicationClaimStatus.STOPPED,
    "historical": MedicationClaimStatus.HISTORICAL,
    "history": MedicationClaimStatus.HISTORICAL,
    "past": MedicationClaimStatus.HISTORICAL,
    "former": MedicationClaimStatus.HISTORICAL,
    "uncertain": MedicationClaimStatus.UNCERTAIN,
    "unknown": MedicationClaimStatus.UNCERTAIN,
    "possible": MedicationClaimStatus.UNCERTAIN,
    "possibly": MedicationClaimStatus.UNCERTAIN,
    "planned": MedicationClaimStatus.UNCERTAIN,
    "considered": MedicationClaimStatus.UNCERTAIN,
}

_ACTIVE_STATUSES = {
    MedicationClaimStatus.STARTED,
    MedicationClaimStatus.CONTINUED,
}
_INACTIVE_STATUSES = {
    MedicationClaimStatus.HELD,
    MedicationClaimStatus.STOPPED,
    MedicationClaimStatus.HISTORICAL,
}
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, repr=False)
class MedicationStatusClaim:
    """Structured medication state for one side of an NLI pair.

    Raw status strings, medication identities, and event times are excluded
    from ``repr`` and every serialized precheck report.
    """

    status: object | None = field(default=None, repr=False)
    event_time: object | None = field(default=None, repr=False)
    medication_key: str | None = field(default=None, repr=False)
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        if self.medication_key is not None and (
            type(self.medication_key) is not str or not self.medication_key.strip()
        ):
            raise MedicationStatusPrecheckError("invalid medication identity")
        if (self.source_start is None) != (self.source_end is None):
            raise MedicationStatusPrecheckError("invalid medication source span")
        if self.source_start is not None and (
            type(self.source_start) is not int
            or type(self.source_end) is not int
            or self.source_start < 0
            or self.source_end <= self.source_start
        ):
            raise MedicationStatusPrecheckError("invalid medication source span")

    def __repr__(self) -> str:
        """Return a value-free representation safe for diagnostics."""

        return (
            "MedicationStatusClaim("
            f"has_status={self.status is not None}, "
            f"has_event_time={self.event_time is not None}, "
            f"source_span={self.source_span!r})"
        )

    @property
    def source_span(self) -> tuple[int, int] | None:
        """Return the half-open source span when supplied."""

        if self.source_start is None or self.source_end is None:
            return None
        return self.source_start, self.source_end

    @classmethod
    def from_obj(
        cls,
        value: MedicationStatusClaim | Mapping[str, Any],
    ) -> MedicationStatusClaim:
        """Coerce common medication-record keys without retaining source text."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise MedicationStatusPrecheckError("medication claim must be a mapping")
        start, end = _coerce_span(value)
        return cls(
            status=_first_present(value, ("status", "state", "action")),
            event_time=_first_present(
                value,
                ("event_time", "effective_time", "timestamp", "normalized_timestamp"),
            ),
            medication_key=_first_present(
                value,
                ("medication_key", "ingredient_id", "concept_id", "code"),
            ),
            source_start=start,
            source_end=end,
        )


@dataclass(frozen=True)
class MedicationStatusEvidence:
    """Privacy-safe evidence for one medication-status precheck finding."""

    reason: MedicationStatusReason
    premise_status: MedicationClaimStatus
    hypothesis_status: MedicationClaimStatus
    time_relation: EventTimeRelation
    premise_fingerprint: str
    hypothesis_fingerprint: str
    premise_span: tuple[int, int] | None
    hypothesis_span: tuple[int, int] | None

    def to_dict(self) -> dict[str, object]:
        """Return controlled statuses and provenance without raw identifiers."""

        return {
            "reason": self.reason.value,
            "premise_status": self.premise_status.value,
            "hypothesis_status": self.hypothesis_status.value,
            "time_relation": self.time_relation.value,
            "premise_fingerprint": self.premise_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "premise_span": list(self.premise_span) if self.premise_span else None,
            "hypothesis_span": list(self.hypothesis_span)
            if self.hypothesis_span
            else None,
        }


@dataclass(frozen=True)
class MedicationStatusPrecheckResult:
    """Fail-closed result returned before local NLI model inference."""

    status: MedicationStatusPrecheckStatus
    evidence: tuple[MedicationStatusEvidence, ...]
    schema_version: int = NLI_MEDICATION_STATUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NLI_MEDICATION_STATUS_SCHEMA_VERSION:
            raise MedicationStatusPrecheckError(
                "unsupported medication-status precheck schema"
            )
        if not isinstance(self.status, MedicationStatusPrecheckStatus):
            raise MedicationStatusPrecheckError("invalid medication precheck status")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, MedicationStatusEvidence) for item in self.evidence
        ):
            raise MedicationStatusPrecheckError("invalid medication precheck evidence")
        if (
            self.status
            in {
                MedicationStatusPrecheckStatus.CONTRADICTION,
                MedicationStatusPrecheckStatus.REVIEW_REQUIRED,
            }
            and not self.evidence
        ):
            raise MedicationStatusPrecheckError(
                "medication precheck finding requires evidence"
            )
        if (
            self.status
            in {
                MedicationStatusPrecheckStatus.COMPATIBLE,
                MedicationStatusPrecheckStatus.NOT_APPLICABLE,
            }
            and self.evidence
        ):
            raise MedicationStatusPrecheckError(
                "medication precheck status forbids evidence"
            )

    @property
    def contradiction(self) -> bool:
        """Return whether explicit same-time states prove a contradiction."""

        return self.status is MedicationStatusPrecheckStatus.CONTRADICTION

    @property
    def requires_review(self) -> bool:
        """Return whether transition ambiguity requires human review."""

        return self.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED

    @property
    def inference_allowed(self) -> bool:
        """Return whether a local NLI model may evaluate the pair."""

        return self.status in {
            MedicationStatusPrecheckStatus.COMPATIBLE,
            MedicationStatusPrecheckStatus.NOT_APPLICABLE,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report with no raw medication values."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "contradiction": self.contradiction,
            "requires_review": self.requires_review,
            "inference_allowed": self.inference_allowed,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class _PreparedClaim:
    raw: MedicationStatusClaim
    status: MedicationClaimStatus
    event_time: str | None
    sortable_time: datetime | None


def normalize_nli_medication_status(value: object | None) -> MedicationClaimStatus:
    """Normalize an explicit medication status without choosing a desired state.

    Missing status is ``uncertain``. Unrecognized text fails with a value-free
    exception instead of being copied into an error or report.
    """

    if value is None:
        return MedicationClaimStatus.UNCERTAIN
    if isinstance(value, MedicationClaimStatus):
        return value
    if isinstance(value, Mapping):
        value = _first_present(value, ("status", "state", "action", "value"))
        if value is None:
            return MedicationClaimStatus.UNCERTAIN
    if type(value) is not str:
        raise MedicationStatusPrecheckError("unsupported medication status")
    normalized = _normalize_text(value)
    if not normalized:
        return MedicationClaimStatus.UNCERTAIN
    status = _STATUS_ALIASES.get(normalized)
    if status is None:
        raise MedicationStatusPrecheckError("unsupported medication status")
    return status


def medication_status_contradiction_precheck(
    premise: MedicationStatusClaim | Mapping[str, Any],
    hypothesis: MedicationStatusClaim | Mapping[str, Any],
) -> MedicationStatusPrecheckResult:
    """Compare explicit medication status and event time before NLI inference.

    Clear active/inactive disagreement is a contradiction only at the same
    normalized event time. Ordered, missing-time, changed, or uncertain states
    are escalated for review; the precheck never selects the newer or desired
    medication state.
    """

    premise_claim = MedicationStatusClaim.from_obj(premise)
    hypothesis_claim = MedicationStatusClaim.from_obj(hypothesis)
    if (
        premise_claim.medication_key is not None
        and hypothesis_claim.medication_key is not None
        and _normalize_text(premise_claim.medication_key)
        != _normalize_text(hypothesis_claim.medication_key)
    ):
        return MedicationStatusPrecheckResult(
            status=MedicationStatusPrecheckStatus.NOT_APPLICABLE,
            evidence=(),
        )

    prepared_premise = _prepare_claim(premise_claim)
    prepared_hypothesis = _prepare_claim(hypothesis_claim)
    relation = _time_relation(prepared_premise, prepared_hypothesis)
    premise_status = prepared_premise.status
    hypothesis_status = prepared_hypothesis.status

    if MedicationClaimStatus.UNCERTAIN in {premise_status, hypothesis_status}:
        return _finding_result(
            MedicationStatusPrecheckStatus.REVIEW_REQUIRED,
            MedicationStatusReason.UNCERTAIN_STATUS,
            prepared_premise,
            prepared_hypothesis,
            relation,
        )
    if MedicationClaimStatus.CHANGED in {premise_status, hypothesis_status}:
        return _finding_result(
            MedicationStatusPrecheckStatus.REVIEW_REQUIRED,
            MedicationStatusReason.AMBIGUOUS_REGIMEN_TRANSITION,
            prepared_premise,
            prepared_hypothesis,
            relation,
        )
    if premise_status == hypothesis_status:
        if relation in {
            EventTimeRelation.PREMISE_BEFORE,
            EventTimeRelation.PREMISE_AFTER,
        }:
            return _finding_result(
                MedicationStatusPrecheckStatus.REVIEW_REQUIRED,
                MedicationStatusReason.EVENT_TIME_MISMATCH,
                prepared_premise,
                prepared_hypothesis,
                relation,
            )
        return MedicationStatusPrecheckResult(
            status=MedicationStatusPrecheckStatus.COMPATIBLE,
            evidence=(),
        )

    if relation is EventTimeRelation.SAME and _statuses_conflict(
        premise_status,
        hypothesis_status,
    ):
        return _finding_result(
            MedicationStatusPrecheckStatus.CONTRADICTION,
            MedicationStatusReason.STATUS_CONTRADICTION,
            prepared_premise,
            prepared_hypothesis,
            relation,
        )

    if relation in {EventTimeRelation.UNKNOWN, EventTimeRelation.PARTIAL}:
        reason = MedicationStatusReason.MISSING_EVENT_TIME
    else:
        reason = MedicationStatusReason.AMBIGUOUS_REGIMEN_TRANSITION
    return _finding_result(
        MedicationStatusPrecheckStatus.REVIEW_REQUIRED,
        reason,
        prepared_premise,
        prepared_hypothesis,
        relation,
    )


def check_medication_status_contradiction(
    premise: MedicationStatusClaim | Mapping[str, Any],
    hypothesis: MedicationStatusClaim | Mapping[str, Any],
) -> MedicationStatusPrecheckResult:
    """Alias for :func:`medication_status_contradiction_precheck`."""

    return medication_status_contradiction_precheck(premise, hypothesis)


def _prepare_claim(claim: MedicationStatusClaim) -> _PreparedClaim:
    status = normalize_nli_medication_status(claim.status)
    try:
        event_time = normalize_medication_timestamp(claim.event_time)
        sortable_time = _sortable_timestamp(event_time)
    except Exception:
        raise MedicationStatusPrecheckError("invalid medication event time") from None
    return _PreparedClaim(
        raw=claim,
        status=status,
        event_time=event_time,
        sortable_time=sortable_time,
    )


def _sortable_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    if re.fullmatch(r"\d{4}", value):
        return datetime(int(value), 1, 1)
    if match := re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{2})", value):
        return datetime(int(match.group("year")), int(match.group("month")), 1)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise MedicationStatusPrecheckError("invalid medication event time") from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _time_relation(
    premise: _PreparedClaim,
    hypothesis: _PreparedClaim,
) -> EventTimeRelation:
    if premise.event_time is None and hypothesis.event_time is None:
        return EventTimeRelation.UNKNOWN
    if premise.event_time is None or hypothesis.event_time is None:
        return EventTimeRelation.PARTIAL
    if premise.event_time == hypothesis.event_time:
        return EventTimeRelation.SAME
    if premise.sortable_time is None or hypothesis.sortable_time is None:
        return EventTimeRelation.PARTIAL
    if premise.sortable_time == hypothesis.sortable_time:
        return EventTimeRelation.PARTIAL
    if premise.sortable_time < hypothesis.sortable_time:
        return EventTimeRelation.PREMISE_BEFORE
    return EventTimeRelation.PREMISE_AFTER


def _statuses_conflict(
    premise: MedicationClaimStatus,
    hypothesis: MedicationClaimStatus,
) -> bool:
    return (premise in _ACTIVE_STATUSES and hypothesis in _INACTIVE_STATUSES) or (
        hypothesis in _ACTIVE_STATUSES and premise in _INACTIVE_STATUSES
    )


def _finding_result(
    status: MedicationStatusPrecheckStatus,
    reason: MedicationStatusReason,
    premise: _PreparedClaim,
    hypothesis: _PreparedClaim,
    relation: EventTimeRelation,
) -> MedicationStatusPrecheckResult:
    evidence = MedicationStatusEvidence(
        reason=reason,
        premise_status=premise.status,
        hypothesis_status=hypothesis.status,
        time_relation=relation,
        premise_fingerprint=_fingerprint(premise.raw),
        hypothesis_fingerprint=_fingerprint(hypothesis.raw),
        premise_span=premise.raw.source_span,
        hypothesis_span=hypothesis.raw.source_span,
    )
    return MedicationStatusPrecheckResult(status=status, evidence=(evidence,))


def _fingerprint(claim: MedicationStatusClaim) -> str:
    try:
        payload = {
            "status": _stable_value(claim.status),
            "event_time": _stable_value(claim.event_time),
            "medication_key": claim.medication_key,
            "source_span": claim.source_span,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    except Exception:
        raise MedicationStatusPrecheckError("invalid medication provenance") from None
    return hashlib.sha256(b"openmed:nli-medication-status:v1\0" + encoded).hexdigest()


def _stable_value(value: object | None) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return _WHITESPACE_RE.sub(" ", text)


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
        raise MedicationStatusPrecheckError("invalid medication source span")
    return mapping.get("source_start", mapping.get("start")), mapping.get(
        "source_end", mapping.get("end")
    )


__all__ = [
    "NLI_MEDICATION_STATUS_SCHEMA_VERSION",
    "EventTimeRelation",
    "MedicationClaimStatus",
    "MedicationStatusClaim",
    "MedicationStatusEvidence",
    "MedicationStatusPrecheckError",
    "MedicationStatusPrecheckResult",
    "MedicationStatusPrecheckStatus",
    "MedicationStatusReason",
    "check_medication_status_contradiction",
    "medication_status_contradiction_precheck",
    "normalize_nli_medication_status",
]
