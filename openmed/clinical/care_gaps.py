"""Evidence-bound care-gap states and human review workflow."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from importlib import resources
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.clinical.measures import (
    MeasureSubjectResult,
    MeasureTimeWindow,
    PopulationResult,
    PopulationState,
)
from openmed.structured.store import StoreResult, StoreState

CARE_GAP_SCHEMA_VERSION: Final = "1.0.0"
CARE_GAP_COMPATIBILITY_POLICY: Final = "same_major"
CARE_GAP_ADVISORY: Final = (
    "Care-gap output is evidence-bound decision support for human review only; "
    "it does not authorize diagnosis, treatment, outreach, enrollment, or ordering."
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class CareGapError(ValueError):
    """Raised when care-gap inputs violate the public contract."""


class CareGapConflictError(CareGapError):
    """Raised when care-gap custody or immutable history conflicts."""


class CareGapUnsupportedError(CareGapError):
    """Raised when a care-gap version, state, or transition is unsupported."""


class CareGapState(str, Enum):
    """Conservative care-gap evaluation states."""

    MET = "met"
    OPEN = "open"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"


class CareGapReviewStatus(str, Enum):
    """Human-review status independent of the evaluated gap state."""

    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


_REVIEW_TRANSITIONS = {
    "begin_review": frozenset(
        {(CareGapReviewStatus.REQUIRED, CareGapReviewStatus.IN_REVIEW)}
    ),
    "complete_review": frozenset(
        {
            (CareGapReviewStatus.IN_REVIEW, CareGapReviewStatus.APPROVED),
            (CareGapReviewStatus.IN_REVIEW, CareGapReviewStatus.REJECTED),
        }
    ),
}


@dataclass(frozen=True, slots=True)
class CareGapPolicy:
    """Versioned mapping from measure populations to conservative gap states."""

    policy_id: str
    version: str
    initial_population_id: str
    denominator_population_id: str
    numerator_population_id: str
    exclusion_population_id: str | None = None
    exception_population_id: str | None = None
    exception_is_not_applicable: bool = True

    def __post_init__(self) -> None:
        _controlled(self.policy_id, "policy_id")
        _semantic_version(self.version, "policy version")
        identifiers = [
            _controlled(self.initial_population_id, "initial_population_id"),
            _controlled(self.denominator_population_id, "denominator_population_id"),
            _controlled(self.numerator_population_id, "numerator_population_id"),
        ]
        for value, name in (
            (self.exclusion_population_id, "exclusion_population_id"),
            (self.exception_population_id, "exception_population_id"),
        ):
            if value is not None:
                identifiers.append(_controlled(value, name))
        if len(identifiers) != len(set(identifiers)):
            raise CareGapConflictError("care-gap population identifiers must differ")
        if type(self.exception_is_not_applicable) is not bool:
            raise CareGapError("exception_is_not_applicable must be boolean")

    @property
    def digest(self) -> str:
        """Return the exact policy digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete care-gap policy."""

        return {
            "denominator_population_id": self.denominator_population_id,
            "exception_is_not_applicable": self.exception_is_not_applicable,
            "exception_population_id": self.exception_population_id,
            "exclusion_population_id": self.exclusion_population_id,
            "initial_population_id": self.initial_population_id,
            "numerator_population_id": self.numerator_population_id,
            "policy_id": self.policy_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CareGapPolicy":
        """Parse one strict care-gap policy."""

        data = _mapping(value, "care-gap policy")
        _exact_keys(
            data,
            {
                "denominator_population_id",
                "exception_is_not_applicable",
                "exception_population_id",
                "exclusion_population_id",
                "initial_population_id",
                "numerator_population_id",
                "policy_id",
                "version",
            },
            "care-gap policy",
        )
        if type(data["exception_is_not_applicable"]) is not bool:
            raise CareGapError("exception_is_not_applicable must be boolean")
        return cls(
            policy_id=_text(data["policy_id"], "policy_id"),
            version=_text(data["version"], "version"),
            initial_population_id=_text(
                data["initial_population_id"], "initial_population_id"
            ),
            denominator_population_id=_text(
                data["denominator_population_id"], "denominator_population_id"
            ),
            numerator_population_id=_text(
                data["numerator_population_id"], "numerator_population_id"
            ),
            exclusion_population_id=_optional_text(
                data["exclusion_population_id"], "exclusion_population_id"
            ),
            exception_population_id=_optional_text(
                data["exception_population_id"], "exception_population_id"
            ),
            exception_is_not_applicable=data["exception_is_not_applicable"],
        )


@dataclass(frozen=True, slots=True)
class CareGapEvidence:
    """Value-free measure, population, fact, and evidence custody."""

    measure_result_id: str
    measure_result_digest: str
    population_result_digests: Mapping[str, str]
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _opaque_id(self.measure_result_id, "measure_result_id")
        _digest(self.measure_result_digest, "measure_result_digest")
        populations = _digest_mapping(
            self.population_result_digests, "population_result_digests"
        )
        fact_ids = _opaque_values(self.fact_ids, "fact_ids")
        evidence_ids = _opaque_values(self.evidence_ids, "evidence_ids")
        conflict_ids = _opaque_values(self.conflict_ids, "conflict_ids")
        object.__setattr__(
            self, "population_result_digests", _mapping_proxy(populations)
        )
        object.__setattr__(self, "fact_ids", fact_ids)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "conflict_ids", conflict_ids)

    @property
    def digest(self) -> str:
        """Return the value-free evidence digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return value-free care-gap evidence."""

        return {
            "conflict_ids": list(self.conflict_ids),
            "evidence_ids": list(self.evidence_ids),
            "fact_ids": list(self.fact_ids),
            "measure_result_digest": self.measure_result_digest,
            "measure_result_id": self.measure_result_id,
            "population_result_digests": dict(self.population_result_digests),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CareGapEvidence":
        """Parse strict care-gap evidence."""

        data = _mapping(value, "care-gap evidence")
        _exact_keys(
            data,
            {
                "conflict_ids",
                "evidence_ids",
                "fact_ids",
                "measure_result_digest",
                "measure_result_id",
                "population_result_digests",
            },
            "care-gap evidence",
        )
        return cls(
            measure_result_id=_text(data["measure_result_id"], "measure_result_id"),
            measure_result_digest=_text(
                data["measure_result_digest"], "measure_result_digest"
            ),
            population_result_digests={
                _text(key, "population_id"): _text(item, "population result digest")
                for key, item in _mapping(
                    data["population_result_digests"], "population_result_digests"
                ).items()
            },
            fact_ids=_text_sequence(data["fact_ids"], "fact_ids"),
            evidence_ids=_text_sequence(data["evidence_ids"], "evidence_ids"),
            conflict_ids=_text_sequence(data["conflict_ids"], "conflict_ids"),
        )


@dataclass(frozen=True, slots=True)
class CareGapReviewEvent:
    """One immutable, identity-free human-review transition."""

    event_id: str
    action: str
    from_status: CareGapReviewStatus
    to_status: CareGapReviewStatus
    occurred_at: str
    reason_code: str
    authorization_digest: str
    decision_digest: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.event_id, "event_id")
        _controlled(self.action, "action")
        object.__setattr__(
            self,
            "from_status",
            _enum(self.from_status, CareGapReviewStatus, "from_status"),
        )
        object.__setattr__(
            self,
            "to_status",
            _enum(self.to_status, CareGapReviewStatus, "to_status"),
        )
        _timestamp(self.occurred_at, "occurred_at")
        _controlled(self.reason_code, "reason_code")
        _digest(self.authorization_digest, "authorization_digest")
        if self.action == "begin_review" and self.decision_digest is not None:
            raise CareGapConflictError("review start cannot carry a decision")
        if self.action == "complete_review" and self.decision_digest is None:
            raise CareGapConflictError("review completion requires a decision")
        if self.decision_digest is not None:
            _digest(self.decision_digest, "decision_digest")

    def to_dict(self) -> dict[str, Any]:
        """Return value-free review-event custody."""

        return {
            "action": self.action,
            "authorization_digest": self.authorization_digest,
            "decision_digest": self.decision_digest,
            "event_id": self.event_id,
            "from_status": self.from_status.value,
            "occurred_at": self.occurred_at,
            "reason_code": self.reason_code,
            "to_status": self.to_status.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CareGapReviewEvent":
        """Parse one strict review event."""

        data = _mapping(value, "care-gap review event")
        _exact_keys(
            data,
            {
                "action",
                "authorization_digest",
                "decision_digest",
                "event_id",
                "from_status",
                "occurred_at",
                "reason_code",
                "to_status",
            },
            "care-gap review event",
        )
        return cls(
            event_id=_text(data["event_id"], "event_id"),
            action=_text(data["action"], "action"),
            from_status=_enum(data["from_status"], CareGapReviewStatus, "from_status"),
            to_status=_enum(data["to_status"], CareGapReviewStatus, "to_status"),
            occurred_at=_text(data["occurred_at"], "occurred_at"),
            reason_code=_text(data["reason_code"], "reason_code"),
            authorization_digest=_text(
                data["authorization_digest"], "authorization_digest"
            ),
            decision_digest=_optional_text(data["decision_digest"], "decision_digest"),
        )


@dataclass(frozen=True, slots=True)
class CareGapEvaluation:
    """Immutable care-gap version bound to an exact measure result."""

    gap_id: str
    subject_id: str
    measure_definition_version_id: str
    measure_definition_digest: str
    source_snapshot_id: str
    source_snapshot_digest: str
    measurement_period: MeasureTimeWindow
    policy: CareGapPolicy
    state: CareGapState
    reason_code: str
    evidence: CareGapEvidence
    evaluated_at: str
    origin_review_status: CareGapReviewStatus
    review_status: CareGapReviewStatus
    review_events: tuple[CareGapReviewEvent, ...] = ()
    parent_version_id: str | None = None
    schema_version: str = CARE_GAP_SCHEMA_VERSION
    compatibility_policy: str = CARE_GAP_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        for name in (
            "gap_id",
            "subject_id",
            "measure_definition_version_id",
            "source_snapshot_id",
        ):
            _opaque_id(getattr(self, name), name)
        for name in ("measure_definition_digest", "source_snapshot_digest"):
            _digest(getattr(self, name), name)
        if not isinstance(self.measurement_period, MeasureTimeWindow):
            raise TypeError("measurement_period must be MeasureTimeWindow")
        if not isinstance(self.policy, CareGapPolicy):
            raise TypeError("policy must be CareGapPolicy")
        object.__setattr__(self, "state", _enum(self.state, CareGapState, "state"))
        _controlled(self.reason_code, "reason_code")
        if not isinstance(self.evidence, CareGapEvidence):
            raise TypeError("evidence must be CareGapEvidence")
        _timestamp(self.evaluated_at, "evaluated_at")
        origin = _enum(
            self.origin_review_status, CareGapReviewStatus, "origin_review_status"
        )
        current = _enum(self.review_status, CareGapReviewStatus, "review_status")
        if origin not in {
            CareGapReviewStatus.NOT_REQUIRED,
            CareGapReviewStatus.REQUIRED,
        }:
            raise CareGapConflictError("care-gap origin review status is invalid")
        events = tuple(self.review_events)
        previous = origin
        previous_time = self.evaluated_at
        seen: set[str] = set()
        for event in events:
            if not isinstance(event, CareGapReviewEvent):
                raise TypeError("review_events must contain CareGapReviewEvent")
            if event.event_id in seen or event.from_status is not previous:
                raise CareGapConflictError("care-gap review history is discontinuous")
            if (event.from_status, event.to_status) not in _REVIEW_TRANSITIONS.get(
                event.action, frozenset()
            ):
                raise CareGapConflictError("care-gap review transition is invalid")
            if _time_key(event.occurred_at) < _time_key(previous_time):
                raise CareGapConflictError(
                    "care-gap review events must be chronological"
                )
            seen.add(event.event_id)
            previous = event.to_status
            previous_time = event.occurred_at
        if current is not previous:
            raise CareGapConflictError("review status differs from event history")
        if (
            self.state is CareGapState.INSUFFICIENT_DATA
            and origin is not CareGapReviewStatus.REQUIRED
        ):
            raise CareGapConflictError("insufficient-data gaps require review")
        if self.evidence.conflict_ids and origin is not CareGapReviewStatus.REQUIRED:
            raise CareGapConflictError("conflicting inputs require review")
        if self.parent_version_id is not None:
            _opaque_id(self.parent_version_id, "parent_version_id")
        object.__setattr__(self, "origin_review_status", origin)
        object.__setattr__(self, "review_status", current)
        object.__setattr__(self, "review_events", events)

    @property
    def version_id(self) -> str:
        """Return the immutable care-gap version identifier."""

        return _derived_id("caregapversion", self.identity_payload)

    @property
    def version_digest(self) -> str:
        """Return the digest of the complete care-gap version."""

        return canonical_digest(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all fields covered by the care-gap version digest."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "evaluated_at": self.evaluated_at,
            "evidence": self.evidence.to_dict(),
            "gap_id": self.gap_id,
            "measure_definition_digest": self.measure_definition_digest,
            "measure_definition_version_id": self.measure_definition_version_id,
            "measurement_period": self.measurement_period.to_dict(),
            "origin_review_status": self.origin_review_status.value,
            "parent_version_id": self.parent_version_id,
            "policy": self.policy.to_dict(),
            "reason_code": self.reason_code,
            "review_events": [item.to_dict() for item in self.review_events],
            "review_status": self.review_status.value,
            "schema_version": self.schema_version,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "state": self.state.value,
            "subject_id": self.subject_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete value-free care-gap artifact."""

        return {
            "advisory": CARE_GAP_ADVISORY,
            "artifact_type": "care_gap_evaluation",
            "version_digest": self.version_digest,
            "version_id": self.version_id,
            **self.identity_payload,
        }

    def to_json(self) -> str:
        """Return canonical care-gap JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CareGapEvaluation":
        """Parse and verify a persisted care-gap version."""

        data = _mapping(value, "care-gap evaluation")
        expected = {
            "advisory",
            "artifact_type",
            "compatibility_policy",
            "evaluated_at",
            "evidence",
            "gap_id",
            "measure_definition_digest",
            "measure_definition_version_id",
            "measurement_period",
            "origin_review_status",
            "parent_version_id",
            "policy",
            "reason_code",
            "review_events",
            "review_status",
            "schema_version",
            "source_snapshot_digest",
            "source_snapshot_id",
            "state",
            "subject_id",
            "version_digest",
            "version_id",
        }
        _exact_keys(data, expected, "care-gap evaluation")
        if data["artifact_type"] != "care_gap_evaluation":
            raise CareGapUnsupportedError("care-gap artifact type is unsupported")
        if data["advisory"] != CARE_GAP_ADVISORY:
            raise CareGapConflictError("care-gap advisory differs")
        result = cls(
            gap_id=_text(data["gap_id"], "gap_id"),
            subject_id=_text(data["subject_id"], "subject_id"),
            measure_definition_version_id=_text(
                data["measure_definition_version_id"],
                "measure_definition_version_id",
            ),
            measure_definition_digest=_text(
                data["measure_definition_digest"], "measure_definition_digest"
            ),
            source_snapshot_id=_text(data["source_snapshot_id"], "source_snapshot_id"),
            source_snapshot_digest=_text(
                data["source_snapshot_digest"], "source_snapshot_digest"
            ),
            measurement_period=MeasureTimeWindow.from_dict(
                _mapping(data["measurement_period"], "measurement_period")
            ),
            policy=CareGapPolicy.from_dict(_mapping(data["policy"], "policy")),
            state=_enum(data["state"], CareGapState, "state"),
            reason_code=_text(data["reason_code"], "reason_code"),
            evidence=CareGapEvidence.from_dict(_mapping(data["evidence"], "evidence")),
            evaluated_at=_text(data["evaluated_at"], "evaluated_at"),
            origin_review_status=_enum(
                data["origin_review_status"],
                CareGapReviewStatus,
                "origin_review_status",
            ),
            review_status=_enum(
                data["review_status"], CareGapReviewStatus, "review_status"
            ),
            review_events=tuple(
                CareGapReviewEvent.from_dict(_mapping(item, "review event"))
                for item in _sequence(data["review_events"], "review_events")
            ),
            parent_version_id=_optional_text(
                data["parent_version_id"], "parent_version_id"
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["version_id"] != result.version_id:
            raise CareGapConflictError("care-gap version identifier differs")
        if data["version_digest"] != result.version_digest:
            raise CareGapConflictError("care-gap version digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "CareGapEvaluation":
        """Parse canonical or human-formatted care-gap JSON."""

        return cls.from_dict(_json_object(value, "care-gap evaluation"))


@dataclass(frozen=True, slots=True)
class CareGapHistory:
    """Ordered immutable versions for one stable care gap."""

    gap_id: str
    versions: tuple[CareGapEvaluation, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.gap_id, "gap_id")
        versions = tuple(self.versions)
        if not versions:
            raise CareGapError("care-gap history requires at least one version")
        seen: set[str] = set()
        previous: CareGapEvaluation | None = None
        for version in versions:
            if not isinstance(version, CareGapEvaluation):
                raise TypeError("versions must contain CareGapEvaluation")
            if version.gap_id != self.gap_id or version.version_id in seen:
                raise CareGapConflictError("care-gap history identity conflicts")
            expected_parent = None if previous is None else previous.version_id
            if version.parent_version_id != expected_parent:
                raise CareGapConflictError("care-gap version ancestry is discontinuous")
            if previous is not None and _time_key(version.evaluated_at) < _time_key(
                previous.evaluated_at
            ):
                raise CareGapConflictError("care-gap versions must be chronological")
            seen.add(version.version_id)
            previous = version
        object.__setattr__(self, "versions", versions)

    @property
    def current(self) -> CareGapEvaluation:
        """Return the newest immutable version."""

        return self.versions[-1]

    def append(self, version: CareGapEvaluation) -> "CareGapHistory":
        """Append one exact child version after validating ancestry."""

        return CareGapHistory(gap_id=self.gap_id, versions=(*self.versions, version))

    def to_dict(self) -> dict[str, Any]:
        """Return all value-free versions in history order."""

        return {
            "artifact_type": "care_gap_history",
            "gap_id": self.gap_id,
            "versions": [item.to_dict() for item in self.versions],
        }


def evaluate_care_gap(
    measure_result: MeasureSubjectResult,
    policy: CareGapPolicy,
    *,
    conflict_ids: tuple[str, ...] = (),
    previous: CareGapEvaluation | None = None,
) -> StoreResult[CareGapEvaluation]:
    """Derive one conservative gap state from an exact measure result."""

    try:
        if not isinstance(measure_result, MeasureSubjectResult):
            raise TypeError("measure_result must be MeasureSubjectResult")
        if not isinstance(policy, CareGapPolicy):
            raise TypeError("policy must be CareGapPolicy")
        conflicts = _opaque_values(conflict_ids, "conflict_ids")
        populations = {item.population_id: item for item in measure_result.populations}
        state, reason = _derive_state(populations, policy, conflicts)
        evidence = CareGapEvidence(
            measure_result_id=measure_result.result_id,
            measure_result_digest=measure_result.result_digest,
            population_result_digests={
                item.population_id: item.digest for item in measure_result.populations
            },
            fact_ids=tuple(
                sorted(
                    {
                        fact_id
                        for item in measure_result.populations
                        for fact_id in item.evidence.fact_ids
                    }
                )
            ),
            evidence_ids=tuple(
                sorted(
                    {
                        evidence_id
                        for item in measure_result.populations
                        for evidence_id in item.evidence.evidence_ids
                    }
                )
            ),
            conflict_ids=conflicts,
        )
        review = (
            CareGapReviewStatus.REQUIRED
            if state is CareGapState.INSUFFICIENT_DATA or conflicts
            else CareGapReviewStatus.NOT_REQUIRED
        )
        gap_id = _derived_id(
            "caregap",
            measure_result.subject_id,
            measure_result.definition_version_id,
            measure_result.measurement_period.to_dict(),
            policy.digest,
        )
        if previous is not None:
            if not isinstance(previous, CareGapEvaluation):
                raise TypeError("previous must be CareGapEvaluation")
            if previous.gap_id != gap_id:
                return StoreResult.outcome(
                    StoreState.CONFLICT, "care_gap_parent_identity_conflict"
                )
            if previous.evidence.measure_result_digest == measure_result.result_digest:
                return StoreResult.outcome(
                    StoreState.CONFLICT, "care_gap_correction_unchanged"
                )
            if _time_key(measure_result.evaluated_at) < _time_key(
                previous.evaluated_at
            ):
                return StoreResult.outcome(
                    StoreState.CONFLICT, "care_gap_correction_time_conflict"
                )
        result = CareGapEvaluation(
            gap_id=gap_id,
            subject_id=measure_result.subject_id,
            measure_definition_version_id=measure_result.definition_version_id,
            measure_definition_digest=measure_result.definition_digest,
            source_snapshot_id=measure_result.source_snapshot_id,
            source_snapshot_digest=measure_result.source_snapshot_digest,
            measurement_period=measure_result.measurement_period,
            policy=policy,
            state=state,
            reason_code=reason,
            evidence=evidence,
            evaluated_at=measure_result.evaluated_at,
            origin_review_status=review,
            review_status=review,
            parent_version_id=None if previous is None else previous.version_id,
        )
    except CareGapUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "care_gap_state_unsupported")
    except CareGapConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "care_gap_evidence_conflict")
    return StoreResult.success(result, created=True)


def begin_care_gap_review(
    evaluation: CareGapEvaluation,
    *,
    authorization_digest: str,
    occurred_at: str,
    reason_code: str = "human_review_started",
) -> StoreResult[CareGapEvaluation]:
    """Begin mandatory human review without recording reviewer identity."""

    if not isinstance(evaluation, CareGapEvaluation):
        raise TypeError("evaluation must be CareGapEvaluation")
    if evaluation.review_status is not CareGapReviewStatus.REQUIRED:
        return StoreResult.outcome(StoreState.CONFLICT, "care_gap_review_not_required")
    try:
        event = CareGapReviewEvent(
            event_id=_derived_id(
                "caregapevent", evaluation.version_id, "begin_review", occurred_at
            ),
            action="begin_review",
            from_status=evaluation.review_status,
            to_status=CareGapReviewStatus.IN_REVIEW,
            occurred_at=occurred_at,
            reason_code=reason_code,
            authorization_digest=authorization_digest,
        )
        return StoreResult.success(
            replace(
                evaluation,
                review_status=CareGapReviewStatus.IN_REVIEW,
                review_events=(*evaluation.review_events, event),
                parent_version_id=evaluation.version_id,
            )
        )
    except CareGapError:
        return StoreResult.outcome(StoreState.CONFLICT, "care_gap_review_conflict")


def complete_care_gap_review(
    evaluation: CareGapEvaluation,
    *,
    approved: bool,
    authorization_digest: str,
    decision_digest: str,
    occurred_at: str,
    reason_code: str,
) -> StoreResult[CareGapEvaluation]:
    """Record a governed review decision without changing evidence state."""

    if not isinstance(evaluation, CareGapEvaluation):
        raise TypeError("evaluation must be CareGapEvaluation")
    if type(approved) is not bool:
        raise TypeError("approved must be boolean")
    if evaluation.review_status is not CareGapReviewStatus.IN_REVIEW:
        return StoreResult.outcome(StoreState.CONFLICT, "care_gap_review_not_started")
    try:
        destination = (
            CareGapReviewStatus.APPROVED if approved else CareGapReviewStatus.REJECTED
        )
        event = CareGapReviewEvent(
            event_id=_derived_id(
                "caregapevent",
                evaluation.version_id,
                "complete_review",
                destination.value,
                occurred_at,
            ),
            action="complete_review",
            from_status=evaluation.review_status,
            to_status=destination,
            occurred_at=occurred_at,
            reason_code=reason_code,
            authorization_digest=authorization_digest,
            decision_digest=decision_digest,
        )
        return StoreResult.success(
            replace(
                evaluation,
                review_status=destination,
                review_events=(*evaluation.review_events, event),
                parent_version_id=evaluation.version_id,
            )
        )
    except CareGapError:
        return StoreResult.outcome(StoreState.CONFLICT, "care_gap_review_conflict")


def load_care_gap_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for care-gap evaluations."""

    text = (
        resources.files("openmed.core.schemas.json")
        .joinpath("care_gap.schema.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - packaged invariant
        raise RuntimeError("care-gap schema must be an object")
    return value


def _derive_state(
    populations: Mapping[str, PopulationResult],
    policy: CareGapPolicy,
    conflict_ids: tuple[str, ...],
) -> tuple[CareGapState, str]:
    required_ids = (
        policy.initial_population_id,
        policy.denominator_population_id,
        policy.numerator_population_id,
    )
    optional_ids = tuple(
        item
        for item in (policy.exclusion_population_id, policy.exception_population_id)
        if item is not None
    )
    if any(item not in populations for item in (*required_ids, *optional_ids)):
        return CareGapState.INSUFFICIENT_DATA, "population_result_missing"
    if conflict_ids:
        return CareGapState.INSUFFICIENT_DATA, "source_evidence_conflict"
    selected = tuple(populations[item] for item in (*required_ids, *optional_ids))
    if any(
        item.state in {PopulationState.UNKNOWN, PopulationState.ERROR}
        for item in selected
    ):
        return CareGapState.INSUFFICIENT_DATA, "population_result_unresolved"
    initial = populations[policy.initial_population_id]
    denominator = populations[policy.denominator_population_id]
    numerator = populations[policy.numerator_population_id]
    if initial.state is PopulationState.NOT_MET:
        return CareGapState.NOT_APPLICABLE, "initial_population_not_met"
    if denominator.state is PopulationState.NOT_MET:
        return CareGapState.NOT_APPLICABLE, "denominator_not_met"
    if policy.exclusion_population_id is not None:
        exclusion = populations[policy.exclusion_population_id]
        if exclusion.state is PopulationState.MET:
            return CareGapState.NOT_APPLICABLE, "denominator_exclusion_met"
    if policy.exception_population_id is not None:
        exception = populations[policy.exception_population_id]
        if (
            exception.state is PopulationState.MET
            and policy.exception_is_not_applicable
        ):
            return CareGapState.NOT_APPLICABLE, "denominator_exception_met"
    if numerator.state is PopulationState.MET:
        return CareGapState.MET, "numerator_met"
    if numerator.state is PopulationState.NOT_MET:
        return CareGapState.OPEN, "numerator_not_met"
    return CareGapState.INSUFFICIENT_DATA, "population_result_unresolved"


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != CARE_GAP_SCHEMA_VERSION:
        raise CareGapUnsupportedError("care-gap schema version is unsupported")
    if compatibility_policy != CARE_GAP_COMPATIBILITY_POLICY:
        raise CareGapUnsupportedError("care-gap compatibility policy is unsupported")


def _derived_id(prefix: str, *materials: Any) -> str:
    return f"{prefix}_{canonical_digest(list(materials)).removeprefix('sha256:')[:32]}"


def _mapping_proxy(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CareGapError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise CareGapError(f"{name} must be a sequence")
    return tuple(value)


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise CareGapError(f"{name} fields differ")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CareGapError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise CareGapError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise CareGapError(f"{name} must be opaque")
    return text


def _opaque_values(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    result = tuple(sorted({_opaque_id(item, name) for item in value}))
    if len(result) != len(value):
        raise CareGapConflictError(f"{name} must be unique")
    return result


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise CareGapError(f"{name} must be a digest")
    return text


def _digest_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    return {
        _controlled(key, "population_id"): _digest(item, "population result digest")
        for key, item in sorted(_mapping(value, name).items())
    }


def _semantic_version(value: Any, name: str) -> str:
    text = _text(value, name)
    if _VERSION_RE.fullmatch(text) is None:
        raise CareGapError(f"{name} must be semantic")
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise CareGapError(f"{name} must be an RFC 3339 timestamp")
    return text


def _time_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError):
        raise CareGapUnsupportedError(f"{name} is unsupported") from None


def _json_object(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise CareGapError(f"{name} must be valid JSON") from None
    return _mapping(parsed, name)


__all__ = [
    "CARE_GAP_ADVISORY",
    "CARE_GAP_COMPATIBILITY_POLICY",
    "CARE_GAP_SCHEMA_VERSION",
    "CareGapConflictError",
    "CareGapError",
    "CareGapEvaluation",
    "CareGapEvidence",
    "CareGapHistory",
    "CareGapPolicy",
    "CareGapReviewEvent",
    "CareGapReviewStatus",
    "CareGapState",
    "CareGapUnsupportedError",
    "begin_care_gap_review",
    "complete_care_gap_review",
    "evaluate_care_gap",
    "load_care_gap_schema",
]
