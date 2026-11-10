"""Deterministic, value-free explanations of cohort membership.

The evaluator consumes developer-authored criterion declarations plus digests
that bind locally retained evidence. It emits criterion states and
time-window references without retaining clinical values. An explanation may
request human review, but it never authorizes enrollment or patient contact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

COHORT_EXPLANATION_SCHEMA: Final = "openmed.agent.workflows.cohort_explanation.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63}){0,7}")
_MAX_ITEMS: Final = 10_000
_MAX_SCHEMA_VERSION: Final = (1 << 31) - 1


class CriterionKind(str, Enum):
    """How a criterion contributes to cohort eligibility."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class EvidenceAssertion(str, Enum):
    """Closed assertion vocabulary for one evidence item."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"


class CriterionState(str, Enum):
    """Deterministic result of evaluating one declared criterion."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class MembershipState(str, Enum):
    """Cohort result before any separately governed operational action."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    REVIEW_REQUIRED = "review_required"


class CohortExplanationError(ValueError):
    """A PHI-safe cohort-explanation validation error.

    Args:
        code: Stable machine-readable error code.
        field_name: Optional public contract field associated with the error.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, order=True)
class TimeWindowReference:
    """Opaque reference to a locally retained temporal window.

    ``reference_id`` is a developer-authored label, not a patient or encounter
    identifier. ``window_digest`` binds the exact window boundaries retained
    inside the caller's trusted clinical-data boundary.
    """

    reference_id: str
    window_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.reference_id, "reference_id")
        _validate_digest(self.window_digest, "window_digest")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic, value-free time-window metadata."""

        return {
            "reference_id": self.reference_id,
            "window_digest": self.window_digest,
        }


@dataclass(frozen=True, slots=True, order=True)
class CohortCriterion:
    """One declarative inclusion or exclusion criterion."""

    criterion_id: str
    kind: CriterionKind
    time_window: TimeWindowReference | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.criterion_id, "criterion_id")
        if type(self.kind) is not CriterionKind:
            raise CohortExplanationError("invalid_criterion_kind", "kind")
        if (
            self.time_window is not None
            and type(self.time_window) is not TimeWindowReference
        ):
            raise CohortExplanationError("invalid_time_window", "time_window")

    def to_dict(self) -> dict[str, Any]:
        """Return the criterion declaration without clinical values."""

        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind.value,
            "time_window": (
                None if self.time_window is None else self.time_window.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True, init=False)
class CohortDefinition:
    """A named, versioned, deterministically ordered criterion set."""

    definition_id: str
    version: int
    criteria: tuple[CohortCriterion, ...]

    def __init__(
        self,
        definition_id: str,
        version: int,
        criteria: Iterable[CohortCriterion],
    ) -> None:
        object.__setattr__(
            self,
            "definition_id",
            _validate_identifier(definition_id, "definition_id"),
        )
        object.__setattr__(self, "version", _validate_version(version))
        values = _bounded_tuple(criteria, field_name="criteria", maximum=_MAX_ITEMS)
        if not values:
            raise CohortExplanationError("empty_collection", "criteria")
        if any(type(value) is not CohortCriterion for value in values):
            raise CohortExplanationError("invalid_criterion", "criteria")
        normalized = tuple(sorted(values, key=lambda value: value.criterion_id))
        identifiers = tuple(value.criterion_id for value in normalized)
        if len(set(identifiers)) != len(identifiers):
            raise CohortExplanationError("duplicate_criterion", "criteria")
        object.__setattr__(self, "criteria", normalized)

    @property
    def definition_digest(self) -> str:
        """Return a stable digest of the exact declaration and version."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic cohort-definition metadata."""

        return {
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "definition_id": self.definition_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class CriterionEvidence:
    """A digest-addressed assertion for one criterion and temporal window."""

    criterion_id: str
    assertion: EvidenceAssertion
    evidence_digest: str
    time_window: TimeWindowReference | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.criterion_id, "criterion_id")
        if type(self.assertion) is not EvidenceAssertion:
            raise CohortExplanationError("invalid_evidence_assertion", "assertion")
        _validate_digest(self.evidence_digest, "evidence_digest")
        if (
            self.time_window is not None
            and type(self.time_window) is not TimeWindowReference
        ):
            raise CohortExplanationError("invalid_time_window", "time_window")

    def to_dict(self) -> dict[str, Any]:
        """Return evidence metadata without source or clinical content."""

        return {
            "assertion": self.assertion.value,
            "criterion_id": self.criterion_id,
            "evidence_digest": self.evidence_digest,
            "time_window": (
                None if self.time_window is None else self.time_window.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True, init=False)
class CohortRecordEvidence:
    """Digest-addressed evidence for one record and cohort-definition version."""

    record_digest: str
    definition_id: str
    definition_version: int
    evidence: tuple[CriterionEvidence, ...] = field(repr=False)

    def __init__(
        self,
        record_digest: str,
        definition_id: str,
        definition_version: int,
        evidence: Iterable[CriterionEvidence],
    ) -> None:
        object.__setattr__(
            self,
            "record_digest",
            _validate_digest(record_digest, "record_digest"),
        )
        object.__setattr__(
            self,
            "definition_id",
            _validate_identifier(definition_id, "definition_id"),
        )
        object.__setattr__(
            self,
            "definition_version",
            _validate_version(definition_version, "definition_version"),
        )
        values = _bounded_tuple(evidence, field_name="evidence", maximum=_MAX_ITEMS)
        if any(type(value) is not CriterionEvidence for value in values):
            raise CohortExplanationError("invalid_evidence", "evidence")
        normalized = tuple(sorted(values, key=_evidence_sort_key))
        evidence_keys = tuple(
            (value.criterion_id, value.evidence_digest) for value in normalized
        )
        if len(set(evidence_keys)) != len(evidence_keys):
            raise CohortExplanationError("duplicate_evidence", "evidence")
        object.__setattr__(self, "evidence", normalized)

    @property
    def evidence_metadata_digest(self) -> str:
        """Return a digest over the ordered, content-free evidence metadata."""

        return _digest([value.to_dict() for value in self.evidence])


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """Criterion-level state, evidence references, and temporal context."""

    criterion_id: str
    kind: CriterionKind
    state: CriterionState
    evidence_digests: tuple[str, ...]
    time_window: TimeWindowReference | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.criterion_id, "criterion_id")
        if type(self.kind) is not CriterionKind:
            raise CohortExplanationError("invalid_criterion_kind", "kind")
        if type(self.state) is not CriterionState:
            raise CohortExplanationError("invalid_criterion_state", "state")
        if type(self.evidence_digests) is not tuple:
            raise CohortExplanationError("invalid_evidence_digests", "evidence_digests")
        if len(self.evidence_digests) > _MAX_ITEMS:
            raise CohortExplanationError("too_many_items", "evidence_digests")
        for value in self.evidence_digests:
            _validate_digest(value, "evidence_digests")
        if self.evidence_digests != tuple(sorted(self.evidence_digests)):
            raise CohortExplanationError(
                "evidence_digests_not_sorted", "evidence_digests"
            )
        if len(set(self.evidence_digests)) != len(self.evidence_digests):
            raise CohortExplanationError("duplicate_evidence", "evidence_digests")
        if (
            self.time_window is not None
            and type(self.time_window) is not TimeWindowReference
        ):
            raise CohortExplanationError("invalid_time_window", "time_window")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete criterion-level explanation."""

        return {
            "criterion_id": self.criterion_id,
            "evidence_count": len(self.evidence_digests),
            "evidence_digests": list(self.evidence_digests),
            "kind": self.kind.value,
            "state": self.state.value,
            "time_window": (
                None if self.time_window is None else self.time_window.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class CohortMembershipExplanation:
    """Deterministic explanation that grants no operational authority."""

    record_digest: str
    definition_digest: str
    evidence_metadata_digest: str
    explanation_digest: str
    membership_state: MembershipState
    criteria: tuple[CriterionEvaluation, ...]
    schema: str = COHORT_EXPLANATION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "record_digest",
            "definition_digest",
            "evidence_metadata_digest",
            "explanation_digest",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if type(self.membership_state) is not MembershipState:
            raise CohortExplanationError("invalid_membership_state", "membership_state")
        if (
            type(self.criteria) is not tuple
            or not self.criteria
            or any(type(value) is not CriterionEvaluation for value in self.criteria)
        ):
            raise CohortExplanationError("invalid_criteria", "criteria")
        identifiers = tuple(value.criterion_id for value in self.criteria)
        if identifiers != tuple(sorted(identifiers)):
            raise CohortExplanationError("criteria_not_sorted", "criteria")
        if len(set(identifiers)) != len(identifiers):
            raise CohortExplanationError("duplicate_criterion", "criteria")
        expected_state = _membership_state(self.criteria)
        if self.membership_state is not expected_state:
            raise CohortExplanationError(
                "inconsistent_membership_state", "membership_state"
            )
        if self.schema != COHORT_EXPLANATION_SCHEMA:
            raise CohortExplanationError("invalid_schema", "schema")

    @property
    def requires_human_review(self) -> bool:
        """Return whether any criterion is unknown or conflicting."""

        return self.membership_state is MembershipState.REVIEW_REQUIRED

    @property
    def authorizes_enrollment(self) -> bool:
        """Return ``False`` because an explanation cannot enroll a record."""

        return False

    @property
    def authorizes_contact(self) -> bool:
        """Return ``False`` because an explanation cannot authorize contact."""

        return False

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic explanation metadata and explicit safety gates."""

        return {
            "authorizes_contact": self.authorizes_contact,
            "authorizes_enrollment": self.authorizes_enrollment,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "definition_digest": self.definition_digest,
            "evidence_metadata_digest": self.evidence_metadata_digest,
            "explanation_digest": self.explanation_digest,
            "membership_state": self.membership_state.value,
            "record_digest": self.record_digest,
            "requires_human_review": self.requires_human_review,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the explanation with stable key ordering."""

        return _canonical_json(self.to_dict())


def explain_cohort_membership(
    record: CohortRecordEvidence,
    definition: CohortDefinition,
) -> CohortMembershipExplanation:
    """Evaluate one record against an exact declarative cohort definition.

    Missing or solely inconclusive evidence produces ``unknown``. Opposing
    decisive assertions produce ``conflict``. Either state forces human review
    for the whole record. This local function performs no enrollment, contact,
    notification, approval, filesystem, database, or network operation.

    Args:
        record: Digest-addressed evidence for one record.
        definition: Exact versioned cohort definition to apply.

    Returns:
        A deterministic criterion-level explanation.

    Raises:
        CohortExplanationError: If types, versions, criteria, or windows differ.
    """

    if type(record) is not CohortRecordEvidence:
        raise CohortExplanationError("invalid_record", "record")
    if type(definition) is not CohortDefinition:
        raise CohortExplanationError("invalid_definition", "definition")
    if record.definition_id != definition.definition_id:
        raise CohortExplanationError("definition_id_mismatch", "definition_id")
    if record.definition_version != definition.version:
        raise CohortExplanationError(
            "definition_version_mismatch", "definition_version"
        )

    criterion_by_id = {
        criterion.criterion_id: criterion for criterion in definition.criteria
    }
    evidence_by_criterion: dict[str, list[CriterionEvidence]] = {}
    for evidence in record.evidence:
        criterion = criterion_by_id.get(evidence.criterion_id)
        if criterion is None:
            raise CohortExplanationError("unknown_criterion", "criterion_id")
        if evidence.time_window != criterion.time_window:
            raise CohortExplanationError("time_window_mismatch", "time_window")
        evidence_by_criterion.setdefault(evidence.criterion_id, []).append(evidence)

    evaluations: list[CriterionEvaluation] = []
    for criterion in definition.criteria:
        evidence_items = evidence_by_criterion.get(criterion.criterion_id, [])
        assertions = {value.assertion for value in evidence_items}
        evaluations.append(
            CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                kind=criterion.kind,
                state=_criterion_state(assertions),
                evidence_digests=tuple(
                    sorted(value.evidence_digest for value in evidence_items)
                ),
                time_window=criterion.time_window,
            )
        )

    normalized_evaluations = tuple(evaluations)
    membership_state = _membership_state(normalized_evaluations)
    explanation_fields = {
        "authorizes_contact": False,
        "authorizes_enrollment": False,
        "criteria": [value.to_dict() for value in normalized_evaluations],
        "definition_digest": definition.definition_digest,
        "evidence_metadata_digest": record.evidence_metadata_digest,
        "membership_state": membership_state.value,
        "record_digest": record.record_digest,
        "requires_human_review": membership_state is MembershipState.REVIEW_REQUIRED,
        "schema": COHORT_EXPLANATION_SCHEMA,
    }
    return CohortMembershipExplanation(
        record_digest=record.record_digest,
        definition_digest=definition.definition_digest,
        evidence_metadata_digest=record.evidence_metadata_digest,
        explanation_digest=_digest(explanation_fields),
        membership_state=membership_state,
        criteria=normalized_evaluations,
    )


def _criterion_state(assertions: set[EvidenceAssertion]) -> CriterionState:
    has_met = EvidenceAssertion.MET in assertions
    has_not_met = EvidenceAssertion.NOT_MET in assertions
    if has_met and has_not_met:
        return CriterionState.CONFLICT
    if has_met:
        return CriterionState.MET
    if has_not_met:
        return CriterionState.NOT_MET
    return CriterionState.UNKNOWN


def _membership_state(
    criteria: tuple[CriterionEvaluation, ...],
) -> MembershipState:
    if any(
        value.state in (CriterionState.UNKNOWN, CriterionState.CONFLICT)
        for value in criteria
    ):
        return MembershipState.REVIEW_REQUIRED
    if any(
        (
            value.kind is CriterionKind.INCLUSION
            and value.state is CriterionState.NOT_MET
        )
        or (value.kind is CriterionKind.EXCLUSION and value.state is CriterionState.MET)
        for value in criteria
    ):
        return MembershipState.INELIGIBLE
    return MembershipState.ELIGIBLE


def _evidence_sort_key(value: CriterionEvidence) -> tuple[str, str, str, str, str]:
    window_id = "" if value.time_window is None else value.time_window.reference_id
    window_digest = "" if value.time_window is None else value.time_window.window_digest
    return (
        value.criterion_id,
        value.assertion.value,
        value.evidence_digest,
        window_id,
        window_digest,
    )


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CohortExplanationError("invalid_identifier", field_name)
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise CohortExplanationError("invalid_digest", field_name)
    return value


def _validate_version(value: Any, field_name: str = "version") -> int:
    if type(value) is not int or not 1 <= value <= _MAX_SCHEMA_VERSION:
        raise CohortExplanationError("invalid_version", field_name)
    return value


def _bounded_tuple(value: Any, *, field_name: str, maximum: int) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise CohortExplanationError("invalid_collection", field_name)
    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise CohortExplanationError("invalid_collection", field_name) from None

    items: list[Any] = []
    try:
        for item in iterator:
            if len(items) >= maximum:
                raise CohortExplanationError("too_many_items", field_name)
            items.append(item)
    except (KeyboardInterrupt, SystemExit, CohortExplanationError):
        raise
    except Exception:
        raise CohortExplanationError("invalid_collection", field_name) from None
    return tuple(items)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "COHORT_EXPLANATION_SCHEMA",
    "CohortCriterion",
    "CohortDefinition",
    "CohortExplanationError",
    "CohortMembershipExplanation",
    "CohortRecordEvidence",
    "CriterionEvaluation",
    "CriterionEvidence",
    "CriterionKind",
    "CriterionState",
    "EvidenceAssertion",
    "MembershipState",
    "TimeWindowReference",
    "explain_cohort_membership",
]
