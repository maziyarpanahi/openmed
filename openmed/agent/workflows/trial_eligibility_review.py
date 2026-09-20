"""Structured review packets for trial-eligibility disagreements.

The comparison is deterministic and local. It combines a value-free rule
explanation with evidence-backed model assessments, classifies disagreement
causes per criterion, and emits only identifiers, digests, offsets, closed
states, and uncertainty scores. A packet requests review but never authorizes
enrollment or clinical outreach.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .cohort_explanations import (
    CohortMembershipExplanation,
    CriterionEvaluation,
    CriterionKind,
    CriterionState,
    TimeWindowReference,
)

TRIAL_ELIGIBILITY_REVIEW_SCHEMA: Final = (
    "openmed.agent.workflows.trial_eligibility_review.v1"
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63}){0,7}")
_MAX_ITEMS: Final = 10_000
_MAX_OFFSET: Final = (1 << 63) - 1


class DisagreementCause(str, Enum):
    """Closed reasons that route a criterion to human review."""

    RULE_EVIDENCE_MISSING = "rule_evidence_missing"
    RULE_EVIDENCE_CONFLICT = "rule_evidence_conflict"
    MODEL_ASSESSMENT_MISSING = "model_assessment_missing"
    MODEL_ASSESSMENT_UNKNOWN = "model_assessment_unknown"
    MODEL_ASSESSMENT_CONFLICT = "model_assessment_conflict"
    OUTCOME_CONFLICT = "outcome_conflict"


class TrialEligibilityReviewError(ValueError):
    """A PHI-safe trial-eligibility comparison error.

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
class EligibilityCitation:
    """Digest-addressed source span retained without source text.

    ``source_digest`` identifies the local source artifact and
    ``evidence_digest`` binds the exact normalized evidence used by the model.
    Offsets locate the cited span without copying its protected contents.
    """

    source_digest: str = field(repr=False)
    start_offset: int
    end_offset: int
    evidence_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_digest(self.source_digest, "source_digest")
        _validate_offset(self.start_offset, "start_offset")
        _validate_offset(self.end_offset, "end_offset")
        if self.end_offset <= self.start_offset:
            raise TrialEligibilityReviewError("invalid_span", "end_offset")
        _validate_digest(self.evidence_digest, "evidence_digest")

    @property
    def citation_digest(self) -> str:
        """Return a stable digest of the complete citation metadata."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, str | int]:
        """Return the value-free source reference."""

        return {
            "end_offset": self.end_offset,
            "evidence_digest": self.evidence_digest,
            "source_digest": self.source_digest,
            "start_offset": self.start_offset,
        }


@dataclass(frozen=True, slots=True, init=False)
class ModelCriterionAssessment:
    """One evidence-backed model assessment for a declared criterion."""

    criterion_id: str
    state: CriterionState
    citations: tuple[EligibilityCitation, ...] = field(repr=False)
    uncertainty: float
    model_digest: str = field(repr=False)

    def __init__(
        self,
        criterion_id: str,
        state: CriterionState,
        citations: Iterable[EligibilityCitation],
        uncertainty: float,
        model_digest: str,
    ) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _validate_identifier(criterion_id, "criterion_id"),
        )
        if type(state) is not CriterionState:
            raise TrialEligibilityReviewError("invalid_assessment_state", "state")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "citations", _normalize_citations(citations))
        object.__setattr__(
            self,
            "uncertainty",
            _validate_uncertainty(uncertainty),
        )
        object.__setattr__(
            self,
            "model_digest",
            _validate_digest(model_digest, "model_digest"),
        )

    @property
    def assessment_digest(self) -> str:
        """Return a digest binding the state, citations, model, and uncertainty."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic model-assessment metadata."""

        return {
            "citations": [citation.to_dict() for citation in self.citations],
            "criterion_id": self.criterion_id,
            "model_digest": self.model_digest,
            "state": self.state.value,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class CriterionDisagreement:
    """Rule/model comparison and review reasons for one criterion."""

    criterion_id: str
    kind: CriterionKind
    rule_state: CriterionState
    rule_evidence_digests: tuple[str, ...] = field(repr=False)
    time_window: TimeWindowReference | None
    model_state: CriterionState | None
    model_assessment_digest: str | None = field(repr=False)
    model_citations: tuple[EligibilityCitation, ...] = field(repr=False)
    model_uncertainty: float | None
    causes: tuple[DisagreementCause, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.criterion_id, "criterion_id")
        if type(self.kind) is not CriterionKind:
            raise TrialEligibilityReviewError("invalid_criterion_kind", "kind")
        if type(self.rule_state) is not CriterionState:
            raise TrialEligibilityReviewError("invalid_criterion_state", "rule_state")
        _validate_digests(self.rule_evidence_digests, "rule_evidence_digests")
        if (
            self.time_window is not None
            and type(self.time_window) is not TimeWindowReference
        ):
            raise TrialEligibilityReviewError("invalid_time_window", "time_window")
        _validate_model_fields(self)
        if type(self.causes) is not tuple or any(
            type(cause) is not DisagreementCause for cause in self.causes
        ):
            raise TrialEligibilityReviewError("invalid_causes", "causes")
        if self.causes != tuple(sorted(self.causes, key=lambda cause: cause.value)):
            raise TrialEligibilityReviewError("causes_not_sorted", "causes")
        if len(set(self.causes)) != len(self.causes):
            raise TrialEligibilityReviewError("duplicate_cause", "causes")
        expected_causes = _classify(self.rule_state, self.model_state)
        if self.causes != expected_causes:
            raise TrialEligibilityReviewError("inconsistent_causes", "causes")

    @property
    def requires_human_review(self) -> bool:
        """Return whether this criterion has a classified disagreement."""

        return bool(self.causes)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete value-free criterion comparison."""

        return {
            "causes": [cause.value for cause in self.causes],
            "criterion_id": self.criterion_id,
            "kind": self.kind.value,
            "model_assessment_digest": self.model_assessment_digest,
            "model_citations": [
                citation.to_dict() for citation in self.model_citations
            ],
            "model_state": (
                None if self.model_state is None else self.model_state.value
            ),
            "model_uncertainty": self.model_uncertainty,
            "requires_human_review": self.requires_human_review,
            "rule_evidence_digests": list(self.rule_evidence_digests),
            "rule_state": self.rule_state.value,
            "time_window": (
                None if self.time_window is None else self.time_window.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class TrialEligibilityReviewPacket:
    """Structured reviewer packet that grants no operational authority."""

    record_digest: str = field(repr=False)
    definition_digest: str = field(repr=False)
    rule_explanation_digest: str = field(repr=False)
    comparisons: tuple[CriterionDisagreement, ...]
    packet_digest: str
    schema: str = TRIAL_ELIGIBILITY_REVIEW_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "record_digest",
            "definition_digest",
            "rule_explanation_digest",
            "packet_digest",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if (
            type(self.comparisons) is not tuple
            or not self.comparisons
            or any(
                type(comparison) is not CriterionDisagreement
                for comparison in self.comparisons
            )
        ):
            raise TrialEligibilityReviewError("invalid_comparisons", "comparisons")
        identifiers = tuple(value.criterion_id for value in self.comparisons)
        if identifiers != tuple(sorted(identifiers)):
            raise TrialEligibilityReviewError("comparisons_not_sorted", "comparisons")
        if len(set(identifiers)) != len(identifiers):
            raise TrialEligibilityReviewError("duplicate_comparison", "comparisons")
        if self.schema != TRIAL_ELIGIBILITY_REVIEW_SCHEMA:
            raise TrialEligibilityReviewError("invalid_schema", "schema")
        if self.packet_digest != _digest(self._unsigned_dict()):
            raise TrialEligibilityReviewError("packet_digest_mismatch", "packet_digest")

    @property
    def requires_human_review(self) -> bool:
        """Return whether any criterion must be reviewed."""

        return any(value.requires_human_review for value in self.comparisons)

    @property
    def authorizes_enrollment(self) -> bool:
        """Return ``False`` because a review packet cannot enroll a candidate."""

        return False

    @property
    def authorizes_contact(self) -> bool:
        """Return ``False`` because a review packet cannot authorize outreach."""

        return False

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "authorizes_contact": False,
            "authorizes_enrollment": False,
            "comparisons": [value.to_dict() for value in self.comparisons],
            "definition_digest": self.definition_digest,
            "record_digest": self.record_digest,
            "requires_human_review": self.requires_human_review,
            "rule_explanation_digest": self.rule_explanation_digest,
            "schema": self.schema,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic reviewer packet and explicit safety gates."""

        value = self._unsigned_dict()
        value["packet_digest"] = self.packet_digest
        return value

    def to_json(self) -> str:
        """Serialize the packet with stable key ordering."""

        return _canonical_json(self.to_dict())


def build_trial_eligibility_review_packet(
    rule_explanation: CohortMembershipExplanation,
    model_assessments: Iterable[ModelCriterionAssessment],
) -> TrialEligibilityReviewPacket:
    """Compare rule and model results and build a structured review packet.

    Missing, unknown, conflicting, or contradictory results are classified for
    each criterion. The function is local and performs no enrollment, contact,
    notification, approval, filesystem, database, or network operation.

    Args:
        rule_explanation: Deterministic criterion-level rule explanation.
        model_assessments: Evidence-backed model assessments to compare.

    Returns:
        A deterministic, value-free review packet.

    Raises:
        TrialEligibilityReviewError: If inputs are malformed or refer to an
            undeclared or duplicate criterion.
    """

    if type(rule_explanation) is not CohortMembershipExplanation:
        raise TrialEligibilityReviewError(
            "invalid_rule_explanation", "rule_explanation"
        )
    assessments = _normalize_assessments(model_assessments)
    rule_ids = {criterion.criterion_id for criterion in rule_explanation.criteria}
    for assessment in assessments:
        if assessment.criterion_id not in rule_ids:
            raise TrialEligibilityReviewError("unknown_criterion", "criterion_id")

    assessment_by_id = {value.criterion_id: value for value in assessments}
    comparisons = tuple(
        _compare_criterion(criterion, assessment_by_id.get(criterion.criterion_id))
        for criterion in rule_explanation.criteria
    )
    unsigned_fields = {
        "authorizes_contact": False,
        "authorizes_enrollment": False,
        "comparisons": [value.to_dict() for value in comparisons],
        "definition_digest": rule_explanation.definition_digest,
        "record_digest": rule_explanation.record_digest,
        "requires_human_review": any(
            value.requires_human_review for value in comparisons
        ),
        "rule_explanation_digest": rule_explanation.explanation_digest,
        "schema": TRIAL_ELIGIBILITY_REVIEW_SCHEMA,
    }
    return TrialEligibilityReviewPacket(
        record_digest=rule_explanation.record_digest,
        definition_digest=rule_explanation.definition_digest,
        rule_explanation_digest=rule_explanation.explanation_digest,
        comparisons=comparisons,
        packet_digest=_digest(unsigned_fields),
    )


def _compare_criterion(
    rule: CriterionEvaluation,
    model: ModelCriterionAssessment | None,
) -> CriterionDisagreement:
    model_state = None if model is None else model.state
    return CriterionDisagreement(
        criterion_id=rule.criterion_id,
        kind=rule.kind,
        rule_state=rule.state,
        rule_evidence_digests=rule.evidence_digests,
        time_window=rule.time_window,
        model_state=model_state,
        model_assessment_digest=(None if model is None else model.assessment_digest),
        model_citations=() if model is None else model.citations,
        model_uncertainty=None if model is None else model.uncertainty,
        causes=_classify(rule.state, model_state),
    )


def _classify(
    rule_state: CriterionState,
    model_state: CriterionState | None,
) -> tuple[DisagreementCause, ...]:
    causes: list[DisagreementCause] = []
    if rule_state is CriterionState.UNKNOWN:
        causes.append(DisagreementCause.RULE_EVIDENCE_MISSING)
    elif rule_state is CriterionState.CONFLICT:
        causes.append(DisagreementCause.RULE_EVIDENCE_CONFLICT)

    if model_state is None:
        causes.append(DisagreementCause.MODEL_ASSESSMENT_MISSING)
    elif model_state is CriterionState.UNKNOWN:
        causes.append(DisagreementCause.MODEL_ASSESSMENT_UNKNOWN)
    elif model_state is CriterionState.CONFLICT:
        causes.append(DisagreementCause.MODEL_ASSESSMENT_CONFLICT)
    elif rule_state in (CriterionState.MET, CriterionState.NOT_MET):
        if rule_state is not model_state:
            causes.append(DisagreementCause.OUTCOME_CONFLICT)
    return tuple(sorted(causes, key=lambda cause: cause.value))


def _validate_model_fields(value: CriterionDisagreement) -> None:
    if value.model_state is None:
        if (
            value.model_assessment_digest is not None
            or value.model_citations
            or value.model_uncertainty is not None
        ):
            raise TrialEligibilityReviewError(
                "inconsistent_model_assessment", "model_state"
            )
        return
    if type(value.model_state) is not CriterionState:
        raise TrialEligibilityReviewError("invalid_assessment_state", "model_state")
    if value.model_assessment_digest is None:
        raise TrialEligibilityReviewError(
            "missing_assessment_digest", "model_assessment_digest"
        )
    _validate_digest(value.model_assessment_digest, "model_assessment_digest")
    if type(value.model_citations) is not tuple or not value.model_citations:
        raise TrialEligibilityReviewError("invalid_citations", "model_citations")
    if any(type(item) is not EligibilityCitation for item in value.model_citations):
        raise TrialEligibilityReviewError("invalid_citation", "model_citations")
    if value.model_citations != tuple(sorted(value.model_citations)):
        raise TrialEligibilityReviewError("citations_not_sorted", "model_citations")
    if len(set(value.model_citations)) != len(value.model_citations):
        raise TrialEligibilityReviewError("duplicate_citation", "model_citations")
    _validate_uncertainty(value.model_uncertainty)


def _normalize_citations(
    citations: Iterable[EligibilityCitation],
) -> tuple[EligibilityCitation, ...]:
    values = _bounded_tuple(citations, "citations")
    if not values:
        raise TrialEligibilityReviewError("empty_collection", "citations")
    if any(type(value) is not EligibilityCitation for value in values):
        raise TrialEligibilityReviewError("invalid_citation", "citations")
    normalized = tuple(sorted(values))
    if len(set(normalized)) != len(normalized):
        raise TrialEligibilityReviewError("duplicate_citation", "citations")
    return normalized


def _normalize_assessments(
    assessments: Iterable[ModelCriterionAssessment],
) -> tuple[ModelCriterionAssessment, ...]:
    values = _bounded_tuple(assessments, "model_assessments")
    if any(type(value) is not ModelCriterionAssessment for value in values):
        raise TrialEligibilityReviewError(
            "invalid_model_assessment", "model_assessments"
        )
    normalized = tuple(sorted(values, key=lambda value: value.criterion_id))
    identifiers = tuple(value.criterion_id for value in normalized)
    if len(set(identifiers)) != len(identifiers):
        raise TrialEligibilityReviewError(
            "duplicate_model_assessment", "model_assessments"
        )
    return normalized


def _bounded_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TrialEligibilityReviewError("invalid_collection", field_name)
    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise TrialEligibilityReviewError("invalid_collection", field_name) from None
    items: list[Any] = []
    try:
        for item in iterator:
            if len(items) >= _MAX_ITEMS:
                raise TrialEligibilityReviewError("too_many_items", field_name)
            items.append(item)
    except (KeyboardInterrupt, SystemExit, TrialEligibilityReviewError):
        raise
    except Exception:
        raise TrialEligibilityReviewError("invalid_collection", field_name) from None
    return tuple(items)


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise TrialEligibilityReviewError("invalid_identifier", field_name)
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise TrialEligibilityReviewError("invalid_digest", field_name)
    return value


def _validate_digests(values: Any, field_name: str) -> None:
    if type(values) is not tuple or len(values) > _MAX_ITEMS:
        raise TrialEligibilityReviewError("invalid_digests", field_name)
    for value in values:
        _validate_digest(value, field_name)
    if values != tuple(sorted(values)):
        raise TrialEligibilityReviewError("digests_not_sorted", field_name)
    if len(set(values)) != len(values):
        raise TrialEligibilityReviewError("duplicate_digest", field_name)


def _validate_offset(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_OFFSET:
        raise TrialEligibilityReviewError("invalid_offset", field_name)
    return value


def _validate_uncertainty(value: Any) -> float:
    if type(value) not in (int, float):
        raise TrialEligibilityReviewError("invalid_uncertainty", "uncertainty")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise TrialEligibilityReviewError("invalid_uncertainty", "uncertainty")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "TRIAL_ELIGIBILITY_REVIEW_SCHEMA",
    "CriterionDisagreement",
    "DisagreementCause",
    "EligibilityCitation",
    "ModelCriterionAssessment",
    "TrialEligibilityReviewError",
    "TrialEligibilityReviewPacket",
    "build_trial_eligibility_review_packet",
]
