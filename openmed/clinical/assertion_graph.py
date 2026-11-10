"""Document-level reconciliation for section-scoped clinical assertions.

The graph is deliberately deterministic and small:

* Nodes are keyed by coreference/entity identity when available, then by coded
  concept identity, then by normalized mention text.
* Edges are contributing span-level ``ClinicalAssertion`` records with section
  and offset provenance.
* Axis reconciliation gives the highest authority to assessment/plan evidence,
  then HPI, then history sections, then social/family history, then unlabeled
  spans. Lower-authority disagreements are preserved as evidence but do not
  override a higher-authority section.
* If the highest-authority evidence for an entity disagrees on an axis, the
  graph reports a contradiction with the conflicting spans and sections instead
  of silently merging it into a clean assertion.

The result is an assistive document-organization artifact for review and
downstream processing. It is not a clinical decision engine.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from .context import (
    AFFIRMED,
    CERTAIN,
    HYPOTHETICAL,
    NEGATED,
    NEGATION_VALUES,
    PATIENT_EXPERIENCER,
    RECENT,
    TEMPORALITY_VALUES,
    UNCERTAIN,
    Certainty,
    ClinicalAssertion,
    Negation,
    apply_section_context,
    assert_context_axes,
    canonical_section_label,
    resolve_negation,
    resolve_temporality,
    resolve_uncertainty,
)
from .problem_list import (
    UNCONFIRMED,
    ProblemClinicalStatus,
    clinical_status_from_assertion,
)

AssertionAxis = Literal["negation", "temporality", "certainty"]
SpanOffset = tuple[int, int]

ASSERTION_GRAPH_ADVISORY = (
    "Document assertion graph reconciliation is an assistive consistency layer "
    "for review and downstream organization, not a clinical decision or "
    "medical-device instruction."
)

ASSERTION_GRAPH_AXES: tuple[AssertionAxis, ...] = (
    "negation",
    "temporality",
    "certainty",
)

# Assessment and plan are treated as the highest-authority document summary
# sections. HPI can resolve incidental history mentions, but assessment/plan
# disagreements remain reportable contradictions.
SECTION_RECONCILIATION_PRECEDENCE = {
    "assessment": 50,
    "plan": 50,
    "history_of_present_illness": 40,
    "past_medical_history": 30,
    "history": 30,
    "social_history": 20,
    "family_history": 10,
}

UNKNOWN_SECTION_PRECEDENCE = 25

_AXIS_VALUE_PRECEDENCE: dict[AssertionAxis, dict[str, int]] = {
    "negation": {NEGATED: 2, AFFIRMED: 1},
    "temporality": {RECENT: 3, "historical": 2, HYPOTHETICAL: 1},
    "certainty": {CERTAIN: 2, UNCERTAIN: 1},
}
_TEXT_KEYS = ("text", "label", "name", "term", "surface", "normalized_text")
_SECTION_KEYS = ("section", "section_label", "section_name")
_ENTITY_KEYS = (
    "entity_id",
    "coref_id",
    "coref_entity_id",
    "concept_id",
    "normalized_concept",
    "normalized_text",
)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AssertionEvidence:
    """A span-level assertion edge contributing to an entity graph node."""

    entity_key: str
    text: str
    assertion: ClinicalAssertion
    section: str | None = None
    canonical_section: str | None = None
    offset: SpanOffset | None = None
    span_id: str | None = None
    system: str | None = None
    code: str | None = None


@dataclass(frozen=True)
class AxisProvenance:
    """Evidence chosen to support one reconciled assertion axis."""

    axis: AssertionAxis
    value: str
    section: str | None
    span_id: str | None
    offset: SpanOffset | None
    text: str
    section_priority: int


@dataclass(frozen=True)
class AssertionConflict:
    """A top-authority intra-document contradiction for one entity axis."""

    entity_key: str
    axis: AssertionAxis
    values: tuple[str, ...]
    evidence: tuple[AxisProvenance, ...]

    @property
    def sections(self) -> tuple[str | None, ...]:
        """Return distinct section labels represented in the conflict."""

        seen: set[str | None] = set()
        sections: list[str | None] = []
        for item in self.evidence:
            if item.section not in seen:
                sections.append(item.section)
                seen.add(item.section)
        return tuple(sections)


@dataclass(frozen=True)
class ReconciledAssertion:
    """Document-level assertion for one graph entity."""

    entity_key: str
    text: str
    assertion: ClinicalAssertion
    clinical_status: ProblemClinicalStatus
    evidence: tuple[AssertionEvidence, ...]
    provenance: tuple[AxisProvenance, ...]
    conflicted_axes: tuple[AssertionAxis, ...] = ()

    @property
    def provenance_by_axis(self) -> dict[AssertionAxis, AxisProvenance]:
        """Return chosen provenance keyed by assertion axis."""

        return {item.axis: item for item in self.provenance}


@dataclass(frozen=True)
class AssertionGraphResult:
    """Return value from ``reconcile_assertions``."""

    assertions: tuple[ReconciledAssertion, ...]
    conflicts: tuple[AssertionConflict, ...]
    disclaimer: str = ASSERTION_GRAPH_ADVISORY

    @property
    def assertions_by_entity(self) -> dict[str, ReconciledAssertion]:
        """Return reconciled assertions keyed by graph entity."""

        return {assertion.entity_key: assertion for assertion in self.assertions}


@dataclass(frozen=True)
class _AxisOutcome:
    value: str
    provenance: AxisProvenance
    conflict: AssertionConflict | None


def reconcile_assertions(
    spans: Iterable[AssertionEvidence | Mapping[str, object]],
) -> AssertionGraphResult:
    """Reconcile span-level clinical assertions into a document graph.

    Args:
        spans: Span-level assertion edges. Mappings may provide ``entity_id`` or
            ``coref_id`` for coreference grouping, ``system`` + ``code`` for
            coded concept grouping, or text-like fields for normalized-text
            grouping. Assertion axes may be supplied under an ``assertion``
            mapping/object or directly as ``negation``, ``temporality``,
            ``certainty``, and ``experiencer`` fields.

    Returns:
        Per-entity reconciled assertions, conflict records, and an assistive
        disclaimer. Output ordering is deterministic and independent of span
        input order.
    """

    grouped: dict[str, list[AssertionEvidence]] = defaultdict(list)
    for raw_span in spans:
        evidence = _coerce_evidence(raw_span)
        grouped[evidence.entity_key].append(evidence)

    assertions: list[ReconciledAssertion] = []
    conflicts: list[AssertionConflict] = []

    for entity_key in sorted(grouped):
        evidence = tuple(sorted(grouped[entity_key], key=_evidence_sort_key))
        reconciled_evidence = _patient_reconciliation_evidence(evidence)
        outcomes = {
            axis: _reconcile_axis(entity_key, axis, reconciled_evidence)
            for axis in ASSERTION_GRAPH_AXES
        }
        entity_conflicts = tuple(
            outcome.conflict for outcome in outcomes.values() if outcome.conflict
        )
        conflicts.extend(entity_conflicts)

        assertion = ClinicalAssertion(
            negation=cast(Negation, outcomes["negation"].value),
            temporality=outcomes["temporality"].value,
            certainty=cast(Certainty, outcomes["certainty"].value),
            experiencer=_reconcile_experiencer(reconciled_evidence),
        )
        clinical_status = (
            UNCONFIRMED
            if entity_conflicts
            else clinical_status_from_assertion(assertion)
        )
        assertions.append(
            ReconciledAssertion(
                entity_key=entity_key,
                text=evidence[0].text,
                assertion=assertion,
                clinical_status=clinical_status,
                evidence=evidence,
                provenance=tuple(outcome.provenance for outcome in outcomes.values()),
                conflicted_axes=tuple(conflict.axis for conflict in entity_conflicts),
            )
        )

    return AssertionGraphResult(
        assertions=tuple(assertions),
        conflicts=tuple(sorted(conflicts, key=_conflict_sort_key)),
    )


def _coerce_evidence(
    raw_span: AssertionEvidence | Mapping[str, object],
) -> AssertionEvidence:
    if isinstance(raw_span, AssertionEvidence):
        return _complete_evidence(raw_span, raw_span, raw_span.section)
    if not isinstance(raw_span, Mapping):
        raise TypeError("assertion graph spans must be mappings or AssertionEvidence")

    text = _clean_text(_first_present(raw_span, _TEXT_KEYS))
    section = _optional_text(_first_present(raw_span, _SECTION_KEYS))
    system = _optional_text(raw_span.get("system"))
    code = _optional_text(raw_span.get("code"))
    evidence = AssertionEvidence(
        entity_key=_entity_key(raw_span, text=text, system=system, code=code),
        text=text,
        assertion=_coerce_assertion(raw_span, section),
        section=section,
        canonical_section=canonical_section_label(section),
        offset=_offset(raw_span),
        span_id=_optional_text(raw_span.get("span_id") or raw_span.get("id")),
        system=system,
        code=code,
    )
    return _complete_evidence(evidence, raw_span, section)


def _complete_evidence(
    evidence: AssertionEvidence,
    span: Any,
    section: Any,
) -> AssertionEvidence:
    assertion = evidence.assertion
    if assertion.negation is None:
        assertion = replace(assertion, negation=resolve_negation(span))
    assertion = apply_section_context(span, section, assertion)
    return replace(
        evidence,
        assertion=assertion,
        canonical_section=evidence.canonical_section
        or canonical_section_label(evidence.section),
    )


def _coerce_assertion(span: Mapping[str, object], section: Any) -> ClinicalAssertion:
    raw_assertion = span.get("assertion")
    if isinstance(raw_assertion, ClinicalAssertion):
        return raw_assertion
    if isinstance(raw_assertion, Mapping):
        return ClinicalAssertion(
            temporality=_axis_or_default(
                raw_assertion.get("temporality"),
                TEMPORALITY_VALUES,
                resolve_temporality(span),
                "temporality",
            ),
            certainty=cast(
                Certainty,
                _axis_or_default(
                    raw_assertion.get("certainty"),
                    (CERTAIN, UNCERTAIN),
                    resolve_uncertainty(span),
                    "certainty",
                ),
            ),
            negation=cast(
                Negation,
                _axis_or_default(
                    raw_assertion.get("negation"),
                    NEGATION_VALUES,
                    resolve_negation(span),
                    "negation",
                ),
            ),
            experiencer=_optional_text(raw_assertion.get("experiencer")),
        )

    if any(axis in span for axis in ASSERTION_GRAPH_AXES) or "experiencer" in span:
        return ClinicalAssertion(
            temporality=_axis_or_default(
                span.get("temporality"),
                TEMPORALITY_VALUES,
                resolve_temporality(span),
                "temporality",
            ),
            certainty=cast(
                Certainty,
                _axis_or_default(
                    span.get("certainty"),
                    (CERTAIN, UNCERTAIN),
                    resolve_uncertainty(span),
                    "certainty",
                ),
            ),
            negation=cast(
                Negation,
                _axis_or_default(
                    span.get("negation"),
                    NEGATION_VALUES,
                    resolve_negation(span),
                    "negation",
                ),
            ),
            experiencer=_optional_text(span.get("experiencer")),
        )

    return assert_context_axes(span, section=section)


def _axis_or_default(
    value: object,
    allowed_values: tuple[str, ...],
    default: str,
    axis: str,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{axis} must be a string")
    normalized = value.strip().casefold()
    if normalized not in allowed_values:
        raise ValueError(f"{axis} must be one of {', '.join(allowed_values)}")
    return normalized


def _patient_reconciliation_evidence(
    evidence: tuple[AssertionEvidence, ...],
) -> tuple[AssertionEvidence, ...]:
    patient_evidence = tuple(
        item
        for item in evidence
        if item.assertion.experiencer in (None, PATIENT_EXPERIENCER)
    )
    return patient_evidence or evidence


def _reconcile_axis(
    entity_key: str,
    axis: AssertionAxis,
    evidence: tuple[AssertionEvidence, ...],
) -> _AxisOutcome:
    chosen = sorted(
        evidence,
        key=lambda item: _axis_candidate_sort_key(axis, item),
    )[0]
    value = _axis_value(chosen.assertion, axis)
    provenance = _axis_provenance(axis, value, chosen)
    conflict = _axis_conflict(entity_key, axis, evidence)
    return _AxisOutcome(value=value, provenance=provenance, conflict=conflict)


def _axis_conflict(
    entity_key: str,
    axis: AssertionAxis,
    evidence: tuple[AssertionEvidence, ...],
) -> AssertionConflict | None:
    highest_priority = max(_section_priority(item) for item in evidence)
    top_evidence = tuple(
        item for item in evidence if _section_priority(item) == highest_priority
    )
    values = tuple(sorted({_axis_value(item.assertion, axis) for item in top_evidence}))
    if len(values) < 2:
        return None
    return AssertionConflict(
        entity_key=entity_key,
        axis=axis,
        values=values,
        evidence=tuple(
            _axis_provenance(axis, _axis_value(item.assertion, axis), item)
            for item in sorted(top_evidence, key=_evidence_sort_key)
        ),
    )


def _reconcile_experiencer(evidence: tuple[AssertionEvidence, ...]) -> str | None:
    explicit = {
        item.assertion.experiencer for item in evidence if item.assertion.experiencer
    }
    if not explicit:
        return None
    if PATIENT_EXPERIENCER in explicit:
        return PATIENT_EXPERIENCER
    return sorted(explicit)[0]


def _axis_value(assertion: ClinicalAssertion, axis: AssertionAxis) -> str:
    if axis == "negation":
        return assertion.negation or AFFIRMED
    if axis == "temporality":
        return assertion.temporality
    return assertion.certainty


def _axis_provenance(
    axis: AssertionAxis,
    value: str,
    evidence: AssertionEvidence,
) -> AxisProvenance:
    return AxisProvenance(
        axis=axis,
        value=value,
        section=evidence.section,
        span_id=evidence.span_id,
        offset=evidence.offset,
        text=evidence.text,
        section_priority=_section_priority(evidence),
    )


def _axis_candidate_sort_key(
    axis: AssertionAxis,
    evidence: AssertionEvidence,
) -> tuple[int, int, int, str, str]:
    start = evidence.offset[0] if evidence.offset is not None else 1_000_000_000
    return (
        -_section_priority(evidence),
        -_AXIS_VALUE_PRECEDENCE[axis].get(_axis_value(evidence.assertion, axis), 0),
        start,
        evidence.span_id or "",
        evidence.text.casefold(),
    )


def _section_priority(evidence: AssertionEvidence) -> int:
    if evidence.canonical_section is None:
        return UNKNOWN_SECTION_PRECEDENCE
    return SECTION_RECONCILIATION_PRECEDENCE.get(
        evidence.canonical_section,
        UNKNOWN_SECTION_PRECEDENCE,
    )


def _evidence_sort_key(
    evidence: AssertionEvidence,
) -> tuple[str, int, str, SpanOffset, str, str]:
    return (
        evidence.entity_key,
        -_section_priority(evidence),
        evidence.canonical_section or "",
        evidence.offset or (1_000_000_000, 1_000_000_000),
        evidence.span_id or "",
        evidence.text.casefold(),
    )


def _conflict_sort_key(
    conflict: AssertionConflict,
) -> tuple[str, str, tuple[str, ...]]:
    return conflict.entity_key, conflict.axis, conflict.values


def _entity_key(
    span: Mapping[str, object],
    *,
    text: str,
    system: str | None,
    code: str | None,
) -> str:
    for key in _ENTITY_KEYS:
        value = _optional_text(span.get(key))
        if value:
            return f"{key}:{_normalize_identity(value)}"
    if system and code:
        return f"code:{_normalize_identity(system)}|{_normalize_identity(code)}"
    return f"text:{_normalize_identity(text)}"


def _normalize_identity(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip()).casefold()


def _first_present(mapping: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("assertion evidence must include text")
    text = _WHITESPACE_RE.sub(" ", value.strip())
    if not text:
        raise ValueError("assertion evidence text must not be empty")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text fields must be strings when provided")
    text = _WHITESPACE_RE.sub(" ", value.strip())
    return text or None


def _offset(span: Mapping[str, object]) -> SpanOffset | None:
    if "offset" in span:
        value = span["offset"]
        if value is None:
            return None
        if (
            not isinstance(value, tuple | list)
            or len(value) != 2
            or not all(isinstance(part, int) for part in value)
        ):
            raise TypeError("offset must be a (start, end) integer pair")
        start, end = value
    else:
        start = span.get("start")
        end = span.get("end")
        if start is None and end is None:
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("start and end must be integers when provided")
    if start < 0 or end < start:
        raise ValueError("offset must satisfy 0 <= start <= end")
    return start, end


__all__ = [
    "ASSERTION_GRAPH_ADVISORY",
    "ASSERTION_GRAPH_AXES",
    "SECTION_RECONCILIATION_PRECEDENCE",
    "AssertionAxis",
    "AssertionConflict",
    "AssertionEvidence",
    "AssertionGraphResult",
    "AxisProvenance",
    "ReconciledAssertion",
    "SpanOffset",
    "reconcile_assertions",
]
