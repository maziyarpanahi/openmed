"""Relation-level assertion propagation (roadmap v2.0).

A relation inherits the assertion state of its arguments. A dose relation to a
medication in a hypothetical plan ("would start metformin if A1c rises") or a
finding relation to a negated problem must not be emitted as an asserted fact.
The base relation extractor ignores the ConText negation/temporality/uncertainty
axes, so grounding and FHIR export would record refuted or conditional relations
as real -- a correctness hazard.

This stage reads :func:`openmed.clinical.context.resolve_span_context` for both
the head and the attribute of a :class:`RelationCandidate` and tags the relation
with a verificationStatus-compatible status:

* ``confirmed``   -- both arguments asserted.
* ``refuted``     -- a negated argument (hardest boundary; takes precedence).
* ``conditional`` -- a hypothetical / conditional argument.
* ``possible``    -- an uncertain argument.

Refuted and conditional relations are excluded from the default asserted-fact
set but retained -- with offsets, tags, and a content hash, never raw note text
-- so the audit trace and FHIR exporter can honor them. The cue lexicons are
reused from ``context.py``; this module adds no new lexicon.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from openmed.core.audit import stable_hash

from ..context import HYPOTHETICAL, NEGATED, UNCERTAIN, resolve_span_context
from .candidate import RelationCandidate, SpanReference

ASSERTION_FILTER_ADVISORY = (
    "Relation assertion tags are derived deterministically from the ConText "
    "axes of each argument. Refuted and conditional relations are withheld from "
    "the default fact set but kept in the audit trace; verify before clinical "
    "use."
)

RELATION_CONFIRMED = "confirmed"
RELATION_REFUTED = "refuted"
RELATION_CONDITIONAL = "conditional"
RELATION_POSSIBLE = "possible"

#: Relation assertion statuses, ordered from asserted to most withheld.
RELATION_ASSERTION_STATUSES = (
    RELATION_CONFIRMED,
    RELATION_POSSIBLE,
    RELATION_CONDITIONAL,
    RELATION_REFUTED,
)

# Statuses kept out of the default patient-fact set (still audited).
_WITHHELD_STATUSES = frozenset({RELATION_REFUTED, RELATION_CONDITIONAL})

# verificationStatus tags -> FHIR Condition.verificationStatus values.
_FHIR_VERIFICATION_STATUS = {
    RELATION_CONFIRMED: "confirmed",
    RELATION_REFUTED: "refuted",
    RELATION_CONDITIONAL: "provisional",
    RELATION_POSSIBLE: "provisional",
}


@dataclass(frozen=True)
class RelationAssertion:
    """A relation tagged with its propagated assertion status."""

    relation: RelationCandidate
    status: str
    fhir_verification_status: str
    head_negation: str
    head_temporality: str
    head_certainty: str
    attribute_negation: str
    attribute_temporality: str
    attribute_certainty: str

    @property
    def asserted(self) -> bool:
        """Whether the relation belongs in the default asserted-fact set."""

        return self.status not in _WITHHELD_STATUSES

    def to_audit_entry(self) -> dict[str, object]:
        """Audit record with offsets, tags, and a content hash -- no note text."""

        head_offset = [self.relation.head.start, self.relation.head.end]
        attribute_offset = [
            self.relation.attribute.start,
            self.relation.attribute.end,
        ]
        payload: dict[str, object] = {
            "relation_type": self.relation.relation_type,
            "head_offset": head_offset,
            "attribute_offset": attribute_offset,
            "status": self.status,
            "fhir_verification_status": self.fhir_verification_status,
            "head_axes": {
                "negation": self.head_negation,
                "temporality": self.head_temporality,
                "certainty": self.head_certainty,
            },
            "attribute_axes": {
                "negation": self.attribute_negation,
                "temporality": self.attribute_temporality,
                "certainty": self.attribute_certainty,
            },
        }
        payload["content_hash"] = stable_hash(payload)
        return payload


def _span_axes(
    reference: SpanReference,
    document_text: str,
    language: str | None,
) -> tuple[str, str, str]:
    span = {
        "text": reference.text,
        "context": document_text,
        "start": reference.start,
        "end": reference.end,
    }
    result = resolve_span_context(span, language=language)
    return result.negation, result.temporality, result.certainty


def _combine_status(
    head: tuple[str, str, str],
    attribute: tuple[str, str, str],
) -> str:
    negations = {head[0], attribute[0]}
    temporalities = {head[1], attribute[1]}
    certainties = {head[2], attribute[2]}

    if NEGATED in negations:
        return RELATION_REFUTED
    if HYPOTHETICAL in temporalities:
        return RELATION_CONDITIONAL
    if UNCERTAIN in certainties:
        return RELATION_POSSIBLE
    return RELATION_CONFIRMED


def propagate_relation_assertion(
    relation: RelationCandidate,
    document_text: str,
    *,
    language: str | None = None,
) -> RelationAssertion:
    """Tag ``relation`` with the assertion status propagated from its arguments."""

    head = _span_axes(relation.head, document_text, language)
    attribute = _span_axes(relation.attribute, document_text, language)
    status = _combine_status(head, attribute)

    return RelationAssertion(
        relation=relation,
        status=status,
        fhir_verification_status=_FHIR_VERIFICATION_STATUS[status],
        head_negation=head[0],
        head_temporality=head[1],
        head_certainty=head[2],
        attribute_negation=attribute[0],
        attribute_temporality=attribute[1],
        attribute_certainty=attribute[2],
    )


def filter_asserted_relations(
    relations: Iterable[RelationCandidate],
    document_text: str | Mapping[int, str],
    *,
    language: str | None = None,
) -> tuple[list[RelationAssertion], list[RelationAssertion]]:
    """Split relations into the default asserted-fact set and withheld ones.

    ``document_text`` is either the shared source text for every relation, or a
    mapping from ``id(relation)`` to that relation's source text. Refuted and
    conditional relations are withheld from the asserted set but returned so the
    audit trace retains them.
    """

    asserted: list[RelationAssertion] = []
    withheld: list[RelationAssertion] = []
    for relation in relations:
        if isinstance(document_text, Mapping):
            text = document_text[id(relation)]
        else:
            text = document_text
        result = propagate_relation_assertion(relation, text, language=language)
        (asserted if result.asserted else withheld).append(result)
    return asserted, withheld


__all__ = [
    "ASSERTION_FILTER_ADVISORY",
    "RELATION_CONFIRMED",
    "RELATION_REFUTED",
    "RELATION_CONDITIONAL",
    "RELATION_POSSIBLE",
    "RELATION_ASSERTION_STATUSES",
    "RelationAssertion",
    "propagate_relation_assertion",
    "filter_asserted_relations",
]
