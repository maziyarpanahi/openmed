"""Typed medication relation candidate schema."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openmed.core.decoding.spans import stable_span_key
from openmed.processing.advanced_ner import EntitySpan

MedicationAttributeType = Literal["dose", "route", "frequency", "duration"]
MedicationRelationType = Literal[
    "drug_to_dose",
    "drug_to_route",
    "drug_to_frequency",
    "drug_to_duration",
]

RELATION_SCHEMA_VERSION = 1
DRUG_TO_DOSE: MedicationRelationType = "drug_to_dose"
DRUG_TO_ROUTE: MedicationRelationType = "drug_to_route"
DRUG_TO_FREQUENCY: MedicationRelationType = "drug_to_frequency"
DRUG_TO_DURATION: MedicationRelationType = "drug_to_duration"

RELATION_ORDER: tuple[MedicationRelationType, ...] = (
    DRUG_TO_DOSE,
    DRUG_TO_ROUTE,
    DRUG_TO_FREQUENCY,
    DRUG_TO_DURATION,
)
RELATION_ATTRIBUTE_TYPES: dict[MedicationRelationType, MedicationAttributeType] = {
    DRUG_TO_DOSE: "dose",
    DRUG_TO_ROUTE: "route",
    DRUG_TO_FREQUENCY: "frequency",
    DRUG_TO_DURATION: "duration",
}
ATTRIBUTE_RELATION_TYPES: dict[MedicationAttributeType, MedicationRelationType] = {
    attribute_type: relation_type
    for relation_type, attribute_type in RELATION_ATTRIBUTE_TYPES.items()
}
SpanLabelPredicate = Callable[[str], bool]


def normalize_relation_label(label: str) -> str:
    """Return a stable normalized label for relation candidate filtering."""

    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


@dataclass(frozen=True)
class SpanReference:
    """Stable snapshot of an entity span and its source offsets."""

    text: str
    label: str
    start: int
    end: int
    score: float
    section: str | None = None

    @classmethod
    def from_entity(
        cls,
        span: EntitySpan,
        *,
        document_text: str | None = None,
        section: str | None = None,
    ) -> "SpanReference":
        """Create a stable reference from an ``EntitySpan``."""

        span_text = span.text
        if document_text is not None and 0 <= span.start <= span.end <= len(
            document_text
        ):
            span_text = document_text[span.start : span.end]
        return cls(
            text=span_text,
            label=span.label,
            start=span.start,
            end=span.end,
            score=float(span.score),
            section=section,
        )

    def offset_key(self) -> tuple[int, int]:
        """Return the character-offset identity for this span."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        payload: dict[str, Any] = {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }
        if self.section is not None:
            payload["section"] = self.section
        return payload


@dataclass(frozen=True)
class SpanPairCandidate:
    """Ordered span pair shared by deterministic relation extractors."""

    left: SpanReference
    right: SpanReference
    char_distance: int
    intervening_span_count: int

    def stable_key(self) -> tuple[int, int, int, int]:
        """Return a deterministic sort key for the span pair."""

        return (
            self.left.start,
            self.right.start,
            self.left.end,
            self.right.end,
        )


def generate_span_pair_candidates(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    include_labels: Iterable[str] | None = None,
    label_predicate: SpanLabelPredicate | None = None,
    max_char_distance: int | None = None,
    section_by_span: Mapping[tuple[int, int], str] | None = None,
) -> tuple[SpanPairCandidate, ...]:
    """Generate ordered non-overlapping span pairs for relation extractors.

    Args:
        text: Original document text.
        spans: Entity spans as ``EntitySpan`` objects or mappings.
        include_labels: Optional case-insensitive label allow-list.
        label_predicate: Optional predicate evaluated against raw span labels.
        max_char_distance: Optional maximum character gap between spans.
        section_by_span: Optional section labels keyed by ``(start, end)``.

    Returns:
        Deterministically ordered span-pair candidates with distance features.
    """

    refs = _coerce_relation_spans(
        text,
        spans,
        include_labels=include_labels,
        label_predicate=label_predicate,
        section_by_span=section_by_span,
    )
    pairs: list[SpanPairCandidate] = []
    for left_index, left in enumerate(refs):
        for right in refs[left_index + 1 :]:
            if left.end > right.start:
                continue
            char_distance = max(0, right.start - left.end)
            if max_char_distance is not None and char_distance > max_char_distance:
                continue
            pairs.append(
                SpanPairCandidate(
                    left=left,
                    right=right,
                    char_distance=char_distance,
                    intervening_span_count=_intervening_pair_span_count(
                        left,
                        right,
                        refs,
                    ),
                )
            )
    return tuple(sorted(pairs, key=lambda pair: pair.stable_key()))


def _coerce_relation_spans(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    include_labels: Iterable[str] | None,
    label_predicate: SpanLabelPredicate | None,
    section_by_span: Mapping[tuple[int, int], str] | None,
) -> tuple[SpanReference, ...]:
    section_by_span = section_by_span or {}
    normalized_labels = (
        {normalize_relation_label(label) for label in include_labels}
        if include_labels is not None
        else None
    )
    refs: list[SpanReference] = []
    for item in spans:
        span = item if isinstance(item, EntitySpan) else EntitySpan.from_mapping(item)
        if span.start < 0 or span.end < span.start or span.end > len(text):
            continue
        if normalized_labels is not None and (
            normalize_relation_label(span.label) not in normalized_labels
        ):
            continue
        if label_predicate is not None and not label_predicate(span.label):
            continue
        refs.append(
            SpanReference.from_entity(
                span,
                document_text=text,
                section=_relation_span_section(
                    item,
                    span=span,
                    section_by_span=section_by_span,
                ),
            )
        )
    return tuple(sorted(refs, key=stable_span_key))


def _relation_span_section(
    item: EntitySpan | Mapping[str, Any],
    *,
    span: EntitySpan,
    section_by_span: Mapping[tuple[int, int], str],
) -> str | None:
    if isinstance(item, Mapping):
        section = item.get("section")
        if section is not None:
            return str(section)
    return section_by_span.get((span.start, span.end))


def _intervening_pair_span_count(
    left: SpanReference,
    right: SpanReference,
    spans: tuple[SpanReference, ...],
) -> int:
    return sum(
        1
        for span in spans
        if span.offset_key() not in {left.offset_key(), right.offset_key()}
        and left.end <= span.start
        and span.end <= right.start
    )


@dataclass(frozen=True)
class RelationCandidate:
    """Candidate ``drug -> attribute`` edge before constrained decoding."""

    relation_type: MedicationRelationType
    head: SpanReference
    attribute: SpanReference
    score: float
    confidence: float
    features: dict[str, float]
    explanation: tuple[str, ...]

    @property
    def attribute_type(self) -> MedicationAttributeType:
        """Return the schema attribute type for this candidate."""

        return RELATION_ATTRIBUTE_TYPES[self.relation_type]

    def stable_key(self) -> tuple[float, int, int, int, int, str]:
        """Sort key used by the deterministic constrained decoder."""

        relation_rank = RELATION_ORDER.index(self.relation_type)
        return (
            -self.score,
            relation_rank,
            self.head.start,
            self.attribute.start,
            self.attribute.end,
            self.attribute.text.casefold(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        return {
            "relation_type": self.relation_type,
            "attribute_type": self.attribute_type,
            "head": self.head.to_dict(),
            "attribute": self.attribute.to_dict(),
            "score": self.score,
            "confidence": self.confidence,
            "features": {key: self.features[key] for key in sorted(self.features)},
            "explanation": list(self.explanation),
        }


@dataclass(frozen=True)
class MedicationRelation:
    """Resolved medication relation with provenance and normalization."""

    relation_type: MedicationRelationType
    attribute_type: MedicationAttributeType
    head: SpanReference
    attribute: SpanReference
    score: float
    confidence: float
    features: dict[str, float]
    normalized: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        payload: dict[str, Any] = {
            "relation_type": self.relation_type,
            "attribute_type": self.attribute_type,
            "head": self.head.to_dict(),
            "attribute": self.attribute.to_dict(),
            "head_offsets": {"start": self.head.start, "end": self.head.end},
            "attribute_offsets": {
                "start": self.attribute.start,
                "end": self.attribute.end,
            },
            "score": self.score,
            "confidence": self.confidence,
            "features": {key: self.features[key] for key in sorted(self.features)},
        }
        if self.normalized is not None:
            payload["normalized"] = dict(self.normalized)
        return payload


@dataclass(frozen=True)
class MedicationRelationGroup:
    """Medication head span plus its resolved typed attribute relations."""

    medication: SpanReference
    relations: tuple[MedicationRelation, ...]
    advisory: str

    @property
    def attributes(self) -> dict[MedicationAttributeType, MedicationRelation]:
        """Return relations keyed by attribute type."""

        return {relation.attribute_type: relation for relation in self.relations}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        return {
            "medication": self.medication.to_dict(),
            "relations": [relation.to_dict() for relation in self.relations],
            "advisory": self.advisory,
        }


__all__ = [
    "ATTRIBUTE_RELATION_TYPES",
    "DRUG_TO_DOSE",
    "DRUG_TO_DURATION",
    "DRUG_TO_FREQUENCY",
    "DRUG_TO_ROUTE",
    "MedicationAttributeType",
    "MedicationRelation",
    "MedicationRelationGroup",
    "MedicationRelationType",
    "RELATION_ATTRIBUTE_TYPES",
    "RELATION_ORDER",
    "RELATION_SCHEMA_VERSION",
    "RelationCandidate",
    "SpanLabelPredicate",
    "SpanPairCandidate",
    "SpanReference",
    "generate_span_pair_candidates",
    "normalize_relation_label",
]
