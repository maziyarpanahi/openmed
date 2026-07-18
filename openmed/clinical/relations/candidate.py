"""Typed medication relation candidate schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    "SpanReference",
]
