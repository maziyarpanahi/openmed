"""Relation candidate schemas and script-agnostic candidate construction."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from openmed.core.decoding import SpanEdge, SpanNode
from openmed.core.labels import normalize_label
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
class RelationCandidateRule:
    """Language-specific rule used to construct typed relation candidates.

    Candidate construction operates only on source character offsets and cue
    substrings. It intentionally does not tokenize on whitespace, which keeps
    the same path valid for Chinese and Indic scripts.
    """

    relation_type: str
    source_relation: str
    head_labels: frozenset[str]
    tail_labels: frozenset[str]
    cues: tuple[str, ...]
    max_character_distance: int = 96

    def __post_init__(self) -> None:
        if not self.relation_type:
            raise ValueError("relation_type must be non-empty")
        if not self.source_relation:
            raise ValueError("source_relation must be non-empty")
        if not self.head_labels or not self.tail_labels:
            raise ValueError("relation rules require head and tail labels")
        if not self.cues:
            raise ValueError("relation rules require at least one cue")
        if self.max_character_distance < 0:
            raise ValueError("max_character_distance must be non-negative")


@dataclass(frozen=True)
class RelationCandidateBatch:
    """Span-graph inputs produced from already-extracted NER spans."""

    nodes: tuple[SpanNode, ...]
    candidates: tuple[SpanEdge, ...]
    spans_by_node_id: Mapping[str, SpanReference]


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？；;\n]")


def build_relation_candidates(
    text: str,
    spans: Iterable[Any],
    rules: Iterable[RelationCandidateRule],
    *,
    language: str,
) -> RelationCandidateBatch:
    """Build graph candidates from existing spans without word tokenization.

    Args:
        text: Original clinical text.
        spans: Existing NER spans with character offsets into ``text``.
        rules: Language-keyed relation rules.
        language: Language code recorded as safe graph provenance.

    Returns:
        Nodes, candidate edges, and the stable node-to-span lookup used by the
        shared :func:`openmed.core.decoding.decode_span_graph` decoder.
    """

    references = _coerce_relation_spans(text, spans)
    nodes: list[SpanNode] = []
    spans_by_node_id: dict[str, SpanReference] = {}
    for index, reference in enumerate(references):
        node_id = f"span-{index}"
        spans_by_node_id[node_id] = reference
        nodes.append(
            SpanNode(
                node_id=node_id,
                start=reference.start,
                end=reference.end,
                label=normalize_label(reference.label),
                score=reference.score,
                metadata={"language": language},
            )
        )

    candidates: list[SpanEdge] = []
    ordered_rules = sorted(
        rules,
        key=lambda rule: (rule.relation_type, rule.source_relation),
    )
    for head_node in nodes:
        for tail_node in nodes:
            if head_node.node_id == tail_node.node_id:
                continue
            head = spans_by_node_id[head_node.node_id]
            tail = spans_by_node_id[tail_node.node_id]
            distance = _character_distance(head, tail)
            between = _text_between(text, head, tail)
            if _SENTENCE_BOUNDARY_RE.search(between):
                continue
            window = text[min(head.start, tail.start) : max(head.end, tail.end)]
            for rule in ordered_rules:
                if head_node.label not in rule.head_labels:
                    continue
                if tail_node.label not in rule.tail_labels:
                    continue
                if distance > rule.max_character_distance:
                    continue
                matched_cue = _matched_cue(window, rule.cues)
                if matched_cue is None:
                    continue
                candidates.append(
                    SpanEdge(
                        head=head_node.node_id,
                        tail=tail_node.node_id,
                        label=rule.relation_type,
                        score=_candidate_score(head, tail, distance),
                        metadata={
                            "character_distance": distance,
                            "language": language,
                            "matched_cue": matched_cue,
                            "source_relation": rule.source_relation,
                        },
                    )
                )

    return RelationCandidateBatch(
        nodes=tuple(nodes),
        candidates=tuple(
            sorted(
                candidates,
                key=lambda edge: (edge.label, edge.head, edge.tail, -edge.score),
            )
        ),
        spans_by_node_id=MappingProxyType(spans_by_node_id),
    )


def _coerce_relation_spans(
    text: str,
    spans: Iterable[Any],
) -> tuple[SpanReference, ...]:
    references: list[SpanReference] = []
    for item in spans:
        if isinstance(item, SpanReference):
            start = item.start
            end = item.end
            label = item.label
            score = item.score
            section = item.section
        elif isinstance(item, EntitySpan):
            start = item.start
            end = item.end
            label = item.label
            score = item.score
            section = None
        else:
            data = item if isinstance(item, Mapping) else vars(item)
            metadata = data.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
            label = str(data.get("label", data.get("entity", "")))
            score = float(data.get("score", metadata.get("confidence", 1.0)))
            raw_section = data.get("section", metadata.get("section"))
            section = None if raw_section is None else str(raw_section)
        if not label or start < 0 or end <= start or end > len(text):
            continue
        references.append(
            SpanReference(
                text=text[start:end],
                label=str(label),
                start=start,
                end=end,
                score=float(score),
                section=section,
            )
        )

    unique = {
        (reference.start, reference.end, normalize_label(reference.label)): reference
        for reference in references
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda reference: (
                reference.start,
                reference.end,
                normalize_label(reference.label),
            ),
        )
    )


def _text_between(
    text: str,
    left: SpanReference,
    right: SpanReference,
) -> str:
    if left.end <= right.start:
        return text[left.end : right.start]
    if right.end <= left.start:
        return text[right.end : left.start]
    return ""


def _character_distance(left: SpanReference, right: SpanReference) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _matched_cue(window: str, cues: tuple[str, ...]) -> str | None:
    normalized_window = window.casefold()
    for cue in sorted(cues, key=lambda value: (-len(value), value)):
        if cue.casefold() in normalized_window:
            return cue
    return None


def _candidate_score(
    head: SpanReference,
    tail: SpanReference,
    distance: int,
) -> float:
    entity_confidence = max(0.0, min((head.score + tail.score) / 2.0, 1.0))
    proximity = 1.0 / (1.0 + float(distance))
    return round(0.65 + 0.2 * entity_confidence + 0.15 * proximity, 6)


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
    "RelationCandidateBatch",
    "RelationCandidateRule",
    "RelationCandidate",
    "SpanReference",
    "build_relation_candidates",
]
