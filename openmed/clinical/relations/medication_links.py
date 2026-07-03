"""Deterministic medication attribute relation linking."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

from openmed.clinical.medication_sig import (
    MEDICATION_SIG_ADVISORY,
    normalize_medication_attribute,
)
from openmed.core.decoding.spans import stable_span_key
from openmed.processing.advanced_ner import EntitySpan

from .candidate import (
    ATTRIBUTE_RELATION_TYPES,
    RELATION_ATTRIBUTE_TYPES,
    RELATION_ORDER,
    RELATION_SCHEMA_VERSION,
    MedicationAttributeType,
    MedicationRelation,
    MedicationRelationGroup,
    MedicationRelationType,
    RelationCandidate,
    SpanReference,
)

MEDICATION_LINK_ADVISORY = (
    "Medication attribute linking is deterministic assistive support, not a "
    "prescription decision, and not a substitute for clinician review. "
    f"{MEDICATION_SIG_ADVISORY}"
)

DEFAULT_WEIGHTS_RESOURCE = "data/medication_link_weights.json"
_TOKEN_RE = re.compile(r"\b\w+(?:[-/]\w+)*\b")
_CLAUSE_BOUNDARY_RE = re.compile(r"[;\n]|(?:\s+-\s+)")


@dataclass(frozen=True)
class MedicationRelationScorer:
    """Feature-based scorer loaded from a versioned relation config."""

    config: Mapping[str, Any]

    @classmethod
    def from_default_config(cls) -> "MedicationRelationScorer":
        """Load the bundled versioned medication relation weights."""

        resource = resources.files(__package__).joinpath(DEFAULT_WEIGHTS_RESOURCE)
        with resource.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(config=_validate_config(config))

    def threshold(self, relation_type: MedicationRelationType) -> float:
        """Return the minimum raw score for a relation type."""

        return float(self.config["relations"][relation_type]["threshold"])

    def score(
        self,
        relation_type: MedicationRelationType,
        head: SpanReference,
        attribute: SpanReference,
        *,
        text: str,
        tokens: tuple["_Token", ...],
        sentences: tuple["_Sentence", ...],
        spans: tuple[SpanReference, ...],
    ) -> RelationCandidate:
        """Score a candidate edge and expose every contributing feature."""

        features = _candidate_features(
            relation_type,
            head,
            attribute,
            text=text,
            tokens=tokens,
            sentences=sentences,
            spans=spans,
        )
        weights = self.config["relations"][relation_type]["weights"]
        raw_score = sum(
            float(weights.get(name, 0.0)) * value for name, value in features.items()
        )
        confidence = 1.0 / (1.0 + math.exp(-raw_score))
        return RelationCandidate(
            relation_type=relation_type,
            head=head,
            attribute=attribute,
            score=round(raw_score, 6),
            confidence=round(confidence, 6),
            features={key: round(value, 6) for key, value in sorted(features.items())},
            explanation=tuple(
                key for key in sorted(features) if features[key] and key != "bias"
            ),
        )


@dataclass(frozen=True)
class _Token:
    start: int
    end: int


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int


def link_medication_attributes(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    section_by_span: Mapping[tuple[int, int], str] | None = None,
    scorer: MedicationRelationScorer | None = None,
) -> tuple[MedicationRelationGroup, ...]:
    """Link medication spans to dose, route, frequency, and duration spans.

    The decoder is deterministic, on-device, and explainable. It is assistive,
    not a prescription decision, and not a substitute for clinician review;
    this disclaimer is intentionally consistent with ``MEDICATION_SIG_ADVISORY``.

    Args:
        text: Original clinical text.
        spans: Entity spans for medications and their possible attributes.
            Each item may be an ``EntitySpan`` or a mapping with ``text``,
            ``label``, ``start``, ``end``, and optional ``score``/``section``.
        section_by_span: Optional section labels keyed by ``(start, end)``.
        scorer: Optional scorer instance. Defaults to bundled versioned weights.

    Returns:
        Ordered medication relation groups. Every emitted relation includes
        head and attribute character offsets that refer back to ``text``.
    """

    scorer = scorer or MedicationRelationScorer.from_default_config()
    span_refs = _coerce_spans(text, spans, section_by_span=section_by_span)
    drugs = tuple(span for span in span_refs if _is_drug_span(span))
    if not drugs:
        return ()

    attributes = tuple(
        (span, attribute_type)
        for span in span_refs
        if (attribute_type := _attribute_type(span)) is not None
    )
    tokens = _tokenize(text)
    sentences = _sentence_spans(text)
    candidates = _candidate_edges(
        drugs=drugs,
        attributes=attributes,
        text=text,
        tokens=tokens,
        sentences=sentences,
        spans=span_refs,
        scorer=scorer,
    )
    selected = _decode_assignments(candidates, scorer=scorer)
    selected_by_head: dict[tuple[int, int], list[RelationCandidate]] = {
        drug.offset_key(): [] for drug in drugs
    }
    for candidate in selected:
        selected_by_head.setdefault(candidate.head.offset_key(), []).append(candidate)

    groups = []
    for drug in drugs:
        relations = tuple(
            _candidate_to_relation(candidate)
            for candidate in sorted(
                selected_by_head.get(drug.offset_key(), ()),
                key=lambda item: (
                    RELATION_ORDER.index(item.relation_type),
                    item.attribute.start,
                    item.attribute.end,
                ),
            )
        )
        groups.append(
            MedicationRelationGroup(
                medication=drug,
                relations=relations,
                advisory=MEDICATION_LINK_ADVISORY,
            )
        )
    return tuple(groups)


def _validate_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if config.get("version") != RELATION_SCHEMA_VERSION:
        msg = (
            "Medication relation config version "
            f"{config.get('version')!r} does not match schema "
            f"{RELATION_SCHEMA_VERSION}."
        )
        raise ValueError(msg)
    relations = config.get("relations")
    if not isinstance(relations, Mapping):
        raise ValueError("Medication relation config must define relations.")
    missing = set(RELATION_ORDER) - set(relations)
    if missing:
        raise ValueError(
            f"Medication relation config missing weights for {sorted(missing)}."
        )
    for relation_type in RELATION_ORDER:
        relation_config = relations[relation_type]
        if not isinstance(relation_config, Mapping):
            raise ValueError(f"Invalid relation config for {relation_type}.")
        if "threshold" not in relation_config:
            raise ValueError(f"Missing threshold for {relation_type}.")
        weights = relation_config.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(f"Missing weights for {relation_type}.")
        if "bias" not in weights:
            raise ValueError(f"Missing bias weight for {relation_type}.")
    return config


def _coerce_spans(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    section_by_span: Mapping[tuple[int, int], str] | None,
) -> tuple[SpanReference, ...]:
    section_by_span = section_by_span or {}
    refs: list[SpanReference] = []
    for item in spans:
        span = item if isinstance(item, EntitySpan) else EntitySpan.from_mapping(item)
        if span.start < 0 or span.end < span.start or span.end > len(text):
            continue
        section = _span_section(item, span=span, section_by_span=section_by_span)
        refs.append(
            SpanReference.from_entity(span, document_text=text, section=section)
        )
    return tuple(sorted(refs, key=stable_span_key))


def _span_section(
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


def _candidate_edges(
    *,
    drugs: tuple[SpanReference, ...],
    attributes: tuple[tuple[SpanReference, MedicationAttributeType], ...],
    text: str,
    tokens: tuple[_Token, ...],
    sentences: tuple[_Sentence, ...],
    spans: tuple[SpanReference, ...],
    scorer: MedicationRelationScorer,
) -> tuple[RelationCandidate, ...]:
    candidates: list[RelationCandidate] = []
    for drug in drugs:
        for attribute, attribute_type in attributes:
            if drug.offset_key() == attribute.offset_key():
                continue
            relation_type = ATTRIBUTE_RELATION_TYPES[attribute_type]
            candidates.append(
                scorer.score(
                    relation_type,
                    drug,
                    attribute,
                    text=text,
                    tokens=tokens,
                    sentences=sentences,
                    spans=spans,
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.stable_key()))


def _decode_assignments(
    candidates: tuple[RelationCandidate, ...],
    *,
    scorer: MedicationRelationScorer,
) -> tuple[RelationCandidate, ...]:
    selected: list[RelationCandidate] = []
    used_attribute_edges: set[tuple[str, int, int]] = set()
    used_head_relation_edges: set[tuple[int, int, MedicationRelationType]] = set()

    for candidate in candidates:
        if candidate.score < scorer.threshold(candidate.relation_type):
            continue
        attribute_key = (
            candidate.relation_type,
            candidate.attribute.start,
            candidate.attribute.end,
        )
        head_relation_key = (
            candidate.head.start,
            candidate.head.end,
            candidate.relation_type,
        )
        if attribute_key in used_attribute_edges:
            continue
        if head_relation_key in used_head_relation_edges:
            continue
        used_attribute_edges.add(attribute_key)
        used_head_relation_edges.add(head_relation_key)
        selected.append(candidate)

    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.head.start,
                item.head.end,
                RELATION_ORDER.index(item.relation_type),
                item.attribute.start,
                item.attribute.end,
            ),
        )
    )


def _candidate_to_relation(candidate: RelationCandidate) -> MedicationRelation:
    attribute_type = RELATION_ATTRIBUTE_TYPES[candidate.relation_type]
    normalized = normalize_medication_attribute(
        attribute_type, candidate.attribute.text
    )
    return MedicationRelation(
        relation_type=candidate.relation_type,
        attribute_type=attribute_type,
        head=candidate.head,
        attribute=candidate.attribute,
        score=candidate.score,
        confidence=candidate.confidence,
        features=candidate.features,
        normalized=normalized,
    )


def _candidate_features(
    relation_type: MedicationRelationType,
    head: SpanReference,
    attribute: SpanReference,
    *,
    text: str,
    tokens: tuple[_Token, ...],
    sentences: tuple[_Sentence, ...],
    spans: tuple[SpanReference, ...],
) -> dict[str, float]:
    del relation_type
    same_sentence = _same_sentence(head, attribute, sentences)
    same_clause = _same_clause(head, attribute, text)
    known_same_section = (
        head.section is not None
        and attribute.section is not None
        and _normalize_section(head.section) == _normalize_section(attribute.section)
    )
    known_different_section = (
        head.section is not None
        and attribute.section is not None
        and _normalize_section(head.section) != _normalize_section(attribute.section)
    )
    return {
        "bias": 1.0,
        "same_sentence": 1.0 if same_sentence else 0.0,
        "cross_sentence": 0.0 if same_sentence else 1.0,
        "attribute_after_head": 1.0 if attribute.start >= head.end else 0.0,
        "attribute_before_head": 1.0 if attribute.end <= head.start else 0.0,
        "token_distance": float(_token_distance(head, attribute, tokens)),
        "intervening_span_count": float(
            _intervening_span_count(head, attribute, spans, drugs_only=False)
        ),
        "intervening_drug_count": float(
            _intervening_span_count(head, attribute, spans, drugs_only=True)
        ),
        "same_clause": 1.0 if same_clause else 0.0,
        "cross_clause": 0.0 if same_clause else 1.0,
        "known_same_section": 1.0 if known_same_section else 0.0,
        "known_different_section": 1.0 if known_different_section else 0.0,
        "recognized_normalization": _recognized_normalization(attribute),
    }


def _tokenize(text: str) -> tuple[_Token, ...]:
    return tuple(
        _Token(match.start(), match.end()) for match in _TOKEN_RE.finditer(text)
    )


def _sentence_spans(text: str) -> tuple[_Sentence, ...]:
    if not text:
        return ()
    sentences: list[_Sentence] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        if start < end:
            sentences.append(_Sentence(start, end))
        start = match.end()
    if start < len(text):
        sentences.append(_Sentence(start, len(text)))
    return tuple(sentences or (_Sentence(0, len(text)),))


def _same_sentence(
    head: SpanReference,
    attribute: SpanReference,
    sentences: tuple[_Sentence, ...],
) -> bool:
    return _sentence_index(head, sentences) == _sentence_index(attribute, sentences)


def _sentence_index(
    span: SpanReference, sentences: tuple[_Sentence, ...]
) -> int | None:
    for index, sentence in enumerate(sentences):
        if sentence.start <= span.start and span.end <= sentence.end:
            return index
    return None


def _same_clause(head: SpanReference, attribute: SpanReference, text: str) -> bool:
    left, right = sorted((head, attribute), key=lambda span: (span.start, span.end))
    between = text[left.end : right.start]
    return _CLAUSE_BOUNDARY_RE.search(between) is None


def _token_distance(
    head: SpanReference,
    attribute: SpanReference,
    tokens: tuple[_Token, ...],
) -> int:
    if not tokens:
        return max(
            0,
            attribute.start - head.end
            if attribute.start >= head.end
            else head.start - attribute.end,
        )
    head_start, head_end = _token_bounds(head, tokens)
    attr_start, attr_end = _token_bounds(attribute, tokens)
    if attr_start > head_end:
        return attr_start - head_end - 1
    if head_start > attr_end:
        return head_start - attr_end - 1
    return 0


def _token_bounds(span: SpanReference, tokens: tuple[_Token, ...]) -> tuple[int, int]:
    covered = [
        index
        for index, token in enumerate(tokens)
        if token.start < span.end and span.start < token.end
    ]
    if covered:
        return covered[0], covered[-1]
    insertion = 0
    for index, token in enumerate(tokens):
        if token.start >= span.start:
            insertion = index
            break
    else:
        insertion = len(tokens)
    return insertion, insertion


def _intervening_span_count(
    head: SpanReference,
    attribute: SpanReference,
    spans: tuple[SpanReference, ...],
    *,
    drugs_only: bool,
) -> int:
    left, right = sorted((head, attribute), key=lambda span: (span.start, span.end))
    count = 0
    for span in spans:
        if span.offset_key() in {head.offset_key(), attribute.offset_key()}:
            continue
        if left.end <= span.start and span.end <= right.start:
            if not drugs_only or _is_drug_span(span):
                count += 1
    return count


def _recognized_normalization(span: SpanReference) -> float:
    attribute_type = _attribute_type(span)
    if attribute_type not in {"frequency", "duration"}:
        return 0.0
    normalized = normalize_medication_attribute(attribute_type, span.text)
    return 1.0 if normalized and normalized.get("recognized") else 0.0


def _is_drug_span(span: SpanReference) -> bool:
    label = _normalize_label(span.label)
    return label in {"drug", "medication", "medicine", "med", "rx"} or (
        "drug" in label or "medication" in label
    )


def _attribute_type(span: SpanReference) -> MedicationAttributeType | None:
    label = _normalize_label(span.label)
    if label in {"dose", "dosage", "strength"} or "dose" in label or "dosage" in label:
        return "dose"
    if label == "route" or "route" in label:
        return "route"
    if label in {"frequency", "freq"} or "frequency" in label or "freq" in label:
        return "frequency"
    if label == "duration" or "duration" in label:
        return "duration"
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def _normalize_section(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", section.casefold()).strip("_")


__all__ = [
    "DEFAULT_WEIGHTS_RESOURCE",
    "MEDICATION_LINK_ADVISORY",
    "MedicationRelationScorer",
    "link_medication_attributes",
]
