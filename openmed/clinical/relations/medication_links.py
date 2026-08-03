"""Deterministic medication attribute relation linking."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

from openmed.clinical.coreference import CoreferenceChain
from openmed.clinical.medication_sig import (
    MEDICATION_SIG_ADVISORY,
    normalize_medication_attribute,
)
from openmed.clinical.sections import detect_sections, validate_section_spans
from openmed.core.decoding.spans import stable_span_key
from openmed.processing.advanced_ner import EntitySpan

from .candidate import (
    ATTRIBUTE_RELATION_TYPES,
    RELATION_ATTRIBUTE_TYPES,
    RELATION_ORDER,
    RELATION_SCHEMA_VERSION,
    CoreferenceProvenance,
    CoreferenceSourceReference,
    MedicationAttributeType,
    MedicationRelation,
    MedicationRelationGroup,
    MedicationRelationType,
    Relation,
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
_DOSAGE_ATTRIBUTE_ORDER: tuple[MedicationAttributeType, ...] = (
    "dose",
    "route",
    "frequency",
    "duration",
    "form",
    "strength",
)


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


@dataclass(frozen=True)
class _CoreferenceBinding:
    chain: CoreferenceChain
    provenance: CoreferenceProvenance


@dataclass(frozen=True)
class MedicationStatementRecord:
    """One non-FHIR medication regimen record with structured dosage spans."""

    medication: SpanReference
    dosage: Mapping[MedicationAttributeType, SpanReference]
    indication: SpanReference | None
    relations: tuple[Relation, ...]
    advisory: str = MEDICATION_LINK_ADVISORY

    def __post_init__(self) -> None:
        """Freeze dosage order and validate the regimen head."""

        unsupported = set(self.dosage) - set(_DOSAGE_ATTRIBUTE_ORDER)
        if unsupported:
            raise ValueError(
                f"unsupported MedicationStatement dosage attributes: {unsupported}"
            )
        ordered_dosage = {
            attribute_type: self.dosage[attribute_type]
            for attribute_type in _DOSAGE_ATTRIBUTE_ORDER
            if attribute_type in self.dosage
        }
        if any(
            relation.head.offset_key() != self.medication.offset_key()
            for relation in self.relations
        ):
            raise ValueError("MedicationStatement relations must share one drug head")
        object.__setattr__(self, "dosage", MappingProxyType(ordered_dosage))

    @property
    def record_type(self) -> str:
        """Return the interoperability-facing record shape name."""

        return "MedicationStatement"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic non-FHIR medication statement mapping."""

        payload: dict[str, Any] = {
            "record_type": self.record_type,
            "medication": self.medication.to_dict(),
            "dosage": {
                attribute_type: span.to_dict()
                for attribute_type, span in self.dosage.items()
            },
            "relations": [relation.to_dict() for relation in self.relations],
            "advisory": self.advisory,
        }
        if self.indication is not None:
            payload["indication"] = self.indication.to_dict()
        return payload


def extract_medication_relations(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    sections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[Relation, ...]:
    """Extract typed medication-to-attribute relations from existing spans.

    The extractor supports dose, route, frequency, duration, form, strength,
    and indication tails. It resolves section metadata with
    :func:`openmed.clinical.sections.detect_sections` when callers do not pass
    precomputed sections, and it never links across sentence, line, clause, or
    known section boundaries.

    Args:
        text: Original clinical text.
        spans: Existing medication and clinical-concept spans with source offsets.
        sections: Optional precomputed contiguous section spans. When omitted,
            sections are detected locally and deterministically.

    Returns:
        Deterministically ordered ``Relation(head, type, tail, score)`` records.
        This assistive output is not a prescription decision and is not a
        substitute for clinician review.
    """

    span_items = tuple(spans)
    section_items = tuple(detect_sections(text) if sections is None else sections)
    section_by_span = _section_labels_by_span(text, span_items, section_items)
    groups = link_medication_attributes(
        text,
        span_items,
        section_by_span=section_by_span,
    )
    return tuple(
        Relation(
            head=relation.head,
            type=relation.attribute_type,
            tail=relation.attribute,
            score=relation.confidence,
        )
        for group in groups
        for relation in group.relations
    )


def reconstruct_medication_statements(
    relations: Iterable[Relation],
) -> tuple[MedicationStatementRecord, ...]:
    """Group medication relations into one non-FHIR record per regimen."""

    grouped: dict[tuple[int, int], list[Relation]] = {}
    medication_by_offset: dict[tuple[int, int], SpanReference] = {}
    for relation in relations:
        if not isinstance(relation, Relation):
            raise TypeError("relations must contain Relation values")
        offset = relation.head.offset_key()
        grouped.setdefault(offset, []).append(relation)
        medication_by_offset.setdefault(offset, relation.head)

    statements: list[MedicationStatementRecord] = []
    for offset in sorted(grouped):
        best_by_type: dict[MedicationAttributeType, Relation] = {}
        for relation in grouped[offset]:
            current = best_by_type.get(relation.type)
            if current is None or _resolved_relation_key(
                relation
            ) < _resolved_relation_key(current):
                best_by_type[relation.type] = relation
        ordered_relations = tuple(
            best_by_type[RELATION_ATTRIBUTE_TYPES[relation_type]]
            for relation_type in RELATION_ORDER
            if RELATION_ATTRIBUTE_TYPES[relation_type] in best_by_type
        )
        dosage = {
            attribute_type: best_by_type[attribute_type].tail
            for attribute_type in _DOSAGE_ATTRIBUTE_ORDER
            if attribute_type in best_by_type
        }
        indication_relation = best_by_type.get("indication")
        statements.append(
            MedicationStatementRecord(
                medication=medication_by_offset[offset],
                dosage=dosage,
                indication=(
                    indication_relation.tail
                    if indication_relation is not None
                    else None
                ),
                relations=ordered_relations,
            )
        )
    return tuple(statements)


def _resolved_relation_key(
    relation: Relation,
) -> tuple[float, int, int, int, str]:
    return (
        -relation.score,
        RELATION_ORDER.index(relation.relation_type),
        relation.tail.start,
        relation.tail.end,
        relation.tail.text.casefold(),
    )


def _section_labels_by_span(
    text: str,
    spans: Sequence[EntitySpan | Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], str]:
    if not sections:
        return {}
    validate_section_spans(text, sections)
    labels: dict[tuple[int, int], str] = {}
    for span in spans:
        offset = _source_span_offset(span)
        if offset is None:
            continue
        start, end = offset
        containing = next(
            (
                section
                for section in sections
                if int(section["start"]) <= start and end <= int(section["end"])
            ),
            None,
        )
        if containing is not None:
            labels[offset] = str(containing["label"])
    return labels


def _source_span_offset(
    span: EntitySpan | Mapping[str, Any],
) -> tuple[int, int] | None:
    if isinstance(span, EntitySpan):
        return span.start, span.end
    try:
        return int(span["start"]), int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None


def link_medication_attributes(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    section_by_span: Mapping[tuple[int, int], str] | None = None,
    scorer: MedicationRelationScorer | None = None,
    coreference_chains: Sequence[CoreferenceChain] = (),
) -> tuple[MedicationRelationGroup, ...]:
    """Link medication spans to seven supported regimen attribute types.

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
        coreference_chains: Optional document-local clinical coreference chains.
            Relation candidates retain mention-local proximity scoring, while
            emitted drug heads are rewritten to cluster representatives and
            same-cluster relation groups are consolidated.

    Returns:
        Ordered medication relation groups. Every emitted relation includes
        head and attribute character offsets that refer back to ``text``.
        Collapsed groups also retain every member offset and HMAC hash without
        copying additional source text into provenance.
    """

    scorer = scorer or MedicationRelationScorer.from_default_config()
    span_refs = _coerce_spans(text, spans, section_by_span=section_by_span)
    drugs = tuple(span for span in span_refs if _is_drug_span(span))
    if not drugs:
        return ()
    coreference_by_offset = _coreference_bindings(
        coreference_chains,
        text_length=len(text),
    )

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
    selected_by_group: dict[tuple[str, str], list[RelationCandidate]] = {
        _medication_group_key(drug, coreference_by_offset): [] for drug in drugs
    }
    for candidate in selected:
        group_key = _medication_group_key(candidate.head, coreference_by_offset)
        selected_by_group.setdefault(group_key, []).append(candidate)

    drug_by_offset = {drug.offset_key(): drug for drug in drugs}
    group_members: dict[tuple[str, str], list[SpanReference]] = {}
    for drug in drugs:
        group_key = _medication_group_key(drug, coreference_by_offset)
        group_members.setdefault(group_key, []).append(drug)

    groups = []
    for group_key, member_drugs in group_members.items():
        binding = coreference_by_offset.get(member_drugs[0].offset_key())
        medication = _representative_medication(
            member_drugs[0],
            binding,
            text=text,
            drug_by_offset=drug_by_offset,
        )
        selected_candidates = _collapse_cluster_candidates(
            selected_by_group.get(group_key, ())
        )
        relations = tuple(
            _candidate_to_relation(
                candidate,
                head=medication,
                coreference=binding.provenance if binding else None,
            )
            for candidate in selected_candidates
        )
        groups.append(
            MedicationRelationGroup(
                medication=medication,
                relations=relations,
                advisory=MEDICATION_LINK_ADVISORY,
                coreference=binding.provenance if binding else None,
            )
        )
    return tuple(groups)


def _coreference_bindings(
    chains: Sequence[CoreferenceChain],
    *,
    text_length: int,
) -> dict[tuple[int, int], _CoreferenceBinding]:
    bindings: dict[tuple[int, int], _CoreferenceBinding] = {}
    cluster_ids: set[str] = set()
    document_id: str | None = None
    for chain in chains:
        if not isinstance(chain, CoreferenceChain):
            raise TypeError("coreference_chains must contain CoreferenceChain values")
        if chain.chain_id in cluster_ids:
            raise ValueError("coreference cluster ids must be unique")
        cluster_ids.add(chain.chain_id)
        if chain.representative not in chain.members:
            raise ValueError("coreference representative must be a chain member")
        chain_document_ids = {member.doc_id for member in chain.members}
        if len(chain_document_ids) != 1:
            raise ValueError("coreference chains must be document-local")
        chain_document_id = next(iter(chain_document_ids))
        if document_id is not None and chain_document_id != document_id:
            raise ValueError("coreference chains must belong to one document")
        document_id = chain_document_id
        sources = tuple(
            CoreferenceSourceReference(
                start=member.start,
                end=member.end,
                text_hash=member.text_hash,
            )
            for member in chain.members
        )
        representative = CoreferenceSourceReference(
            start=chain.representative.start,
            end=chain.representative.end,
            text_hash=chain.representative.text_hash,
        )
        provenance = CoreferenceProvenance(
            cluster_id=chain.chain_id,
            representative=representative,
            supporting_mentions=sources,
        )
        binding = _CoreferenceBinding(chain=chain, provenance=provenance)
        for member in chain.members:
            if member.end > text_length:
                raise ValueError("coreference offsets are outside the source text")
            offset = (member.start, member.end)
            if offset in bindings:
                raise ValueError(
                    "one coreference source offset cannot belong to multiple clusters"
                )
            bindings[offset] = binding
    return bindings


def _medication_group_key(
    medication: SpanReference,
    bindings: Mapping[tuple[int, int], _CoreferenceBinding],
) -> tuple[str, str]:
    binding = bindings.get(medication.offset_key())
    if binding is not None:
        return "coreference", binding.chain.chain_id
    return "offset", f"{medication.start}:{medication.end}"


def _representative_medication(
    source: SpanReference,
    binding: _CoreferenceBinding | None,
    *,
    text: str,
    drug_by_offset: Mapping[tuple[int, int], SpanReference],
) -> SpanReference:
    if binding is None:
        return source
    representative = binding.chain.representative
    offset = (representative.start, representative.end)
    existing = drug_by_offset.get(offset)
    if existing is not None:
        return existing
    return SpanReference(
        text=text[representative.start : representative.end],
        label=source.label,
        start=representative.start,
        end=representative.end,
        score=(
            float(representative.score)
            if representative.score is not None
            else source.score
        ),
        section=representative.section or source.section,
    )


def _collapse_cluster_candidates(
    candidates: Iterable[RelationCandidate],
) -> tuple[RelationCandidate, ...]:
    best_by_type: dict[MedicationRelationType, RelationCandidate] = {}
    for candidate in candidates:
        current = best_by_type.get(candidate.relation_type)
        if current is None or candidate.stable_key() < current.stable_key():
            best_by_type[candidate.relation_type] = candidate
    return tuple(
        best_by_type[relation_type]
        for relation_type in RELATION_ORDER
        if relation_type in best_by_type
    )


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
            if not _candidate_is_in_scope(
                drug,
                attribute,
                text=text,
                sentences=sentences,
            ):
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


def _candidate_is_in_scope(
    head: SpanReference,
    attribute: SpanReference,
    *,
    text: str,
    sentences: tuple[_Sentence, ...],
) -> bool:
    if not _same_sentence(head, attribute, sentences):
        return False
    if not _same_clause(head, attribute, text):
        return False
    return not (
        head.section is not None
        and attribute.section is not None
        and _normalize_section(head.section) != _normalize_section(attribute.section)
    )


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


def _candidate_to_relation(
    candidate: RelationCandidate,
    *,
    head: SpanReference,
    coreference: CoreferenceProvenance | None,
) -> MedicationRelation:
    attribute_type = RELATION_ATTRIBUTE_TYPES[candidate.relation_type]
    normalized = normalize_medication_attribute(
        attribute_type, candidate.attribute.text
    )
    return MedicationRelation(
        relation_type=candidate.relation_type,
        attribute_type=attribute_type,
        head=head,
        attribute=candidate.attribute,
        score=candidate.score,
        confidence=candidate.confidence,
        features=candidate.features,
        normalized=normalized,
        coreference=coreference,
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
    if label == "strength" or "strength" in label:
        return "strength"
    if label in {"dose", "dosage"} or "dose" in label or "dosage" in label:
        return "dose"
    if label == "route" or "route" in label:
        return "route"
    if label in {"frequency", "freq"} or "frequency" in label or "freq" in label:
        return "frequency"
    if label == "duration" or "duration" in label:
        return "duration"
    if label in {"form", "dose_form", "dosage_form"} or label.endswith("_form"):
        return "form"
    if label in {"indication", "reason", "reason_for_use"} or ("indication" in label):
        return "indication"
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def _normalize_section(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", section.casefold()).strip("_")


__all__ = [
    "DEFAULT_WEIGHTS_RESOURCE",
    "MEDICATION_LINK_ADVISORY",
    "MedicationStatementRecord",
    "MedicationRelationScorer",
    "extract_medication_relations",
    "link_medication_attributes",
    "reconstruct_medication_statements",
]
