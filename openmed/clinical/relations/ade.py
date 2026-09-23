"""Assertion-aware medication adverse-event and indication relations."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

from openmed.clinical.context import (
    FAMILY_EXPERIENCER,
    HYPOTHETICAL,
    NEGATED,
    UNCERTAIN,
    assert_context,
    canonical_section_label,
)
from openmed.clinical.sections import detect_sections
from openmed.core.audit import stable_hash

from .candidate import (
    RelationCandidate,
    SpanReference,
    generate_relation_candidates,
    split_sentence_offsets,
)
from .medication_links import (
    MedicationStatementRecord,
    extract_medication_relations,
    reconstruct_medication_statements,
)

ADERelationType = Literal["drug_to_ade", "drug_to_reason"]
ADEAssertionStatus = Literal["asserted", "refuted", "conditional", "possible"]

DRUG_TO_ADE: ADERelationType = "drug_to_ade"
DRUG_TO_REASON: ADERelationType = "drug_to_reason"
ADE_RELATION_SCHEMA_VERSION = 1
ADE_RELATION_DISCLAIMER = (
    "Medication adverse-event relations are assistive evidence for clinician "
    "review, not autonomous causality determinations or medical-device advice."
)

_ADE_LABELS = frozenset({"ADE", "ADVERSE_EVENT", "ADVERSE_REACTION"})
_REASON_LABELS = frozenset({"INDICATION", "REASON"})
_ADE_SECTION_HINTS = frozenset(
    {"allergies", "adverse_events", "adverse_reactions", "intolerances"}
)
_ADE_CUES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("caused", re.compile(r"(?<!\w)caus(?:ed|es|ing)(?!\w)", re.I), 0.94),
    ("induced", re.compile(r"(?<!\w)induc(?:ed|es|ing)(?!\w)", re.I), 0.93),
    ("after", re.compile(r"(?<!\w)after(?!\w)", re.I), 0.91),
    ("developed", re.compile(r"(?<!\w)develop(?:ed|s|ing)(?!\w)", re.I), 0.89),
    ("adverse_reaction", re.compile(r"(?<!\w)adverse\s+reaction(?!\w)", re.I), 0.95),
    ("side_effect", re.compile(r"(?<!\w)side[ -]?effect(?!\w)", re.I), 0.95),
    ("on_medication", re.compile(r"(?<!\w)(?:while\s+)?on(?!\w)", re.I), 0.82),
)
_REASON_CUES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("indicated_for", re.compile(r"(?<!\w)indicated\s+for(?!\w)", re.I), 0.95),
    ("treated_with", re.compile(r"(?<!\w)treat(?:ed|ing)?\s+with(?!\w)", re.I), 0.93),
    ("to_treat", re.compile(r"(?<!\w)to\s+treat(?!\w)", re.I), 0.92),
    ("for", re.compile(r"(?<!\w)for(?!\w)", re.I), 0.88),
    ("because_of", re.compile(r"(?<!\w)because\s+of(?!\w)", re.I), 0.86),
)


class DrugProtRelationHead(Protocol):
    """Optional local relation-head slot compatible with DrugProt fine-tuning."""

    def predict_relation_scores(
        self,
        text: str,
        *,
        head_offsets: tuple[int, int],
        tail_offsets: tuple[int, int],
    ) -> Mapping[str, float]:
        """Return bounded scores keyed by ``drug_to_ade``/``drug_to_reason``."""


@dataclass(frozen=True)
class ADERelation:
    """One assertion-bound medication relation with offset-only provenance."""

    relation_type: ADERelationType
    drug: SpanReference
    event: SpanReference
    score: float
    tail_assertion_status: ADEAssertionStatus
    tail_negation: str
    tail_certainty: str
    tail_temporality: str
    tail_experiencer: str
    evidence_offsets: tuple[tuple[int, int], ...]
    provenance: Mapping[str, Any]
    disclaimer_flag: bool = True
    disclaimer: str = ADE_RELATION_DISCLAIMER

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("ADE relation score must be between 0 and 1")
        if len(self.evidence_offsets) < 2:
            raise ValueError("ADE relations require head and tail evidence offsets")
        if not self.disclaimer_flag:
            raise ValueError("ADE relation disclaimer_flag must remain true")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def head(self) -> SpanReference:
        """Return the medication relation head."""

        return self.drug

    @property
    def tail(self) -> SpanReference:
        """Return the adverse-event or indication tail."""

        return self.event

    @property
    def asserted_positive(self) -> bool:
        """Return whether this is an asserted patient-level causal ADE link."""

        return (
            self.relation_type == DRUG_TO_ADE
            and self.tail_assertion_status == "asserted"
            and self.tail_experiencer != FAMILY_EXPERIENCER
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic assistive relation payload."""

        return {
            "schema_version": ADE_RELATION_SCHEMA_VERSION,
            "relation_type": self.relation_type,
            "head": self.drug.to_dict(),
            "tail": self.event.to_dict(),
            "score": self.score,
            "tail_assertion_status": self.tail_assertion_status,
            "tail_context": {
                "negation": self.tail_negation,
                "certainty": self.tail_certainty,
                "temporality": self.tail_temporality,
                "experiencer": self.tail_experiencer,
            },
            "evidence_offsets": [
                {"start": start, "end": end} for start, end in self.evidence_offsets
            ],
            "provenance": dict(self.provenance),
            "disclaimer_flag": self.disclaimer_flag,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class MedicationADERecord:
    """One medication regimen plus its assertion-gated relation set."""

    medication: SpanReference
    regimen: MedicationStatementRecord | None
    relations: tuple[ADERelation, ...]
    disclaimer_flag: bool = True
    disclaimer: str = ADE_RELATION_DISCLAIMER

    def __post_init__(self) -> None:
        if not self.disclaimer_flag:
            raise ValueError("Medication ADE disclaimer_flag must remain true")
        if any(
            relation.drug.offset_key() != self.medication.offset_key()
            for relation in self.relations
        ):
            raise ValueError("Medication ADE relations must share one drug head")

    @property
    def adverse_events(self) -> tuple[ADERelation, ...]:
        """Return only asserted patient-level adverse events."""

        return tuple(
            relation for relation in self.relations if relation.asserted_positive
        )

    @property
    def reasons(self) -> tuple[ADERelation, ...]:
        """Return asserted patient-level medication indications."""

        return tuple(
            relation
            for relation in self.relations
            if relation.relation_type == DRUG_TO_REASON
            and relation.tail_assertion_status == "asserted"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a FHIR-exporter-consumable medication evidence record."""

        return {
            "record_type": "MedicationADERecord",
            "medication": self.medication.to_dict(),
            "regimen": self.regimen.to_dict() if self.regimen is not None else None,
            "adverse_events": [relation.to_dict() for relation in self.adverse_events],
            "reasons": [relation.to_dict() for relation in self.reasons],
            "relations": [relation.to_dict() for relation in self.relations],
            "disclaimer_flag": self.disclaimer_flag,
            "disclaimer": self.disclaimer,
        }


def extract_ade_relations(
    text: str,
    spans: Iterable[Any],
    *,
    relation_head: DrugProtRelationHead | None = None,
    min_score: float = 0.5,
) -> tuple[ADERelation, ...]:
    """Extract assertion-aware drug-to-ADE and drug-to-Reason relations.

    The default path is deterministic and cue-based. A caller may inject a
    local DrugProt-compatible relation head; no model or network dependency is
    loaded automatically. Negated ADE tails are emitted as ``refuted`` and are
    never positive causal links. Family-experiencer tails are excluded from the
    returned patient relation set.

    Args:
        text: Original clinical note indexed by every span.
        spans: Existing medication, problem, ADE, indication, and regimen spans.
        relation_head: Optional caller-supplied local relation scorer.
        min_score: Minimum score for an emitted typed relation.

    Returns:
        Deterministically ordered assistive relations with evidence offsets,
        assertion status, provenance, and a mandatory disclaimer flag.
    """

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    span_items = tuple(spans)
    candidates = generate_relation_candidates(
        text,
        span_items,
        max_window=0,
        allow_adjacent_sections=False,
    )
    sections = detect_sections(text)
    relations: list[ADERelation] = []
    for candidate in candidates:
        if candidate.relation_type not in {"drug_to_problem", "drug_to_indication"}:
            continue
        drug = _restore_surface(candidate.head, text)
        event = _restore_surface(candidate.tail, text)
        context_span: dict[str, Any] = {
            "text": event.text,
            "start": event.start,
            "end": event.end,
        }
        if event.section is not None:
            context_span["section"] = event.section
        [context] = assert_context(
            text,
            [context_span],
            section_experiencer=(
                FAMILY_EXPERIENCER
                if canonical_section_label(event.section) == "family_history"
                else None
            ),
            sections=sections,
        )
        experiencer = str(context["experiencer"])
        if experiencer == FAMILY_EXPERIENCER:
            continue
        typed = _type_relation(text, candidate, relation_head=relation_head)
        if typed is None:
            continue
        relation_type, score, cue_id, cue_offsets, source = typed
        if score < min_score:
            continue
        assertion_status = _assertion_status(context)
        evidence_offsets = (
            drug.offset_key(),
            event.offset_key(),
            *((cue_offsets,) if cue_offsets is not None else ()),
        )
        provenance_payload = {
            "candidate_hash": stable_hash(
                {
                    "head": drug.offset_key(),
                    "tail": event.offset_key(),
                    "relation_type": relation_type,
                }
            ),
            "cue_id": cue_id,
            "source": source,
        }
        relations.append(
            ADERelation(
                relation_type=relation_type,
                drug=drug,
                event=event,
                score=round(score, 6),
                tail_assertion_status=assertion_status,
                tail_negation=str(context["negation"]),
                tail_certainty=str(context["uncertainty"]),
                tail_temporality=str(context["temporality"]),
                tail_experiencer=experiencer,
                evidence_offsets=evidence_offsets,
                provenance=provenance_payload,
            )
        )

    return tuple(
        sorted(
            relations,
            key=lambda relation: (
                relation.drug.start,
                relation.event.start,
                relation.relation_type,
            ),
        )
    )


def reconstruct_medication_ade_records(
    text: str,
    spans: Iterable[Any],
    *,
    relations: Iterable[ADERelation] | None = None,
    relation_head: DrugProtRelationHead | None = None,
) -> tuple[MedicationADERecord, ...]:
    """Compose medication regimen and ADE evidence into one record per drug."""

    span_items = tuple(spans)
    ade_relations = tuple(
        relations
        if relations is not None
        else extract_ade_relations(text, span_items, relation_head=relation_head)
    )
    regimen_relations = extract_medication_relations(text, span_items)
    regimens = reconstruct_medication_statements(regimen_relations)
    regimen_by_offset = {
        regimen.medication.offset_key(): regimen for regimen in regimens
    }
    drug_by_offset = {
        relation.drug.offset_key(): relation.drug for relation in ade_relations
    }
    drug_by_offset.update(
        {regimen.medication.offset_key(): regimen.medication for regimen in regimens}
    )

    return tuple(
        MedicationADERecord(
            medication=medication,
            regimen=regimen_by_offset.get(offset),
            relations=tuple(
                relation
                for relation in ade_relations
                if relation.drug.offset_key() == offset
            ),
        )
        for offset, medication in sorted(drug_by_offset.items())
    )


def _restore_surface(reference: SpanReference, text: str) -> SpanReference:
    return SpanReference(
        text=text[reference.start : reference.end],
        label=reference.label,
        start=reference.start,
        end=reference.end,
        score=reference.score,
        section=reference.section,
        text_hash=reference.text_hash,
    )


def _assertion_status(context: Mapping[str, Any]) -> ADEAssertionStatus:
    if context["negation"] == NEGATED:
        return "refuted"
    if context["temporality"] == HYPOTHETICAL:
        return "conditional"
    if context["uncertainty"] == UNCERTAIN:
        return "possible"
    return "asserted"


def _type_relation(
    text: str,
    candidate: RelationCandidate,
    *,
    relation_head: DrugProtRelationHead | None,
) -> tuple[ADERelationType, float, str, tuple[int, int] | None, str] | None:
    if relation_head is not None:
        raw_scores = relation_head.predict_relation_scores(
            text,
            head_offsets=candidate.head.offset_key(),
            tail_offsets=candidate.tail.offset_key(),
        )
        scores = {
            relation_type: _bounded_score(raw_scores.get(relation_type, 0.0))
            for relation_type in (DRUG_TO_ADE, DRUG_TO_REASON)
        }
        relation_type = max(scores, key=lambda key: (scores[key], key))
        return relation_type, scores[relation_type], "model_head", None, "model"

    tail_label = candidate.tail.label.upper().replace("-", "_").replace(" ", "_")
    if tail_label in _ADE_LABELS:
        return DRUG_TO_ADE, 0.97, "ade_label", None, "cue_fallback"
    if tail_label in _REASON_LABELS:
        return DRUG_TO_REASON, 0.97, "reason_label", None, "cue_fallback"
    section = (candidate.tail.section or "").casefold().replace(" ", "_")
    if section in _ADE_SECTION_HINTS:
        return DRUG_TO_ADE, 0.9, "ade_section", None, "cue_fallback"

    sentence_offsets = split_sentence_offsets(text)
    sentence_start, sentence_end = next(
        (
            offsets
            for offsets in sentence_offsets
            if offsets[0] <= candidate.head.start < offsets[1]
        ),
        (
            min(candidate.head.start, candidate.tail.start),
            max(candidate.head.end, candidate.tail.end),
        ),
    )
    sentence = text[sentence_start:sentence_end]
    corridor_start = min(candidate.head.end, candidate.tail.end)
    corridor_end = max(candidate.head.start, candidate.tail.start)
    matches: list[tuple[float, ADERelationType, str, tuple[int, int]]] = []
    for cue_id, pattern, score in _ADE_CUES:
        for match in pattern.finditer(sentence):
            cue_offsets = (
                sentence_start + match.start(),
                sentence_start + match.end(),
            )
            if corridor_start <= cue_offsets[0] and cue_offsets[1] <= corridor_end:
                matches.append((score, DRUG_TO_ADE, cue_id, cue_offsets))
    for cue_id, pattern, score in _REASON_CUES:
        for match in pattern.finditer(sentence):
            cue_offsets = (
                sentence_start + match.start(),
                sentence_start + match.end(),
            )
            if corridor_start <= cue_offsets[0] and cue_offsets[1] <= corridor_end:
                matches.append((score, DRUG_TO_REASON, cue_id, cue_offsets))
    if not matches:
        return None
    score, relation_type, cue_id, cue_offsets = max(
        matches,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return relation_type, score, cue_id, cue_offsets, "cue_fallback"


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(max(score, 0.0), 1.0)


__all__ = [
    "ADE_RELATION_DISCLAIMER",
    "ADE_RELATION_SCHEMA_VERSION",
    "DRUG_TO_ADE",
    "DRUG_TO_REASON",
    "ADEAssertionStatus",
    "ADERelation",
    "ADERelationType",
    "DrugProtRelationHead",
    "MedicationADERecord",
    "extract_ade_relations",
    "reconstruct_medication_ade_records",
]
