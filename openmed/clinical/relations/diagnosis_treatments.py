"""Explicit-language diagnosis-to-treatment review candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._guarded import (
    GuardedAssertion,
    GuardedEvidenceSpan,
    GuardedSpanInput,
    assertion_for_span,
    coerce_guarded_spans,
    evidence_from_offsets,
    evidence_window,
    same_sentence,
    span_gap,
    stable_candidate_id,
)

DIAGNOSIS_TREATMENT_ADVISORY = (
    "Diagnosis-to-treatment links are unconfirmed evidence candidates. They do "
    "not recommend, evaluate, or prioritize treatment."
)

RelationUncertainty = Literal["none", "possible", "conditional", "refuted"]

_DIAGNOSIS_LABELS = frozenset(
    {"CONDITION", "DIAGNOSIS", "DISEASE", "FINDING", "PROBLEM"}
)
_TREATMENT_LABELS = frozenset(
    {"DRUG", "INTERVENTION", "MEDICATION", "PROCEDURE", "THERAPY", "TREATMENT"}
)
_DIAGNOSIS_FIRST_CUES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("treated_with", r"\btreat(?:ed|ment)?\s+with\b"),
        ("managed_with", r"\bmanag(?:ed|ement)\s+with\b"),
        ("started_on", r"\bstart(?:ed)?\s+on\b"),
        ("prescribed", r"\bprescribed\b"),
    )
)
_TREATMENT_FIRST_CUES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("indicated_for", r"\bindicated\s+for\b"),
        ("to_treat", r"\bto\s+treat\b"),
        ("for", r"\bfor\b"),
    )
)


@dataclass(frozen=True)
class DiagnosisTreatmentCandidate:
    """One explicit diagnosis/treatment association requiring review."""

    candidate_id: str
    diagnosis: GuardedEvidenceSpan
    treatment: GuardedEvidenceSpan
    linking_cue: GuardedEvidenceSpan
    diagnosis_assertion: GuardedAssertion
    treatment_assertion: GuardedAssertion
    uncertainty: RelationUncertainty
    confidence: float
    review_required: bool = True
    treatment_recommendation: bool = False
    treatment_evaluated: bool = False
    advisory: str = DIAGNOSIS_TREATMENT_ADVISORY

    def __post_init__(self) -> None:
        if not self.review_required:
            raise ValueError("diagnosis-treatment candidates must require review")
        if self.treatment_recommendation or self.treatment_evaluated:
            raise ValueError("diagnosis-treatment candidates cannot judge treatment")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic offset-and-hash evidence."""

        return {
            "candidate_id": self.candidate_id,
            "relation_type": "diagnosis_to_treatment",
            "diagnosis": self.diagnosis.to_dict(),
            "treatment": self.treatment.to_dict(),
            "linking_cue": self.linking_cue.to_dict(),
            "diagnosis_assertion": self.diagnosis_assertion.to_dict(),
            "treatment_assertion": self.treatment_assertion.to_dict(),
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "review_required": self.review_required,
            "treatment_recommendation": self.treatment_recommendation,
            "treatment_evaluated": self.treatment_evaluated,
            "advisory": self.advisory,
        }


def generate_diagnosis_treatment_candidates(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any]] | None = None,
    *,
    max_distance: int = 160,
    allowed_sections: Iterable[str] | None = None,
) -> tuple[DiagnosisTreatmentCandidate, ...]:
    """Generate candidates only when explicit local linking language exists.

    Args:
        text: Source note indexed by supplied spans.
        spans: Diagnosis and treatment spans from an upstream extractor.
        sections: Optional validated section spans.
        max_distance: Maximum character gap between relation endpoints.
        allowed_sections: Optional exact section-label allowlist.

    Returns:
        Deterministically ordered, review-required candidates.
    """

    if type(max_distance) is not int or max_distance < 0:
        raise ValueError("max_distance must be a non-negative integer")
    section_allowlist = (
        frozenset(str(value) for value in allowed_sections)
        if allowed_sections is not None
        else None
    )
    inputs, _ = coerce_guarded_spans(text, spans, sections)
    diagnoses = _with_labels(inputs, _DIAGNOSIS_LABELS)
    treatments = _with_labels(inputs, _TREATMENT_LABELS)
    candidates: list[DiagnosisTreatmentCandidate] = []
    for diagnosis in diagnoses:
        for treatment in treatments:
            if not _eligible(
                text,
                diagnosis,
                treatment,
                max_distance=max_distance,
                allowed_sections=section_allowlist,
            ):
                continue
            cue = _explicit_cue(text, diagnosis.evidence, treatment.evidence)
            if cue is None:
                continue
            cue_name, cue_start, cue_end = cue
            cue_evidence = evidence_from_offsets(
                text,
                label=f"DIAGNOSIS_TREATMENT_CUE:{cue_name}",
                start=cue_start,
                end=cue_end,
                section=diagnosis.evidence.section,
            )
            diagnosis_assertion = assertion_for_span(text, diagnosis)
            treatment_assertion = assertion_for_span(text, treatment)
            distance = span_gap(diagnosis.evidence, treatment.evidence)
            confidence = round(
                min(diagnosis.evidence.score, treatment.evidence.score)
                * (1.0 / (1.0 + (distance / 80.0))),
                6,
            )
            candidate_id = stable_candidate_id(
                "diagnosis-treatment",
                diagnosis.evidence.offsets,
                treatment.evidence.offsets,
                cue_evidence.offsets,
            )
            candidates.append(
                DiagnosisTreatmentCandidate(
                    candidate_id=candidate_id,
                    diagnosis=diagnosis.evidence,
                    treatment=treatment.evidence,
                    linking_cue=cue_evidence,
                    diagnosis_assertion=diagnosis_assertion,
                    treatment_assertion=treatment_assertion,
                    uncertainty=_relation_uncertainty(
                        diagnosis_assertion,
                        treatment_assertion,
                    ),
                    confidence=confidence,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.diagnosis.start,
                item.treatment.start,
                item.linking_cue.start,
                item.candidate_id,
            ),
        )
    )


def _with_labels(
    inputs: Sequence[GuardedSpanInput], labels: frozenset[str]
) -> tuple[GuardedSpanInput, ...]:
    return tuple(item for item in inputs if item.evidence.label.upper() in labels)


def _eligible(
    text: str,
    diagnosis: GuardedSpanInput,
    treatment: GuardedSpanInput,
    *,
    max_distance: int,
    allowed_sections: frozenset[str] | None,
) -> bool:
    section = diagnosis.evidence.section
    return (
        section == treatment.evidence.section
        and (allowed_sections is None or section in allowed_sections)
        and same_sentence(text, diagnosis.evidence, treatment.evidence)
        and span_gap(diagnosis.evidence, treatment.evidence) <= max_distance
    )


def _explicit_cue(
    text: str,
    diagnosis: GuardedEvidenceSpan,
    treatment: GuardedEvidenceSpan,
) -> tuple[str, int, int] | None:
    window_start, window = evidence_window(text, diagnosis, treatment)
    cues = (
        _DIAGNOSIS_FIRST_CUES
        if diagnosis.start <= treatment.start
        else _TREATMENT_FIRST_CUES
    )
    matches: list[tuple[int, int, str]] = []
    for name, pattern in cues:
        for match in pattern.finditer(window):
            start = window_start + match.start()
            end = window_start + match.end()
            if _inside_endpoint(start, end, diagnosis, treatment):
                continue
            matches.append((start, end, name))
    if not matches:
        return None
    start, end, cue_name = min(
        matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2])
    )
    return cue_name, start, end


def _inside_endpoint(
    start: int,
    end: int,
    left: GuardedEvidenceSpan,
    right: GuardedEvidenceSpan,
) -> bool:
    return (left.start <= start and end <= left.end) or (
        right.start <= start and end <= right.end
    )


def _relation_uncertainty(
    *assertions: GuardedAssertion,
) -> RelationUncertainty:
    if any(assertion.negation == "negated" for assertion in assertions):
        return "refuted"
    if any(assertion.temporality == "hypothetical" for assertion in assertions):
        return "conditional"
    if any(assertion.certainty != "certain" for assertion in assertions):
        return "possible"
    return "none"


__all__ = [
    "DIAGNOSIS_TREATMENT_ADVISORY",
    "DiagnosisTreatmentCandidate",
    "RelationUncertainty",
    "generate_diagnosis_treatment_candidates",
]
