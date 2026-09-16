"""Guarded procedure-to-indication relation candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

PROCEDURE_INDICATION_ADVISORY = (
    "Procedure-to-indication links are unconfirmed evidence candidates. They do "
    "not judge appropriateness and require human confirmation."
)

_PROCEDURE_LABELS = frozenset({"INTERVENTION", "PROCEDURE", "SURGERY"})
_INDICATION_LABELS = frozenset(
    {"CONDITION", "DIAGNOSIS", "DISEASE", "FINDING", "INDICATION", "PROBLEM"}
)
_LINK_CUES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("performed_for", r"\bperformed\s+for\b"),
        ("indicated_for", r"\bindicated\s+for\b"),
        ("because_of", r"\bbecause\s+of\b"),
        ("due_to", r"\bdue\s+to\b"),
        ("to_evaluate", r"\bto\s+(?:evaluate|assess|investigate)\b"),
        ("to_treat", r"\bto\s+treat\b"),
        ("for", r"\bfor\b"),
    )
)


@dataclass(frozen=True)
class ProcedureIndicationCandidate:
    """One section-scoped procedure/indication pair for review."""

    candidate_id: str
    procedure: GuardedEvidenceSpan
    indication: GuardedEvidenceSpan
    linking_cue: GuardedEvidenceSpan
    indication_assertion: GuardedAssertion
    confidence: float
    confirmation_required: bool = True
    appropriateness_assessed: bool = False
    advisory: str = PROCEDURE_INDICATION_ADVISORY

    def __post_init__(self) -> None:
        if not self.confirmation_required:
            raise ValueError("procedure-indication candidates require confirmation")
        if self.appropriateness_assessed:
            raise ValueError(
                "procedure-indication candidates cannot assess appropriateness"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic offset-and-hash evidence."""

        return {
            "candidate_id": self.candidate_id,
            "relation_type": "procedure_to_indication",
            "procedure": self.procedure.to_dict(),
            "indication": self.indication.to_dict(),
            "linking_cue": self.linking_cue.to_dict(),
            "indication_assertion": self.indication_assertion.to_dict(),
            "confidence": self.confidence,
            "confirmation_required": self.confirmation_required,
            "appropriateness_assessed": self.appropriateness_assessed,
            "advisory": self.advisory,
        }


def generate_procedure_indication_candidates(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any]] | None = None,
    *,
    max_distance: int = 160,
) -> tuple[ProcedureIndicationCandidate, ...]:
    """Generate explicit, section-local procedure-to-indication candidates."""

    if type(max_distance) is not int or max_distance < 0:
        raise ValueError("max_distance must be a non-negative integer")
    inputs, _ = coerce_guarded_spans(text, spans, sections)
    procedures = _with_labels(inputs, _PROCEDURE_LABELS)
    indications = _with_labels(inputs, _INDICATION_LABELS)
    candidates: list[ProcedureIndicationCandidate] = []
    for procedure in procedures:
        for indication in indications:
            if not _eligible(text, procedure, indication, max_distance=max_distance):
                continue
            cue = _linking_cue(text, procedure.evidence, indication.evidence)
            if cue is None:
                continue
            cue_name, cue_start, cue_end = cue
            cue_evidence = evidence_from_offsets(
                text,
                label=f"PROCEDURE_INDICATION_CUE:{cue_name}",
                start=cue_start,
                end=cue_end,
                section=procedure.evidence.section,
            )
            distance = span_gap(procedure.evidence, indication.evidence)
            confidence = round(
                min(procedure.evidence.score, indication.evidence.score)
                * (1.0 / (1.0 + (distance / 80.0))),
                6,
            )
            candidate_id = stable_candidate_id(
                "procedure-indication",
                procedure.evidence.offsets,
                indication.evidence.offsets,
                cue_evidence.offsets,
            )
            candidates.append(
                ProcedureIndicationCandidate(
                    candidate_id=candidate_id,
                    procedure=procedure.evidence,
                    indication=indication.evidence,
                    linking_cue=cue_evidence,
                    indication_assertion=assertion_for_span(text, indication),
                    confidence=confidence,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.procedure.start,
                item.indication.start,
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
    procedure: GuardedSpanInput,
    indication: GuardedSpanInput,
    *,
    max_distance: int,
) -> bool:
    return (
        procedure.evidence.section == indication.evidence.section
        and same_sentence(text, procedure.evidence, indication.evidence)
        and span_gap(procedure.evidence, indication.evidence) <= max_distance
    )


def _linking_cue(
    text: str,
    procedure: GuardedEvidenceSpan,
    indication: GuardedEvidenceSpan,
) -> tuple[str, int, int] | None:
    window_start, window = evidence_window(text, procedure, indication)
    matches: list[tuple[int, int, str]] = []
    for name, pattern in _LINK_CUES:
        for match in pattern.finditer(window):
            start = window_start + match.start()
            end = window_start + match.end()
            if _inside_endpoint(start, end, procedure, indication):
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


__all__ = [
    "PROCEDURE_INDICATION_ADVISORY",
    "ProcedureIndicationCandidate",
    "generate_procedure_indication_candidates",
]
