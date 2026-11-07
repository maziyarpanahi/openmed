"""Guarded medication-change relation candidates for human confirmation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openmed.clinical.events import extract_medication_change_events

from ._guarded import (
    GuardedAssertion,
    GuardedEvidenceSpan,
    GuardedSpanInput,
    assertion_for_span,
    coerce_guarded_spans,
    evidence_from_offsets,
    stable_candidate_id,
)

MedicationChangeType = Literal[
    "start",
    "stop",
    "hold",
    "resume",
    "dose_increase",
    "dose_decrease",
]

MEDICATION_CHANGE_ADVISORY = (
    "Medication-change relations are unconfirmed review candidates, not "
    "prescribing actions or treatment recommendations."
)

_ACTION_MAP: Mapping[str, MedicationChangeType] = {
    "started": "start",
    "stopped": "stop",
    "held": "hold",
    "restarted": "resume",
    "increased": "dose_increase",
    "decreased": "dose_decrease",
}
_MEDICATION_LABELS = frozenset(
    {"DRUG", "MED", "MEDICATION", "MEDICATION_NAME", "TREATMENT"}
)
_TIME_LABELS = frozenset({"DATE", "TIME", "TIMEX", "TIME_ANCHOR", "TIME_WINDOW"})
_DOSE_LABELS = frozenset({"DOSE", "DOSAGE", "OLD_DOSE", "NEW_DOSE", "STRENGTH"})


@dataclass(frozen=True)
class MedicationChangeCandidate:
    """One assertion-aware medication change that always requires review."""

    candidate_id: str
    change_type: MedicationChangeType
    medication: GuardedEvidenceSpan
    trigger: GuardedEvidenceSpan
    event_time: GuardedEvidenceSpan | None
    medication_assertion: GuardedAssertion
    event_assertion: GuardedAssertion
    confidence: float
    conflict_state: Literal["none", "unresolved"]
    review_required: bool = True
    prescribing_action: bool = False
    advisory: str = MEDICATION_CHANGE_ADVISORY

    def __post_init__(self) -> None:
        if not self.review_required:
            raise ValueError("medication-change candidates must require review")
        if self.prescribing_action:
            raise ValueError(
                "medication-change candidates cannot be prescribing actions"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic offset-and-hash evidence without source values."""

        return {
            "candidate_id": self.candidate_id,
            "relation_type": "medication_change",
            "change_type": self.change_type,
            "medication": self.medication.to_dict(),
            "trigger": self.trigger.to_dict(),
            "event_time": self.event_time.to_dict() if self.event_time else None,
            "medication_assertion": self.medication_assertion.to_dict(),
            "event_assertion": self.event_assertion.to_dict(),
            "confidence": self.confidence,
            "conflict_state": self.conflict_state,
            "review_required": self.review_required,
            "prescribing_action": self.prescribing_action,
            "advisory": self.advisory,
        }


def generate_medication_change_candidates(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any]] | None = None,
    *,
    max_distance: int = 160,
    include_detected_time_anchors: bool = True,
    language: str = "en",
) -> tuple[MedicationChangeCandidate, ...]:
    """Link medication spans to explicit change cues and optional event times.

    The existing deterministic clinical-event extractor supplies the bounded
    role graph. This adapter removes raw surfaces, adds assertion axes, and
    guarantees that every result remains an unconfirmed review candidate.

    Args:
        text: Source note indexed by the supplied spans.
        spans: Medication, dose, and time spans from an upstream extractor.
        sections: Optional validated section spans.
        max_distance: Maximum character gap used by event role linking.
        include_detected_time_anchors: Whether local English time cues may be
            detected by the existing event extractor.
        language: Explicit event-cue language supported by that extractor.

    Returns:
        Deterministically ordered, assertion-aware review candidates.
    """

    inputs, _ = coerce_guarded_spans(text, spans, sections)
    by_offsets = {item.evidence.offsets: item for item in inputs}
    mentions = tuple(
        mention for item in inputs if (mention := _event_mention(item)) is not None
    )
    frames = extract_medication_change_events(
        text,
        mentions,
        max_distance=max_distance,
        include_detected_time_anchors=include_detected_time_anchors,
        language=language,
    )
    candidates: list[MedicationChangeCandidate] = []
    for frame in frames:
        action_slots = frame.role_slots("action")
        medication_slots = frame.role_slots("drug")
        if not action_slots or not medication_slots:
            continue
        action_slot = action_slots[0]
        medication_slot = medication_slots[0]
        change_type = _ACTION_MAP.get(action_slot.value)
        medication_input = by_offsets.get((medication_slot.start, medication_slot.end))
        if change_type is None or medication_input is None:
            continue
        section = medication_input.evidence.section
        trigger = evidence_from_offsets(
            text,
            label="MEDICATION_CHANGE_TRIGGER",
            start=action_slot.start,
            end=action_slot.end,
            section=section,
            score=action_slot.score or 1.0,
        )
        trigger_input = GuardedSpanInput(evidence=trigger, data={})
        time_slots = frame.role_slots("time")
        event_time = (
            evidence_from_offsets(
                text,
                label="EVENT_TIME",
                start=time_slots[0].start,
                end=time_slots[0].end,
                section=section,
                score=time_slots[0].score or 1.0,
            )
            if time_slots
            else None
        )
        conflict_state = "unresolved" if frame.conflicts else "none"
        confidence = min(
            medication_input.evidence.score,
            trigger.score,
            time_slots[0].score if time_slots and time_slots[0].score else 1.0,
        )
        candidate_id = stable_candidate_id(
            "medication-change",
            change_type,
            medication_input.evidence.offsets,
            trigger.offsets,
            event_time.offsets if event_time else None,
        )
        candidates.append(
            MedicationChangeCandidate(
                candidate_id=candidate_id,
                change_type=change_type,
                medication=medication_input.evidence,
                trigger=trigger,
                event_time=event_time,
                medication_assertion=assertion_for_span(text, medication_input),
                event_assertion=assertion_for_span(text, trigger_input),
                confidence=round(float(confidence), 6),
                conflict_state=conflict_state,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.trigger.start,
                item.medication.start,
                item.change_type,
                item.candidate_id,
            ),
        )
    )


def _event_mention(item: GuardedSpanInput) -> dict[str, Any] | None:
    label = item.evidence.label.upper()
    if label in _MEDICATION_LABELS:
        role = "drug"
    elif label in _TIME_LABELS:
        role = "time"
    elif label in _DOSE_LABELS:
        role = "dose"
    else:
        return None
    return {
        "id": f"guarded:{item.evidence.start}:{item.evidence.end}:{role}",
        "label": role,
        "start": item.evidence.start,
        "end": item.evidence.end,
        "score": item.evidence.score,
        "text_hash": item.evidence.text_hash,
    }


__all__ = [
    "MEDICATION_CHANGE_ADVISORY",
    "MedicationChangeCandidate",
    "MedicationChangeType",
    "generate_medication_change_candidates",
]
