"""Deterministic synthetic longitudinal trajectories using existing Journey contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import islice

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
    validate_contract_graph,
)

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_TYPES = frozenset(
    {"encounter", "condition", "medication", "laboratory", "procedure", "outcome"}
)
_SOURCE = "openmed.synthetic.longitudinal"


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    """A declared synthetic fact with effective/observed days and causal parents."""

    code: str
    kind: str
    day: int
    observed_day: int
    encounter_code: str
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.code, self.encounter_code):
            if not isinstance(value, str) or not _CODE.fullmatch(value):
                raise ValueError("invalid synthetic trajectory event code")
        if not isinstance(self.kind, str) or self.kind not in _TYPES:
            raise ValueError("invalid synthetic trajectory event kind")
        if (
            type(self.day) is not int
            or type(self.observed_day) is not int
            or not 0 <= self.day <= self.observed_day <= 36500
        ):
            raise ValueError("invalid synthetic trajectory event time")
        parents = tuple(islice(self.parents, 129))
        if (
            len(parents) > 128
            or any(not isinstance(v, str) or not _CODE.fullmatch(v) for v in parents)
            or len(set(parents)) != len(parents)
        ):
            raise ValueError("invalid synthetic trajectory parent codes")
        object.__setattr__(self, "parents", tuple(sorted(parents)))


@dataclass(frozen=True, slots=True)
class SyntheticTrajectoryDocument:
    """One observable synthetic note bound to an immutable Journey artifact."""

    artifact_id: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SyntheticTrajectory:
    """Shared synthetic patient identity, hidden fact graph and observable notes."""

    subject_id: str
    facts: tuple[ClinicalFact, ...]
    artifacts: tuple[ClinicalArtifact, ...]
    evidence: tuple[EvidenceLocator, ...]
    documents: tuple[SyntheticTrajectoryDocument, ...]
    source: str = _SOURCE
    synthetic: bool = True

    def report(self) -> dict[str, object]:
        """Return counts and opaque references without observable document text."""
        if validate_trajectory(self):
            raise ValueError("invalid synthetic trajectory report")
        return {
            "synthetic": self.synthetic,
            "source": self.source,
            "subject_id": self.subject_id,
            "fact_ids": [f.fact_id for f in self.facts],
            "artifact_ids": [a.artifact_id for a in self.artifacts],
            "fact_count": len(self.facts),
            "encounter_count": len({f.encounter_id for f in self.facts}),
        }


def generate_trajectory(
    events: tuple[TrajectoryEvent, ...],
    *,
    seed: int = 0,
    start_date: str = "2000-01-01",
    date_shift_days: int = 0,
) -> SyntheticTrajectory:
    """Materialize causal synthetic events, preserving intervals under date shifts."""
    if (
        type(seed) is not int
        or not 0 <= seed <= 2**63 - 1
        or type(date_shift_days) is not int
        or not -36500 <= date_shift_days <= 36500
    ):
        raise ValueError("invalid synthetic trajectory generation settings")
    try:
        start = date.fromisoformat(start_date) + timedelta(days=date_shift_days)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid synthetic trajectory start date") from None
    events = tuple(islice(events, 1001))
    if (
        not events
        or len(events) > 1000
        or any(not isinstance(e, TrajectoryEvent) for e in events)
    ):
        raise ValueError("invalid synthetic trajectory event collection")
    indexed = {event.code: event for event in events}
    if len(indexed) != len(events):
        raise ValueError("duplicate synthetic trajectory event code")
    for event in events:
        encounter = indexed.get(event.encounter_code)
        if (
            encounter is None
            or encounter.kind != "encounter"
            or encounter.encounter_code != encounter.code
            or encounter.day > event.day
        ):
            raise ValueError("invalid synthetic trajectory encounter reference")
        for parent_code in event.parents:
            parent = indexed.get(parent_code)
            if (
                parent is None
                or parent.code == event.code
                or parent.day > event.day
                or parent.observed_day > event.observed_day
            ):
                raise ValueError("invalid synthetic trajectory causal timeline")
    # Reject same-day parent cycles independently of temporal ordering.
    completed = set()
    remaining = set(indexed)
    while remaining:
        ready = {code for code in remaining if set(indexed[code].parents) <= completed}
        if not ready:
            raise ValueError("synthetic trajectory causal cycle")
        completed |= ready
        remaining -= ready
    subject = derived_opaque_id("subject", _SOURCE, seed)
    fact_ids = {code: derived_opaque_id("fact", subject, code) for code in indexed}
    encounter_ids = {
        code: derived_opaque_id("encounter", subject, code)
        for code, event in indexed.items()
        if event.kind == "encounter"
    }
    facts = []
    artifacts = []
    evidence = []
    documents = []
    for event in sorted(events, key=lambda e: (e.observed_day, e.day, e.code)):
        try:
            effective = (start + timedelta(days=event.day)).isoformat()
            observed = (
                start + timedelta(days=event.observed_day)
            ).isoformat() + "T00:00:00Z"
        except (ValueError, OverflowError):
            raise ValueError("synthetic trajectory exceeds date range") from None
        text = f"SYNTHETIC {event.kind.upper()} NOTE\nEvent: {event.code}\nEffective: {effective}\n"
        artifact_id = derived_opaque_id(
            "artifact", subject, event.code, date_shift_days
        )
        locator_id = derived_opaque_id("evidence", artifact_id, event.code)
        span_start = text.index(event.code)
        encounter_id = encounter_ids[event.encounter_code]
        artifact = ClinicalArtifact(
            artifact_id,
            "synthetic_note",
            "text/plain",
            sha256_digest(text),
            len(text.encode()),
            derived_opaque_id("source", _SOURCE),
            observed,
            subject_id=subject,
            encounter_id=encounter_id,
            attributes={"synthetic": True, "synthetic_source": _SOURCE},
        )
        locator = EvidenceLocator(
            locator_id,
            artifact_id,
            "text_span",
            {"start": span_start, "end": span_start + len(event.code)},
        )
        fact = ClinicalFact(
            fact_ids[event.code],
            subject,
            event.kind,
            {"synthetic_code": event.code},
            "synthetic",
            (locator_id,),
            canonical_digest(
                {
                    "source": _SOURCE,
                    "seed": seed,
                    "event": event.code,
                    "effective": effective,
                    "observed": observed,
                }
            ),
            encounter_id=encounter_id,
            parent_fact_ids=tuple(fact_ids[c] for c in event.parents),
            effective_time={"date": effective},
            attributes={
                "synthetic": True,
                "observed_at": observed,
                "event_code": event.code,
            },
        )
        facts.append(fact)
        artifacts.append(artifact)
        evidence.append(locator)
        documents.append(SyntheticTrajectoryDocument(artifact_id, text))
    trajectory = SyntheticTrajectory(
        subject, tuple(facts), tuple(artifacts), tuple(evidence), tuple(documents)
    )
    if validate_trajectory(trajectory):
        raise ValueError("generated synthetic trajectory violates declared constraints")
    return trajectory


def validate_trajectory(trajectory: SyntheticTrajectory) -> tuple[str, ...]:
    """Independently validate runtime graph references, causal time and note bindings."""
    if not isinstance(trajectory, SyntheticTrajectory) or len(trajectory.facts) > 1000:
        raise ValueError("invalid synthetic trajectory")
    errors = set()
    if not trajectory.facts:
        return ("empty_trajectory",)
    try:
        validate_contract_graph(
            artifacts=trajectory.artifacts,
            evidence=trajectory.evidence,
            facts=trajectory.facts,
        )
    except ValueError:
        errors.add("contract_graph_invalid")
    if trajectory.synthetic is not True or trajectory.source != _SOURCE:
        errors.add("synthetic_provenance_missing")
    facts = {f.fact_id: f for f in trajectory.facts}
    artifacts = {a.artifact_id: a for a in trajectory.artifacts}
    documents = {d.artifact_id: d.text for d in trajectory.documents}
    locators = {e.locator_id: e for e in trajectory.evidence}
    if len(documents) != len(trajectory.documents) or set(documents) != set(artifacts):
        errors.add("document_inventory_mismatch")
    for artifact_id, artifact in artifacts.items():
        text = documents.get(artifact_id)
        if (
            text is None
            or artifact.content_hash != sha256_digest(text)
            or artifact.byte_size != len(text.encode())
        ):
            errors.add("document_hash_mismatch")
    for fact in trajectory.facts:
        if (
            fact.subject_id != trajectory.subject_id
            or fact.attributes.get("synthetic") is not True
        ):
            errors.add("subject_or_synthetic_mismatch")
        effective = fact.effective_time.get("date")
        observed = fact.attributes.get("observed_at")
        valid_time = True
        try:
            event_date = date.fromisoformat(effective)
            observed_time = datetime.fromisoformat(observed)
            if observed_time.tzinfo is None or event_date > observed_time.date():
                errors.add("observation_before_event")
        except (TypeError, ValueError):
            errors.add("invalid_event_time")
            valid_time = False
        for parent_id in fact.parent_fact_ids:
            parent = facts.get(parent_id)
            if valid_time and parent is not None:
                try:
                    parent_date = date.fromisoformat(parent.effective_time.get("date"))
                    parent_observed = datetime.fromisoformat(
                        parent.attributes.get("observed_at")
                    )
                    if parent_observed.tzinfo is None:
                        raise ValueError
                    if parent_date > event_date or parent_observed > observed_time:
                        errors.add("causal_time_mismatch")
                except (TypeError, ValueError):
                    errors.add("invalid_event_time")
        for locator_id in fact.evidence_ids:
            locator = locators.get(locator_id)
            if locator is None:
                continue
            text = documents.get(locator.artifact_id, "")
            start, end = locator.location.get("start"), locator.location.get("end")
            if (
                locator.location_type != "text_span"
                or type(start) is not int
                or type(end) is not int
                or not isinstance(fact.value, Mapping)
                or text[start:end] != fact.value.get("synthetic_code")
            ):
                errors.add("evidence_binding_mismatch")
    return tuple(sorted(errors))
