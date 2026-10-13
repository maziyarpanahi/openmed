"""Cross-artifact synthetic truth checks using existing Journey fact contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
)

_MODALITIES = frozenset({"note", "fhir", "omop", "image", "waveform"})
_CATEGORIES = frozenset(
    {
        "fact_value",
        "fact_identity",
        "effective_time",
        "fact_label",
        "coordinates",
        "missing_fact",
        "unexpected_fact",
        "missing_evidence",
        "artifact_reference",
    }
)
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class SyntheticArtifactView:
    """One synthetic representation expressed through existing Journey contracts."""

    modality: str
    artifact: ClinicalArtifact
    facts: tuple[ClinicalFact, ...]
    evidence: tuple[EvidenceLocator, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.modality, str)
            or self.modality not in _MODALITIES
            or not isinstance(self.artifact, ClinicalArtifact)
            or self.artifact.attributes.get("synthetic") is not True
        ):
            raise ValueError("artifact view requires a declared synthetic modality")
        if (
            not isinstance(self.facts, Sequence)
            or not isinstance(self.evidence, Sequence)
            or len(self.facts) > 10_000
            or len(self.evidence) > 10_000
        ):
            raise ValueError("synthetic artifact view exceeds record limits")
        if not all(
            isinstance(fact, ClinicalFact) and fact.attributes.get("synthetic") is True
            for fact in self.facts
        ) or not all(isinstance(locator, EvidenceLocator) for locator in self.evidence):
            raise ValueError(
                "view facts and evidence must use synthetic Journey contracts"
            )
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "evidence", tuple(self.evidence))


def validate_truth_alignment(
    truth_facts: Sequence[ClinicalFact],
    truth_evidence: Sequence[EvidenceLocator],
    views: Sequence[SyntheticArtifactView],
    *,
    expected_modalities: Sequence[str] = tuple(sorted(_MODALITIES)),
    partition: str = "train",
    negative_control_name: str | None = None,
    expected_corruptions: Sequence[str] = (),
) -> dict[str, object]:
    """Compare declared value, time, label and coordinates without echoing them.

    Every view declares the same shared facts; partial views must be represented
    by a separate truth subset. No modality or clinical fact is inferred. Named
    negative controls can establish expected failures only in a negative-control
    partition, and always remain ineligible for normal training export.
    """
    if not isinstance(partition, str) or partition not in {
        "train",
        "validation",
        "test",
        "negative_control",
    }:
        raise ValueError("unsupported synthetic dataset partition")
    for values, limit in (
        (expected_modalities, 5),
        (expected_corruptions, len(_CATEGORIES)),
    ):
        if (
            not isinstance(values, Sequence)
            or isinstance(values, str)
            or len(values) > limit
            or any(not isinstance(value, str) for value in values)
        ):
            raise ValueError("invalid bounded alignment category declarations")
    if not expected_modalities:
        raise ValueError("alignment requires at least one expected modality")
    expected = set(expected_modalities)
    if not expected <= _MODALITIES or len(expected) != len(expected_modalities):
        raise ValueError("invalid expected modalities")
    corruptions = set(expected_corruptions)
    if not corruptions <= _CATEGORIES or len(corruptions) != len(expected_corruptions):
        raise ValueError("invalid negative-control categories")
    if partition == "negative_control":
        if (
            not isinstance(negative_control_name, str)
            or not _CODE.fullmatch(negative_control_name)
            or not corruptions
        ):
            raise ValueError(
                "negative-control partition requires a named expected corruption"
            )
    elif negative_control_name is not None or corruptions:
        raise ValueError("intentional corruption cannot enter a normal partition")
    if (
        not isinstance(truth_facts, Sequence)
        or not isinstance(truth_evidence, Sequence)
        or not isinstance(views, Sequence)
        or not views
        or len(truth_facts) > 10_000
        or len(truth_evidence) > 10_000
        or len(views) > 128
    ):
        raise ValueError("synthetic truth package exceeds record limits")
    if (
        not all(
            isinstance(fact, ClinicalFact) and fact.attributes.get("synthetic") is True
            for fact in truth_facts
        )
        or not all(isinstance(locator, EvidenceLocator) for locator in truth_evidence)
        or not all(isinstance(view, SyntheticArtifactView) for view in views)
    ):
        raise ValueError("expected synthetic Journey truth and artifact views")
    truth = {fact.fact_id: fact for fact in truth_facts}
    locations = {locator.locator_id: locator for locator in truth_evidence}
    if len(truth) != len(truth_facts) or len(locations) != len(truth_evidence):
        raise ValueError("duplicate synthetic truth identity")
    if len({view.artifact.artifact_id for view in views}) != len(views):
        raise ValueError("duplicate synthetic artifact identity")
    if not truth:
        raise ValueError("synthetic truth must declare at least one fact")
    findings = []

    def finding(view: SyntheticArtifactView, fact_id: str, category: str) -> None:
        findings.append(
            {
                "artifact_reference": hashlib.sha256(
                    view.artifact.artifact_id.encode()
                ).hexdigest(),
                "fact_reference": hashlib.sha256(fact_id.encode()).hexdigest(),
                "category": category,
            }
        )

    for view in sorted(views, key=lambda item: item.artifact.artifact_id):
        observed = {fact.fact_id: fact for fact in view.facts}
        locators = {locator.locator_id: locator for locator in view.evidence}
        if len(observed) != len(view.facts) or len(locators) != len(view.evidence):
            raise ValueError("duplicate identity in synthetic artifact view")
        for fact_id in sorted(set(truth) - set(observed)):
            finding(view, fact_id, "missing_fact")
        for fact_id in sorted(set(observed) - set(truth)):
            finding(view, fact_id, "unexpected_fact")
        for fact_id in sorted(set(truth) & set(observed)):
            baseline, actual = truth[fact_id], observed[fact_id]
            for different, category in (
                (
                    actual.subject_id != baseline.subject_id
                    or actual.encounter_id != baseline.encounter_id,
                    "fact_identity",
                ),
                (
                    actual.value != baseline.value or actual.unit != baseline.unit,
                    "fact_value",
                ),
                (actual.effective_time != baseline.effective_time, "effective_time"),
                (
                    actual.fact_type != baseline.fact_type
                    or actual.status != baseline.status,
                    "fact_label",
                ),
            ):
                if different:
                    finding(view, fact_id, category)
            required_ids = {
                locator_id
                for locator_id in baseline.evidence_ids
                if locator_id not in locations
                or locations[locator_id].artifact_id == view.artifact.artifact_id
            }
            for _locator_id in sorted(required_ids - set(actual.evidence_ids)):
                finding(view, fact_id, "missing_evidence")
            for locator_id in actual.evidence_ids:
                actual_locator, expected_locator = (
                    locators.get(locator_id),
                    locations.get(locator_id),
                )
                if (
                    locator_id not in baseline.evidence_ids
                    or actual_locator is None
                    or expected_locator is None
                ):
                    finding(view, fact_id, "missing_evidence")
                    continue
                if actual_locator.artifact_id != view.artifact.artifact_id:
                    finding(view, fact_id, "artifact_reference")
                if (
                    actual_locator.location_type != expected_locator.location_type
                    or actual_locator.location != expected_locator.location
                    or actual_locator.transform != expected_locator.transform
                ):
                    finding(view, fact_id, "coordinates")
    missing = sorted(expected - {view.modality for view in views})
    categories = {row["category"] for row in findings}
    negative_passed = (
        partition == "negative_control" and not missing and categories == corruptions
    )
    return {
        "schema_version": 1,
        "partition": partition,
        "valid_for_training": partition != "negative_control"
        and not findings
        and not missing,
        "negative_control_passed": negative_passed,
        "negative_control_name": negative_control_name,
        "missing_modalities": missing,
        "artifact_count": len(views),
        "fact_count": len(truth),
        "mismatch_count": len(findings),
        "findings": sorted(
            findings,
            key=lambda row: (
                row["artifact_reference"],
                row["fact_reference"],
                row["category"],
            ),
        ),
    }
