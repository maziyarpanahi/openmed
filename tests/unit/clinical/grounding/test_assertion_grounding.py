"""Assertion-aware grounding: safety invariants, policy, and the trap gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openmed.clinical.context import ClinicalAssertion
from openmed.clinical.exporters.codeable_concept import GroundedSpan
from openmed.clinical.exporters.fhir.condition import to_condition
from openmed.clinical.grounding import (
    Candidate,
    ground_with_context,
)
from openmed.clinical.grounding.assertion_grounding import (
    GROUNDING_ASSERTION_STATUSES,
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_PRESENT,
    GROUNDING_REFUTED,
    GROUNDING_UNCERTAIN,
    POLICY_DROP,
    POLICY_STATUS,
    POLICY_SUPPRESS,
    assertion_grounding_status,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TRAP_SUITE_PATH = REPO_ROOT / "eval" / "suites" / "negation_grounding_traps.py"


def _load_trap_suite():
    """Load the out-of-package trap suite by file path (repo's script pattern)."""

    module_name = "negation_grounding_traps"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, TRAP_SUITE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


traps = _load_trap_suite()


def _span(text: str, surface: str, *candidates: Candidate) -> GroundedSpan:
    start = text.index(surface)
    return GroundedSpan(
        text=surface,
        start=start,
        end=start + len(surface),
        candidates=candidates
        or (Candidate(system="ICD10CM", code="J18.9", display=surface, score=1.0),),
    )


def _only(text: str, surface: str, *candidates: Candidate):
    return ground_with_context(text, [_span(text, surface, *candidates)])[0]


# --- Acceptance: axes attached and code-level status set -------------------


def test_ground_with_context_attaches_all_four_axes():
    asserted = _only("The patient has acute pneumonia.", "pneumonia")

    assert asserted.assertion.negation == "affirmed"
    assert asserted.assertion.temporality == "recent"
    assert asserted.assertion.certainty == "certain"
    assert asserted.assertion.experiencer == "patient"
    assert asserted.status.status == GROUNDING_PRESENT


def test_status_labels_are_from_the_published_set():
    for text, surface in (
        ("No evidence of pneumonia.", "pneumonia"),
        ("Return if pneumonia develops.", "pneumonia"),
        ("Family history of colon cancer.", "colon cancer"),
        ("History of pneumonia, resolved.", "pneumonia"),
        ("Possible pneumonia.", "pneumonia"),
        ("The patient has acute pneumonia.", "pneumonia"),
    ):
        asserted = _only(text, surface)
        assert asserted.status.status in GROUNDING_ASSERTION_STATUSES


# --- Acceptance: the four safety proofs ------------------------------------


def test_negated_finding_is_never_present():
    asserted = _only("No evidence of pneumonia.", "pneumonia")

    assert asserted.status.status == GROUNDING_REFUTED
    assert asserted.status.verification_status == "refuted"
    assert not asserted.status.is_present
    assert not asserted.is_active_patient_condition


def test_hypothetical_finding_is_never_present():
    asserted = _only(
        "Start heparin if pulmonary embolism is suspected.", "pulmonary embolism"
    )

    assert asserted.status.status == GROUNDING_HYPOTHETICAL
    assert not asserted.is_active_patient_condition


def test_family_history_is_not_coded_to_the_patient():
    asserted = _only("Family history of colon cancer.", "colon cancer")

    assert asserted.status.status == GROUNDING_NON_PATIENT
    assert not asserted.status.patient_subject
    assert not asserted.is_active_patient_condition
    assert to_condition(asserted, subject_reference="Patient/1") is None


def test_affirmed_finding_still_grounds_normally():
    asserted = _only(
        "The patient has acute myocardial infarction.", "myocardial infarction"
    )

    assert asserted.status.status == GROUNDING_PRESENT
    assert asserted.is_active_patient_condition
    condition = to_condition(asserted, subject_reference="Patient/1")
    assert condition is not None
    assert condition["clinicalStatus"]["coding"][0]["code"] == "active"
    assert condition["verificationStatus"]["coding"][0]["code"] == "confirmed"
    assert condition["code"]["coding"][0]["code"] == "J18.9"


# --- Historical / uncertain mapping ----------------------------------------


def test_historical_maps_to_inactive():
    asserted = _only("History of pneumonia, now resolved.", "pneumonia")

    assert asserted.status.status == GROUNDING_HISTORICAL
    assert asserted.status.clinical_status == "inactive"
    assert not asserted.is_active_patient_condition


def test_uncertain_maps_to_provisional():
    asserted = _only("Possible pneumonia pending imaging.", "pneumonia")

    assert asserted.status.status == GROUNDING_UNCERTAIN
    assert asserted.status.verification_status == "provisional"


# --- Composite / post-coordinated inheritance ------------------------------


def test_composite_candidates_inherit_the_focus_status():
    focus = Candidate(system="ICD10CM", code="J18.9", display="pneumonia", score=1.0)
    qualifier = Candidate(system="SNOMED", code="263654008", display="acute", score=0.9)
    asserted = _only("No evidence of pneumonia.", "pneumonia", focus, qualifier)

    statuses = [status for _candidate, status in asserted.iter_coded_concepts()]
    assert len(statuses) == 2
    assert all(status.status == GROUNDING_REFUTED for status in statuses)
    assert not any(status.is_active_patient_condition for status in statuses)


# --- Policy invariants ------------------------------------------------------


def test_policy_status_keeps_every_span_with_status():
    text = "No evidence of pneumonia. The patient has acute diabetes mellitus."
    spans = [_span(text, "pneumonia"), _span(text, "diabetes mellitus")]
    results = ground_with_context(text, spans, policy=POLICY_STATUS)

    assert len(results) == 2
    assert results[0].status.status == GROUNDING_REFUTED
    assert results[0].grounded.candidates  # candidates retained under status
    assert results[1].status.status == GROUNDING_PRESENT


def test_policy_drop_removes_non_present_spans():
    text = "No evidence of pneumonia. The patient has acute diabetes mellitus."
    spans = [_span(text, "pneumonia"), _span(text, "diabetes mellitus")]
    results = ground_with_context(text, spans, policy=POLICY_DROP)

    assert [r.status.status for r in results] == [GROUNDING_PRESENT]
    # Invariant: nothing that survives drop is anything but an active patient code.
    assert all(r.is_active_patient_condition for r in results)


def test_policy_suppress_strips_candidates_but_keeps_the_span():
    asserted = ground_with_context(
        "No evidence of pneumonia.",
        [_span("No evidence of pneumonia.", "pneumonia")],
        policy=POLICY_SUPPRESS,
    )[0]

    assert asserted.status.status == GROUNDING_REFUTED
    assert asserted.grounded.candidates == ()


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        ground_with_context("x", [], policy="bogus")


# --- Provenance carries no raw note text (no PHI) --------------------------


def test_provenance_records_offsets_and_labels_without_note_text():
    text = "No evidence of pneumonia."
    asserted = _only(text, "pneumonia")
    provenance = asserted.provenance

    assert provenance["status"] == GROUNDING_REFUTED
    assert provenance["axes"]["negation"] == "negated"
    assert {"category": "negation", "start": 0, "end": 14} in provenance["cues"]
    assert provenance["content_hash"].startswith("sha256:")

    serialized = json.dumps(provenance)
    assert "pneumonia" not in serialized
    assert "No evidence of" not in serialized


# --- The curated trap suite is the hard release gate -----------------------


def test_trap_suite_has_no_active_patient_condition_leaks():
    leaks = traps.active_patient_condition_leaks()
    assert leaks == [], [case.text for case in leaks]


def test_trap_suite_status_mapping_accuracy_at_least_090():
    assert traps.status_mapping_accuracy() >= 0.90


def test_trap_gold_covers_every_status():
    covered = {case.expected_status for case in traps.TRAP_CASES}
    assert covered == set(GROUNDING_ASSERTION_STATUSES)


def test_non_canonical_axis_labels_fail_closed():
    # A foreign producer (e.g. an ML tagger) emitting a non-canonical axis label
    # must fail CLOSED — never a confirmed, active, patient-coded condition.
    neg = assertion_grounding_status(
        ClinicalAssertion(temporality="recent", certainty="certain", negation="neg")
    )
    assert neg.status != GROUNDING_PRESENT
    assert not neg.is_active_patient_condition

    temp = assertion_grounding_status(
        ClinicalAssertion(temporality="whenever", certainty="certain")
    )
    assert temp.status != GROUNDING_PRESENT
    assert not temp.is_active_patient_condition

    # An unknown certainty is withheld from "present" (provisional, not confirmed).
    cert = assertion_grounding_status(
        ClinicalAssertion(temporality="recent", certainty="maybe")
    )
    assert cert.status != GROUNDING_PRESENT
    assert cert.verification_status != "confirmed"

    # An unset negation (None) remains present-compatible — no false refutation.
    affirmed = assertion_grounding_status(
        ClinicalAssertion(temporality="recent", certainty="certain")
    )
    assert affirmed.status == GROUNDING_PRESENT
    assert affirmed.is_active_patient_condition
