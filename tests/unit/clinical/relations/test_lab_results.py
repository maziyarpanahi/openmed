"""Tests for guarded laboratory-result relation candidates."""

import json

from openmed.clinical.relations.guarded_lab_result_candidates import (
    generate_lab_result_candidates,
)


def _span(text: str, value: str, label: str, occurrence: int = 0, **extra):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return {"label": label, "start": start, "end": start + len(value), **extra}


def test_links_complete_lab_evidence_without_serializing_values() -> None:
    text = "Labs: Sodium 140 mmol/L (135-145 mmol/L), serum, today."
    spans = [
        _span(text, "Sodium", "ANALYTE", expected_unit="mmol/L"),
        _span(text, "140", "LAB_VALUE"),
        _span(text, "mmol/L", "UNIT"),
        _span(text, "135-145 mmol/L", "REFERENCE_INTERVAL", unit="mmol/L"),
        _span(text, "serum", "SPECIMEN"),
        _span(text, "today", "OBSERVATION_TIME"),
    ]

    (candidate,) = generate_lab_result_candidates(text, spans)

    assert candidate.unit_status == "compatible"
    assert candidate.canonical_unit == "mol/L"
    assert candidate.evidence_complete is True
    assert candidate.conflict_state == "none"
    payload = json.dumps(candidate.to_dict(), sort_keys=True)
    for raw_value in ("Sodium", "140", "serum", "today"):
        assert raw_value not in payload


def test_rejects_dimensionally_incompatible_units() -> None:
    text = "Sodium 140 mmol/L."
    spans = [
        _span(text, "Sodium", "ANALYTE", expected_unit="mmHg"),
        _span(text, "140", "LAB_VALUE"),
        _span(text, "mmol/L", "UNIT"),
    ]

    assert generate_lab_result_candidates(text, spans) == ()


def test_preserves_competing_analyte_links() -> None:
    text = "Sodium or potassium 140 mmol/L."
    spans = [
        _span(text, "Sodium", "ANALYTE", expected_unit="mmol/L"),
        _span(text, "potassium", "ANALYTE", expected_unit="mmol/L"),
        _span(text, "140", "LAB_VALUE"),
        _span(text, "mmol/L", "UNIT"),
    ]

    candidates = generate_lab_result_candidates(text, spans)

    assert len(candidates) == 2
    assert {candidate.conflict_state for candidate in candidates} == {"competing"}
    assert all(len(candidate.competing_candidate_ids) == 1 for candidate in candidates)
