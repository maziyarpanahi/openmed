"""Synthetic offline tests for the discharge-summary profile."""

from __future__ import annotations

import json

from openmed.clinical import (
    DISCHARGE_PROFILE_ADVISORY,
    DischargeSummaryProfile,
    extract_discharge_profile,
)

NOTE = (
    "DISCHARGE SUMMARY\n"
    "Discharge Diagnoses:\n"
    "1. possible synthetic pneumonia\n"
    "2. Hypertension\n"
    "Procedures Performed:\n"
    "- Laparoscopic appendectomy\n"
    "Discharge Medications:\n"
    "- Metformin 500 mg PO BID for 7 days\n"
    "- Aspirin 81 mg daily\n"
    "Follow-Up: Primary care in 7 days.\n"
    "Instructions:\n"
    "- Keep dressing dry.\n"
    "- Return if fever.\n"
)


def test_extracts_section_fields_with_evidence_offsets() -> None:
    result = extract_discharge_profile(NOTE)

    assert isinstance(result, DischargeSummaryProfile)
    assert [item.value for item in result.diagnoses] == [
        "possible synthetic pneumonia",
        "Hypertension",
    ]
    assert [item.value for item in result.procedures] == ["Laparoscopic appendectomy"]
    assert [item.value for item in result.medications] == ["Metformin", "Aspirin"]
    assert [item.value for item in result.follow_up] == ["Primary care in 7 days."]
    assert [item.value for item in result.instructions] == [
        "Keep dressing dry.",
        "Return if fever.",
    ]

    for item in result.items:
        assert NOTE[item.start : item.end] == item.text
        assert item.evidence_span == {"start": item.start, "end": item.end}
        assert item.section

    medication = result.medications[0]
    assert NOTE[medication.medication_span[0] : medication.medication_span[1]] == (
        medication.value
    )
    assert medication.sig["frequency_per_day"] == 2.0
    assert result.field_provenance["diagnoses"][0] == result.diagnoses[0].evidence_span


def test_preserves_local_uncertainty_negation_and_temporality() -> None:
    text = (
        "Discharge Summary\n"
        "Discharge Diagnoses:\n"
        "- possible synthetic infection\n"
        "- no evidence of fracture\n"
        "- history of asthma\n"
        "Follow-Up:\n"
        "- if symptoms return, contact the clinic.\n"
    )

    result = extract_discharge_profile(text)

    possible, absent, historical = result.diagnoses
    assert possible.certainty == "uncertain"
    assert possible.negation == "affirmed"
    assert absent.certainty == "certain"
    assert absent.negation == "negated"
    assert historical.temporality == "historical"
    assert result.follow_up[0].temporality == "hypothetical"
    assert result.follow_up[0].uncertain is True


def test_caller_assertion_is_preserved_without_network_or_inference() -> None:
    value = "synthetic pneumonia"
    text = f"Discharge Diagnoses: {value}\n"
    start = text.index(value)
    spans = [
        {
            "label": "CONDITION",
            "start": start,
            "end": start + len(value),
            "assertion": {
                "certainty": "uncertain",
                "negation": "affirmed",
                "temporality": "recent",
            },
        }
    ]

    result = extract_discharge_profile(text, spans=spans)
    item = result.diagnoses[0]

    assert item.value == value
    assert item.assertion.to_dict() == {
        "temporality": "recent",
        "certainty": "uncertain",
        "negation": "affirmed",
    }
    assert "recommendation" not in item.text.casefold()
    assert result.advisory == DISCHARGE_PROFILE_ADVISORY


def test_profile_serialization_is_deterministic_and_json_ready() -> None:
    first = extract_discharge_profile(NOTE)
    second = extract_discharge_profile(NOTE)

    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["schema_version"] == "openmed.clinical.discharge_profile.v1"
    assert payload["field_provenance"] == first.field_provenance
    assert payload["sections"]
