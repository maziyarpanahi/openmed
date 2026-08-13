"""Synthetic offline tests for the structured discharge-summary extractor."""

from __future__ import annotations

from openmed.clinical import (
    DISCHARGE_SUMMARY_ADVISORY,
    DischargeSummary,
    discharge_summary_field_f1,
    extract_discharge_summary,
)

NOTE = (
    "DISCHARGE SUMMARY\n"
    "Discharge Diagnoses:\n"
    "1. Synthetic pneumonia\n"
    "2. Hypertension\n"
    "Procedures Performed:\n"
    "- Laparoscopic appendectomy\n"
    "Discharge Medications:\n"
    "- Metformin 500 mg PO BID for 7 days\n"
    "- Aspirin 81 mg daily\n"
    "Follow-Up:\n"
    "- Primary care in 7 days.\n"
)


def test_extracts_typed_fields_and_round_trips_source_provenance() -> None:
    result = extract_discharge_summary(NOTE)

    assert isinstance(result, DischargeSummary)
    assert result.document_type == "discharge_summary"
    assert [item.text for item in result.discharge_diagnoses] == [
        "Synthetic pneumonia",
        "Hypertension",
    ]
    assert [item.text for item in result.procedures] == ["Laparoscopic appendectomy"]
    assert [item.medication for item in result.discharge_medications] == [
        "Metformin",
        "Aspirin",
    ]
    assert result.discharge_medications[0].sig["frequency_per_day"] == 2.0
    assert result.discharge_medications[0].sig["duration_days"] == 7
    assert [item.text for item in result.follow_up] == ["Primary care in 7 days."]

    for field in (
        result.discharge_diagnoses,
        result.procedures,
        result.discharge_medications,
        result.follow_up,
    ):
        for item in field:
            assert NOTE[item.start : item.end] == item.text
            assert item.provenance["start"] == item.start
            assert item.provenance["end"] == item.end

    assert result.field_provenance["discharge_medications"][0][
        "medication_start"
    ] == NOTE.index("Metformin")
    assert "review aid" in result.advisory
    assert result.advisory == DISCHARGE_SUMMARY_ADVISORY
    assert result.summary_card.to_dict()["entity_counts"] == {
        "problems": 2,
        "medications": 2,
        "labs": 0,
        "procedures": 1,
        "other": 0,
    }


def test_discharge_medications_are_not_taken_from_current_medications() -> None:
    text = (
        "DISCHARGE SUMMARY\n"
        "Current Medications:\n"
        "- Home tablet 10 mg daily\n"
        "Discharge Medications:\n"
        "- New tablet 20 mg daily\n"
        "Follow-up: Primary care in one week."
    )

    result = extract_discharge_summary(text)

    assert [item.medication for item in result.discharge_medications] == ["New tablet"]
    assert "Home tablet" not in result.discharge_medications[0].text


def test_existing_spans_are_reconciled_and_grounding_is_preserved() -> None:
    diagnosis = "Synthetic pneumonia"
    medication = "Metformin"
    text = (
        "DISCHARGE SUMMARY\n"
        f"Discharge Diagnoses: {diagnosis}\n"
        f"Repeated diagnosis: {diagnosis}\n"
        f"Discharge Medications: {medication} 500 mg PO BID\n"
    )
    first_diagnosis = text.index(diagnosis)
    second_diagnosis = text.index(diagnosis, first_diagnosis + 1)
    medication_start = text.index(medication)
    spans = [
        {
            "label": "CONDITION",
            "start": first_diagnosis,
            "end": first_diagnosis + len(diagnosis),
        },
        {
            "label": "CONDITION",
            "start": second_diagnosis,
            "end": second_diagnosis + len(diagnosis),
        },
        {
            "label": "MEDICATION",
            "start": medication_start,
            "end": medication_start + len(medication),
        },
    ]

    result = extract_discharge_summary(text, spans=spans)

    assert [item.text for item in result.discharge_diagnoses] == [diagnosis]
    assert len(result.discharge_medications) == 1
    assert result.discharge_medications[0].medication == medication
    assert result.discharge_medications[0].reconciled is True


def test_fhir_export_contains_four_reviewable_resource_shapes() -> None:
    result = extract_discharge_summary(NOTE)
    resources = result.to_fhir(
        subject_reference="Patient/synthetic",
        document_id="synthetic-discharge",
        bundle=False,
    )

    assert isinstance(resources, list)
    assert [resource["resourceType"] for resource in resources] == [
        "Condition",
        "Condition",
        "Procedure",
        "MedicationStatement",
        "MedicationStatement",
        "CarePlan",
    ]
    medication = resources[3]
    assert medication["subject"] == {"reference": "Patient/synthetic"}
    assert medication["dosage"][0]["timing"]["repeat"]["frequency"] == 2
    assert medication["note"] == [{"text": DISCHARGE_SUMMARY_ADVISORY}]
    follow_up = resources[-1]
    assert follow_up["description"] == "Primary care in 7 days."
    assert follow_up["activity"][0]["detail"]["description"] == follow_up["description"]


def test_synthetic_gold_field_f1_meets_acceptance_thresholds() -> None:
    gold = {
        "discharge_diagnoses": ["Synthetic pneumonia", "Hypertension"],
        "procedures": ["Laparoscopic appendectomy"],
        "discharge_medications": ["Metformin", "Aspirin"],
        "follow_up": ["Primary care in 7 days."],
    }

    scores = discharge_summary_field_f1(extract_discharge_summary(NOTE), gold)

    assert scores["discharge_diagnoses"] >= 0.80
    assert scores["procedures"] >= 0.80
    assert scores["discharge_medications"] >= 0.80
    assert scores["follow_up"] >= 0.75
