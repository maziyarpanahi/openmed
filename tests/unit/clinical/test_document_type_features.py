"""Synthetic offline tests for document-type feature extraction."""

from __future__ import annotations

from collections import Counter

import pytest

from openmed.clinical.sections import (
    DOCUMENT_TYPE_CONFIDENCE_THRESHOLD,
    DOCUMENT_TYPES,
    GENERIC_DOCUMENT_TYPE,
    LOINC_DOCUMENT_ONTOLOGY_AXES,
    LOINC_DOCUMENT_TYPE_HINTS,
    UNKNOWN_DOCUMENT_TYPE,
    classify_document,
    extract_doctype_features,
)

SYNTHETIC_LABELED_NOTES = (
    ("progress_note", "PROGRESS NOTE\nSubjective: synthetic update.\nPlan: observe."),
    (
        "progress_note",
        "PROGRESS NOTE\nObjective: synthetic value.\nAssessment: stable.",
    ),
    (
        "progress_note",
        "DAILY PROGRESS NOTE\nInterval history: unchanged.\nPlan: review.",
    ),
    (
        "progress_note",
        "PROGRESS NOTE\nHPI: synthetic symptom.\nAssessment and Plan: continue.",
    ),
    (
        "progress_note",
        "PROGRESS NOTE\nSubjective: improved.\nObjective: calm.\nPlan: follow up.",
    ),
    (
        "discharge_summary",
        "DISCHARGE SUMMARY\nHospital course: synthetic and stable.\nDisposition: home.",
    ),
    (
        "discharge_summary",
        "DISCHARGE SUMMARY\nDischarge diagnoses: synthetic condition.\nFollow-up instructions: review.",
    ),
    (
        "discharge_summary",
        "HOSPITAL DISCHARGE SUMMARY\nHospital course: routine.\nDisposition: home.",
    ),
    (
        "discharge_summary",
        "DISCHARGE SUMMARY\nHospital course: completed.\nDischarge diagnoses: none.",
    ),
    (
        "discharge_summary",
        "DISCHARGE SUMMARY\nDisposition: clinic.\nFollow-up instructions: synthetic plan.",
    ),
    (
        "radiology_report",
        "RADIOLOGY REPORT\nFindings: synthetic image.\nImpression: no acute finding.",
    ),
    (
        "radiology_report",
        "IMAGING REPORT\nTechnique: synthetic sequence.\nImpression: stable.",
    ),
    (
        "radiology_report",
        "X-RAY REPORT\nComparison: synthetic prior.\nFindings: clear.",
    ),
    (
        "radiology_report",
        "CT REPORT\nTechnique: synthetic protocol.\nImpression: unchanged.",
    ),
    (
        "radiology_report",
        "RADIOLOGY REPORT\nFindings: synthetic observation.\nImpression: benign.",
    ),
    (
        "pathology_report",
        "PATHOLOGY REPORT\nSpecimen: synthetic sample.\nFinal diagnosis: benign.",
    ),
    (
        "pathology_report",
        "SURGICAL PATHOLOGY\nGross description: synthetic fragment.\nFinal diagnosis: clear.",
    ),
    (
        "pathology_report",
        "HISTOPATHOLOGY REPORT\nSpecimen: synthetic tissue.\nMicroscopic description: bland.",
    ),
    (
        "pathology_report",
        "PATHOLOGY REPORT\nGross description: synthetic material.\nMicroscopic description: benign.",
    ),
    (
        "pathology_report",
        "PATHOLOGY REPORT\nSpecimen: synthetic biopsy.\nFinal diagnosis: negative.",
    ),
    (
        "consult_note",
        "CONSULTATION NOTE\nReason for consultation: synthetic question.\nRecommendations: review.",
    ),
    (
        "consult_note",
        "CONSULT NOTE\nConsulting service: synthetic team.\nRecommendations: observe.",
    ),
    (
        "consult_note",
        "SPECIALIST CONSULTATION\nReferring provider: synthetic service.\nClinical opinion: stable.",
    ),
    (
        "consult_note",
        "CONSULTATION NOTE\nReason for consultation: synthetic symptom.\nPlan: follow up.",
    ),
    (
        "consult_note",
        "CONSULT NOTE\nConsulting service: synthetic team.\nClinical opinion: no concern.",
    ),
    (
        "operative_note",
        "OPERATIVE NOTE\nPreoperative diagnosis: synthetic lesion.\nProcedure performed: excision.",
    ),
    (
        "operative_note",
        "OPERATION NOTE\nPostoperative diagnosis: synthetic finding.\nEstimated blood loss: minimal.",
    ),
    (
        "operative_note",
        "SURGICAL OPERATIVE NOTE\nProcedure performed: synthetic repair.\nAnesthesia: general.",
    ),
    (
        "operative_note",
        "OPERATIVE NOTE\nPreoperative diagnosis: synthetic condition.\nPostoperative diagnosis: resolved.",
    ),
    (
        "operative_note",
        "OPERATIVE NOTE\nProcedure performed: synthetic procedure.\nEstimated blood loss: none.",
    ),
)


def test_document_type_labels_and_loinc_hints_cover_the_public_set() -> None:
    assert set(DOCUMENT_TYPES) == {
        "progress_note",
        "discharge_summary",
        "radiology_report",
        "pathology_report",
        "consult_note",
        "operative_note",
    }
    assert set(LOINC_DOCUMENT_TYPE_HINTS) == set(DOCUMENT_TYPES)
    assert LOINC_DOCUMENT_ONTOLOGY_AXES == ("kind-of-document", "setting")
    assert all(LOINC_DOCUMENT_TYPE_HINTS[label] for label in DOCUMENT_TYPES)


def test_extract_doctype_features_is_deterministic_and_uses_sections() -> None:
    text = (
        "PROGRESS NOTE\n"
        "HPI: synthetic cough.\n"
        "ASSESSMENT/PLAN: continue observation.\n"
        "MEDICATIONS: synthetic tablet."
    )

    features = extract_doctype_features(text)

    assert features == extract_doctype_features(text)
    assert features["header_hits"]["progress_note"] == 1
    assert features["keyword_cues"]["progress_note"] >= 2
    assert features["loinc_hints"]["progress_note"] >= 1
    assert features["section_histogram"]["history_of_present_illness"] == 1
    assert features["section_histogram"]["assessment_and_plan"] == 1
    assert features["section_histogram"]["medications"] == 1
    assert features["token_count"] > 0


def test_extract_doctype_features_bounds_header_and_keyword_window() -> None:
    text = " ".join(["synthetic"] * 8 + ["DISCHARGE", "SUMMARY"])

    features = extract_doctype_features(text, max_tokens=8)

    assert features["max_tokens"] == 8
    assert features["token_count"] == 8
    assert all(value == 0 for value in features["header_hits"].values())
    assert all(value == 0 for value in features["keyword_cues"].values())


def test_synthetic_labeled_notes_reach_top_one_accuracy_threshold() -> None:
    predictions = [
        classify_document(text)["type"] for _, text in SYNTHETIC_LABELED_NOTES
    ]
    correct = sum(
        predicted == expected
        for predicted, (expected, _) in zip(predictions, SYNTHETIC_LABELED_NOTES)
    )

    assert len(SYNTHETIC_LABELED_NOTES) >= len(DOCUMENT_TYPES) * 5
    assert correct / len(SYNTHETIC_LABELED_NOTES) >= 0.85
    assert Counter(predictions).keys() >= set(DOCUMENT_TYPES)


def test_low_confidence_notes_abstain_at_the_documented_threshold() -> None:
    classification = classify_document(
        "Synthetic narrative with one uninformative finding."
    )

    assert DOCUMENT_TYPE_CONFIDENCE_THRESHOLD == 0.5
    assert GENERIC_DOCUMENT_TYPE == UNKNOWN_DOCUMENT_TYPE
    assert classification == {
        "type": UNKNOWN_DOCUMENT_TYPE,
        "loinc_code": None,
        "loinc_axes": None,
        "confidence": 0.0,
    }


@pytest.mark.parametrize("expected_type", DOCUMENT_TYPES)
def test_each_document_type_has_multiple_synthetic_gold_stubs(
    expected_type: str,
) -> None:
    assert sum(label == expected_type for label, _ in SYNTHETIC_LABELED_NOTES) >= 5
