"""Synthetic offline tests for clinical document-type classification."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from openmed.clinical.sections import (
    DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE,
    DOCUMENT_TYPE_TO_LOINC,
    LOINC_AXIS_NAMES,
    LOINC_DOCUMENT_PROVENANCE,
    LOINC_DOCUMENT_SUBSET,
    LOINC_DOCUMENT_SUBSET_MAX_ROWS,
    UNKNOWN_DOCUMENT_TYPE,
    classify_document,
    document_type_loinc_coverage,
)


@pytest.mark.parametrize(
    ("expected_type", "text"),
    (
        (
            "discharge_summary",
            "DISCHARGE SUMMARY\nHospital course: Stable.\nDischarge diagnoses: Cough.",
        ),
        (
            "progress_note",
            "PROGRESS NOTE\nSubjective: Better.\nAssessment: Stable.\nPlan: Review.",
        ),
        (
            "radiology_report",
            "RADIOLOGY REPORT\nFINDINGS: Clear.\nIMPRESSION: No acute finding.",
        ),
        (
            "pathology_report",
            "PATHOLOGY REPORT\nSPECIMEN: Synthetic sample.\nMICROSCOPIC DESCRIPTION: Benign.",
        ),
        (
            "operative_note",
            "OPERATIVE NOTE\nPreoperative diagnosis: Cyst.\nProcedure performed: Excision.",
        ),
        (
            "history_and_physical",
            "HISTORY AND PHYSICAL\nChief complaint: Cough.\nPhysical examination: Stable.",
        ),
        (
            "consult_note",
            "CONSULTATION NOTE\nReason for consultation: Cough.\nRecommendations: Review.",
        ),
    ),
)
def test_classify_document_covers_canonical_note_types(
    expected_type: str,
    text: str,
) -> None:
    classification = classify_document(text)

    assert classification["type"] == expected_type
    assert 0.5 <= classification["confidence"] <= 1.0
    assert set(classification) == {
        "type",
        "loinc_code",
        "loinc_axes",
        "confidence",
    }
    assert classification["loinc_code"] == DOCUMENT_TYPE_TO_LOINC[expected_type]["code"]
    assert set(classification["loinc_axes"] or {}) == set(LOINC_AXIS_NAMES)


@pytest.mark.parametrize(
    "text",
    (
        "",
        "Brief synthetic clinical narrative without a note-type signature.",
        "RADIOLOGY REPORT\nPATHOLOGY REPORT\nSynthetic mixed template.",
    ),
)
def test_classify_document_abstains_for_unknown_or_ambiguous_notes(text: str) -> None:
    classification = classify_document(text)

    assert classification == {
        "type": UNKNOWN_DOCUMENT_TYPE,
        "loinc_code": None,
        "loinc_axes": None,
        "confidence": 0.0,
    }


def test_classify_document_uses_only_the_first_configured_token_window() -> None:
    text = " ".join(["synthetic"] * 256 + ["DISCHARGE", "SUMMARY"])

    assert classify_document(text) == {
        "type": UNKNOWN_DOCUMENT_TYPE,
        "loinc_code": None,
        "loinc_axes": None,
        "confidence": 0.0,
    }


def test_synthetic_classifier_predictions_have_complete_loinc_coverage() -> None:
    notes = (
        "DISCHARGE SUMMARY\nHospital course: Stable.",
        "RADIOLOGY REPORT\nCT chest findings: Clear.\nIMPRESSION: Stable.",
        "PATHOLOGY REPORT\nSPECIMEN: Synthetic sample.",
        "PROGRESS NOTE\nSubjective: Better.\nPlan: Review.",
        "OPERATIVE NOTE\nProcedure performed: Excision.",
        "HISTORY AND PHYSICAL\nChief complaint: Cough.\nPhysical examination: Stable.",
    )

    classifications = [classify_document(note) for note in notes]
    coverage = document_type_loinc_coverage(classifications)

    assert coverage == {
        "total_predictions": 6,
        "mapped_predictions": 6,
        "unmapped_predictions": 0,
        "mapping_coverage": 1.0,
        "unmapped_types": [],
    }
    assert all(classification["loinc_code"] for classification in classifications)


def test_ct_chest_report_resolves_radiology_and_imaging_axes() -> None:
    classification = classify_document(
        "CT CHEST REPORT\nFindings: Clear.\nImpression: No acute finding."
    )

    assert classification["type"] == "radiology_report"
    assert classification["loinc_code"] == "18748-4"
    assert classification["loinc_axes"] == {
        "type_of_service": "radiology",
        "subject_matter_domain": "imaging",
        "role": "report",
        "setting": "diagnostic",
    }


def test_loinc_subset_is_small_and_license_checked() -> None:
    assert len(LOINC_DOCUMENT_SUBSET) == len(DOCUMENT_TYPE_TO_LOINC)
    assert len(LOINC_DOCUMENT_SUBSET) < LOINC_DOCUMENT_SUBSET_MAX_ROWS
    assert LOINC_DOCUMENT_PROVENANCE["license_checked"] is True
    assert LOINC_DOCUMENT_PROVENANCE["restricted_data"] is False
    assert LOINC_DOCUMENT_PROVENANCE["full_release_bundled"] is False
    assert all(set(row) == {"code", "label"} for row in LOINC_DOCUMENT_SUBSET)


def test_document_type_signatures_are_committed_synthetic_local_data() -> None:
    resource = resources.files("openmed.clinical").joinpath(
        DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["provenance"] == {
        "source": "OpenMed synthetic clinical note-type signatures",
        "license": "Apache-2.0",
        "restricted_data": False,
        "synthetic": True,
    }
    assert len(payload["document_types"]) >= 6
    assert all(entry["rules"] for entry in payload["document_types"])
