"""Synthetic offline tests for clinical document-type classification."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from openmed.clinical.sections import (
    DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE,
    UNKNOWN_DOCUMENT_TYPE,
    classify_document,
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
    assert set(classification) == {"type", "confidence"}


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

    assert classification == {"type": UNKNOWN_DOCUMENT_TYPE, "confidence": 0.0}


def test_classify_document_uses_only_the_first_configured_token_window() -> None:
    text = " ".join(["synthetic"] * 256 + ["DISCHARGE", "SUMMARY"])

    assert classify_document(text) == {
        "type": UNKNOWN_DOCUMENT_TYPE,
        "confidence": 0.0,
    }


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
