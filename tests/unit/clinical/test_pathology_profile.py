"""Tests for the deterministic pathology result extraction profile."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    PATHOLOGY_RESULT_ADVISORY,
    extract_pathology_profile,
    extract_pathology_result,
)


def _assert_field_spans_index_original(text: str, fields: list[dict]) -> None:
    for field in fields:
        span = field["span"]
        assert text[span["start"] : span["end"]] == field["value"]


def test_extracts_labelled_fields_and_reported_biomarkers_with_spans() -> None:
    text = (
        "Patient: Synthetic Subject 01\n"
        "Accession: SYN-001\n"
        "SPECIMEN: Left breast core biopsy\n"
        "FINAL DIAGNOSIS: Invasive ductal carcinoma\n"
        "HISTOLOGIC GRADE: 2 of 3\n"
        "BIOMARKERS:\n"
        "ER: positive\n"
        "HER2 (IHC): 0 (negative)\n"
    )

    result = extract_pathology_result(text)

    assert [item["value"] for item in result["specimen"]] == ["Left breast core biopsy"]
    assert [item["value"] for item in result["diagnosis"]] == [
        "Invasive ductal carcinoma"
    ]
    assert [item["value"] for item in result["grade"]] == ["2 of 3"]
    assert [(item["name"], item["result"]) for item in result["biomarkers"]] == [
        ("ER", "positive"),
        ("HER2 (IHC)", "0 (negative)"),
    ]

    _assert_field_spans_index_original(text, result["specimen"])
    _assert_field_spans_index_original(text, result["diagnosis"])
    _assert_field_spans_index_original(text, result["grade"])
    for biomarker in result["biomarkers"]:
        assert (
            text[biomarker["name_span"]["start"] : biomarker["name_span"]["end"]]
            == biomarker["name"]
        )
        assert (
            text[biomarker["result_span"]["start"] : biomarker["result_span"]["end"]]
            == biomarker["result"]
        )


def test_section_forms_keep_multiple_values_and_do_not_store_source_text() -> None:
    text = (
        "PATIENT: Synthetic Subject 02\n"
        "SPECIMEN\n"
        "A. Skin punch biopsy\n"
        "B. Margin fragment\n"
        "Patient: Synthetic Subject 02\n"
        "FINAL DIAGNOSIS\n"
        "A. Melanocytic lesion\n"
        "B. Margin is free of lesion\n"
        "IMMUNOHISTOCHEMISTRY\n"
        "PD-L1 (22C3): CPS 10\n"
        "BRAF V600E detected\n"
    )

    result = extract_pathology_profile(text)

    assert [item["value"] for item in result["specimen"]] == [
        "Skin punch biopsy",
        "Margin fragment",
    ]
    assert [item["value"] for item in result["diagnosis"]] == [
        "Melanocytic lesion",
        "Margin is free of lesion",
    ]
    assert [(item["name"], item["result"]) for item in result["biomarkers"]] == [
        ("PD-L1 (22C3)", "CPS 10"),
        ("BRAF V600E", "detected"),
    ]
    serialized = json.dumps(result, sort_keys=True)
    assert "Synthetic Subject 02" not in serialized
    assert "PATIENT" not in serialized
    assert "source" not in result


def test_grade_is_captured_only_when_explicitly_reported() -> None:
    assert (
        extract_pathology_result(
            "FINAL DIAGNOSIS: Invasive carcinoma. No grade is reported."
        )["grade"]
        == []
    )

    result = extract_pathology_result(
        "FINAL DIAGNOSIS: Invasive carcinoma, histologic grade 3."
    )
    assert [item["value"] for item in result["grade"]] == ["3"]

    gleason = extract_pathology_result("Gleason score 3+4=7.")
    assert [item["value"] for item in gleason["grade"]] == ["3+4=7"]


def test_unknown_metadata_and_unstructured_prose_are_not_promoted_to_fields() -> None:
    text = (
        "Patient: Synthetic Subject 03\n"
        "Accession: SYN-003\n"
        "Clinical history: Screening examination.\n"
        "The diagnosis was discussed with the care team.\n"
        "No biomarker result is stated.\n"
    )

    result = extract_pathology_result(text)

    assert result["specimen"] == []
    assert result["diagnosis"] == []
    assert result["grade"] == []
    assert result["biomarkers"] == []


def test_parser_is_deterministic_offline_and_advisory_is_review_only() -> None:
    text = "SPECIMEN: Synthetic tissue\nDIAGNOSIS: Benign lesion\nER: positive"
    first = extract_pathology_result(text)
    second = extract_pathology_result(text)

    assert first == second
    assert first["advisory"] == PATHOLOGY_RESULT_ADVISORY
    assert "not a diagnostic decision engine" in first["advisory"]
    assert "network" not in json.dumps(first).casefold()


def test_non_string_input_fails_without_echoing_sensitive_values() -> None:
    with pytest.raises(TypeError, match="must be a string") as error:
        extract_pathology_result({"patient": "Synthetic Subject 04"})  # type: ignore[arg-type]
    assert "Synthetic Subject 04" not in str(error.value)
