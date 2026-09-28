"""Synthetic offline tests for employment, living, and food SDOH cues."""

from __future__ import annotations

import pytest

from openmed.clinical.sdoh import (
    FOOD_INSECURITY_EXTENSION_NOTE,
    available_determinant_extractors,
    extract_sdoh,
    load_sdoh_social_cues,
)
from openmed.clinical.sections import detect_sections


def test_social_extractors_are_registered_without_dispatcher_changes() -> None:
    assert {"employment", "food_insecurity", "living_status"} <= set(
        available_determinant_extractors()
    )


def test_registered_social_extractors_respect_social_history_scope() -> None:
    text = (
        "Assessment: retired teacher.\n"
        "Social History: lives alone.\n"
        "Plan: reports food insecurity."
    )

    findings = extract_sdoh(text, spans=[], sections=detect_sections(text))

    assert [(finding.category, finding.value) for finding in findings] == [
        ("living_status", "lives_alone")
    ]


def test_retired_teacher_emits_employment_status_and_type() -> None:
    text = "retired teacher"

    finding = _only_finding(text, "employment")

    assert finding.status == "retired"
    assert finding.value == "teacher"
    assert finding.temporality == "recent"
    assert finding.span == (0, len(text))


@pytest.mark.parametrize(
    ("text", "expected_status"),
    [
        ("currently employed engineer", "employed"),
        ("unemployed", "unemployed"),
        ("retired", "retired"),
        ("on disability", "disabled"),
        ("student", "student"),
    ],
)
def test_employment_status_cues(text: str, expected_status: str) -> None:
    finding = _only_finding(text, "employment")

    assert finding.status == expected_status


def test_previously_employed_reuses_context_temporality() -> None:
    finding = _only_finding("previously employed", "employment")

    assert finding.status == "former"
    assert finding.value == "former"
    assert finding.temporality == "historical"


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("stable housing", "housed"),
        ("homeless", "homeless"),
        ("lives alone", "lives_alone"),
        ("lives with family", "lives_with_family"),
        ("assisted living", "assisted_living"),
    ],
)
def test_living_status_types(text: str, expected_type: str) -> None:
    finding = _only_finding(text, "living_status")

    assert finding.status == expected_type
    assert finding.value == expected_type


def test_homeless_takes_priority_in_combined_living_status_clause() -> None:
    text = "lives alone, homeless x2 years"

    finding = _only_finding(text, "living_status")

    assert finding.status == "homeless"
    assert finding.value == "homeless"
    assert text[slice(*finding.span)] == "homeless"
    assert finding.temporality == "recent"


@pytest.mark.parametrize(
    "text",
    [
        "reports food insecurity",
        "food insecure",
        "receives SNAP",
        "skips meals",
    ],
)
def test_food_insecurity_cues_emit_extension_finding(text: str) -> None:
    finding = _only_finding(text, "food_insecurity")

    assert finding.status == "current"
    assert finding.value == "food_insecure"
    assert finding.temporality == "recent"


def test_food_cues_document_unrestricted_shac_extension() -> None:
    payload = load_sdoh_social_cues()
    food = payload["determinants"]["food_insecurity"]

    assert payload["provenance"]["restricted_data"] is False
    assert food["extension_beyond_core_shac"] is True
    assert food["extension_note"] == FOOD_INSECURITY_EXTENSION_NOTE


def _only_finding(text: str, category: str):
    findings = [
        finding
        for finding in extract_sdoh(text, spans=[])
        if finding.category == category
    ]
    assert len(findings) == 1
    return findings[0]
