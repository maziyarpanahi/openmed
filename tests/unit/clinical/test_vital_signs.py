"""Tests for deterministic vital-sign structuring helpers."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    VITAL_SIGNS_ADVISORY,
    empty_vital_sign_result,
    structure_vital_sign,
)


def test_structure_blood_pressure_components():
    assert structure_vital_sign("120/80 mmHg") == {
        "kind": "blood_pressure",
        "value": None,
        "unit": "mmHg",
        "components": [
            {"kind": "systolic", "value": 120, "unit": "mmHg"},
            {"kind": "diastolic", "value": 80, "unit": "mmHg"},
        ],
    }


def test_structure_oxygen_saturation_with_trailing_context():
    assert structure_vital_sign("SpO2 96% on RA") == {
        "kind": "oxygen_saturation",
        "value": 96,
        "unit": "%",
        "components": [],
    }


@pytest.mark.parametrize(
    ("text", "kind", "value", "unit"),
    [
        ("HR 88", "heart_rate", 88, ""),
        ("heart rate: 88 bpm", "heart_rate", 88, "bpm"),
        ("Temp 37.2 C", "body_temperature", 37.2, "C"),
        ("temperature was 98.6 F", "body_temperature", 98.6, "F"),
        ("RR 18 /min", "respiratory_rate", 18, "/min"),
        ("respiratory rate: 20 breaths/min", "respiratory_rate", 20, "breaths/min"),
    ],
)
def test_structure_common_vital_phrases(text, kind, value, unit):
    assert structure_vital_sign(text) == {
        "kind": kind,
        "value": value,
        "unit": unit,
        "components": [],
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a vital",
        "glucose 120 mg/dL",
        "blood pressure unavailable",
        None,
        object(),
    ],
)
def test_unparseable_input_returns_explicit_unknown_result(text):
    assert structure_vital_sign(text) == empty_vital_sign_result()


def test_output_is_stable_across_runs():
    first = json.dumps(structure_vital_sign("BP: 120/80 mm Hg"), sort_keys=True)
    second = json.dumps(structure_vital_sign("BP: 120/80 mm Hg"), sort_keys=True)
    assert first == second


def test_temperature_units_are_not_converted_or_normalized():
    fahrenheit = structure_vital_sign("Temp 98.6 F")
    celsius = structure_vital_sign("Temp 37 C")

    assert fahrenheit["value"] == 98.6
    assert fahrenheit["unit"] == "F"
    assert celsius["value"] == 37
    assert celsius["unit"] == "C"


def test_vital_signs_advisory_documents_heuristic_scope():
    assert "heuristic structuring" in VITAL_SIGNS_ADVISORY
    assert "originating device" in VITAL_SIGNS_ADVISORY
