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


# --- Unicode whitespace in copied clinical text (#3106) ---

UNICODE_SPACES = {
    "no-break-space": "\u00a0",
    "narrow-no-break-space": "\u202f",
    "thin-space": "\u2009",
    "em-space": "\u2003",
    "ideographic-space": "\u3000",
}

# Each ASCII phrase uses ordinary spaces; every space in it is swapped for one
# Unicode space character below, including spaces inside multi-word labels and
# units, which is where copied text used to break parsing.
ASCII_VITAL_PHRASES = [
    pytest.param("BP 120/80 mmHg", id="blood-pressure"),
    pytest.param("BP 120/80 mm Hg", id="blood-pressure-spaced-unit"),
    pytest.param("blood pressure 118 / 76 mm Hg", id="blood-pressure-label"),
    pytest.param("Temp 37.2 C", id="temperature"),
    pytest.param("temperature was 98.6 F", id="temperature-label"),
    pytest.param("SpO2 96 %", id="oxygen-saturation"),
    pytest.param("oxygen saturation 96%", id="oxygen-saturation-label"),
    pytest.param("O2 sat 95% on RA", id="oxygen-saturation-short-label"),
    pytest.param("RR 18 /min", id="respiratory-rate"),
    pytest.param("respiratory rate: 20 breaths/min", id="respiratory-rate-label"),
    pytest.param("RR 16 breaths per min", id="respiratory-rate-spaced-unit"),
    pytest.param("HR 88 bpm", id="heart-rate"),
    pytest.param("heart rate: 72 beats per minute", id="heart-rate-label"),
]


def test_ascii_vital_phrases_parse_to_a_known_kind():
    # Guards the fixture itself: an ASCII phrase that no longer parses would make
    # the Unicode comparison below pass trivially against "unknown".
    for param in ASCII_VITAL_PHRASES:
        (text,) = param.values
        assert structure_vital_sign(text)["kind"] != "unknown", text


@pytest.mark.parametrize("ascii_text", ASCII_VITAL_PHRASES)
@pytest.mark.parametrize(
    "space", list(UNICODE_SPACES.values()), ids=list(UNICODE_SPACES)
)
def test_unicode_whitespace_preserves_ascii_values_and_units(ascii_text, space):
    expected = structure_vital_sign(ascii_text)

    assert structure_vital_sign(ascii_text.replace(" ", space)) == expected


@pytest.mark.parametrize(
    "space", list(UNICODE_SPACES.values()), ids=list(UNICODE_SPACES)
)
def test_surrounding_unicode_whitespace_is_stripped(space):
    assert structure_vital_sign(f"{space}HR 88 bpm{space}") == structure_vital_sign(
        "HR 88 bpm"
    )


def test_unit_with_no_break_space_is_captured_with_an_ascii_space():
    result = structure_vital_sign("BP 120/80 mm\u00a0Hg")

    assert result["unit"] == "mm Hg"
    assert [component["unit"] for component in result["components"]] == [
        "mm Hg",
        "mm Hg",
    ]


def test_unicode_whitespace_output_is_deterministic():
    text = "oxygen\u202fsaturation\u00a096\u2009%"

    first = json.dumps(structure_vital_sign(text), sort_keys=True)
    second = json.dumps(structure_vital_sign(text), sort_keys=True)

    assert (
        first
        == second
        == json.dumps(structure_vital_sign("oxygen saturation 96 %"), sort_keys=True)
    )


def test_whitespace_only_unicode_input_is_unknown():
    assert structure_vital_sign("\u00a0\u202f\u2009") == empty_vital_sign_result()
