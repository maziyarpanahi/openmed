"""Tests for UCUM-subset measurement normalization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openmed.clinical import (
    MEASUREMENT_NORMALIZATION_ADVISORY,
    ROUND_TRIP_ABS_TOLERANCE,
    ROUND_TRIP_REL_TOLERANCE,
    derive_abnormal_flag,
    normalize_to,
    normalize_vital_measurement,
    parse_measurement,
    parse_reference_range,
    parse_unit,
)


def test_parse_unit_records_alias_provenance():
    parsed = parse_unit("mg per dL")

    assert parsed["status"] == "ok"
    assert parsed["unit"] == "mg/dL"
    assert parsed["canonical_unit"] == "g/L"
    assert parsed["dimension"] == {"mass": 1, "volume": -1}
    assert parsed["provenance"]["source"] == "alias_table"


def test_parse_measurement_returns_canonical_magnitude_and_advisory():
    parsed = parse_measurement("5 mg/dL")

    assert parsed["status"] == "ok"
    assert parsed["magnitude"] == pytest.approx(0.05)
    assert parsed["canonical_unit"] == "g/L"
    assert parsed["dimension"] == {"mass": 1, "volume": -1}
    assert parsed["advisory"] == MEASUREMENT_NORMALIZATION_ADVISORY
    assert "originating laboratory" in parsed["advisory"]
    assert parsed["provenance"]["unit"]["normalized_unit"] == "mg/dL"


def test_normalize_to_converts_commensurable_units():
    converted = normalize_to(1.2, "g/L", "mg/dL")

    assert converted["status"] == "ok"
    assert converted["magnitude"] == pytest.approx(120)
    assert converted["canonical_magnitude"] == pytest.approx(1.2)
    assert converted["unit"] == "mg/dL"
    assert converted["provenance"]["conversion"] == "dimension_checked_ucum_subset"


@pytest.mark.parametrize(
    ("value", "source_unit", "target_unit"),
    [
        (1.4, "g/L", "mg/dL"),
        (25, "mg/L", "mg/dL"),
        (4.5, "10*3/uL", "10*9/L"),
        (98.6, "F", "C"),
        (120, "mmHg", "kPa"),
        (96, "%", "1"),
    ],
)
def test_round_trip_conversions_are_stable(
    value,
    source_unit,
    target_unit,
):
    forward = normalize_to(value, source_unit, target_unit)
    assert forward["status"] == "ok"

    backward = normalize_to(forward["magnitude"], target_unit, source_unit)
    assert backward["status"] == "ok"
    assert math.isclose(
        backward["magnitude"],
        value,
        rel_tol=ROUND_TRIP_REL_TOLERANCE,
        abs_tol=ROUND_TRIP_ABS_TOLERANCE,
    )


def test_incommensurable_units_refuse_without_fabricating_number():
    converted = normalize_to(5, "mg/dL", "mmHg")

    assert converted["status"] == "incommensurable"
    assert converted["magnitude"] is None
    assert converted["canonical_magnitude"] is None
    assert "dimensionally commensurable" in converted["reason"]


def test_analyte_dependent_conversions_refuse_without_fabricating_number():
    converted = normalize_to(5, "mg/dL", "mmol/L")

    assert converted["status"] == "analyte_context_required"
    assert converted["magnitude"] is None
    assert converted["canonical_magnitude"] is None
    assert "analyte-specific" in converted["reason"]


def test_ambiguous_unit_strings_return_ambiguous_status():
    parsed = parse_measurement(10, "units")

    assert parsed["status"] == "ambiguous"
    assert parsed["magnitude"] is None
    assert "ambiguous" in parsed["reason"]


def test_lab_flag_converts_value_and_range_to_canonical_units():
    assert (
        derive_abnormal_flag(
            1.4,
            {"low": 70, "high": 110, "unit": "mg/dL"},
            value_unit="g/L",
        )
        == "high"
    )
    assert derive_abnormal_flag(13.5, "120-160 g/L", value_unit="g/dL") == "normal"
    assert derive_abnormal_flag("1.4 g/L", "70-110 mg/dL") == "high"


def test_lab_flag_returns_unknown_for_analyte_dependent_range_units():
    assert (
        derive_abnormal_flag(
            0.8,
            {"low": 70, "high": 110, "unit": "umol/L"},
            value_unit="mg/dL",
        )
        == "unknown"
    )


def test_inclusive_and_exclusive_bounds_survive_unit_conversion():
    assert derive_abnormal_flag(0.05, "<5 mg/dL", value_unit="g/L") == "high"
    assert derive_abnormal_flag(0.05, "<=5 mg/dL", value_unit="g/L") == "normal"
    assert derive_abnormal_flag(0.1, ">10 mg/dL", value_unit="g/L") == "low"
    assert derive_abnormal_flag(0.101, ">10 mg/dL", value_unit="g/L") == "normal"


def test_parse_reference_range_preserves_trailing_unit():
    parsed = parse_reference_range("70-99 mg/dL")

    assert parsed == {
        "low": 70.0,
        "high": 99.0,
        "low_inclusive": True,
        "high_inclusive": True,
        "unit": "mg/dL",
    }


def test_vital_measurement_normalization_is_explicit():
    converted = normalize_vital_measurement(98.6, "F", "C")

    assert converted["status"] == "ok"
    assert converted["magnitude"] == pytest.approx(37.0)
    assert converted["provenance"]["target_unit"]["normalized_unit"] == "Cel"


def test_synthetic_gold_corpus_cross_unit_accuracy_gate():
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "clinical"
        / "ucum_measurement_gold.jsonl"
    )
    records = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct = 0
    for record in records:
        predicted = derive_abnormal_flag(
            record["value"],
            record["range"],
            value_unit=record["value_unit"],
        )
        correct += int(predicted == record["expected"])

    accuracy = correct / len(records)
    assert accuracy >= 0.95
