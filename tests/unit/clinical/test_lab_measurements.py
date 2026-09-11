"""Focused tests for deterministic synthetic lab measurement normalization."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    LAB_MEASUREMENT_ADVISORY,
    normalize_lab_measurement,
    normalize_lab_measurements,
)


def test_normalizes_value_range_interpretation_and_offsets() -> None:
    record = normalize_lab_measurement(
        {
            "analyte": "Glucose",
            "value": 120,
            "unit": "mg/dL",
            "reference_range": "70-99 mg/dL",
            "flag": "H",
            "qualifiers": ["fasting", " fasting "],
            "start": 18,
            "end": 44,
        }
    )

    assert record["status"] == "ok"
    assert record["value"] == 120.0
    assert record["unit"] == "mg/dL"
    assert record["canonical_value"] == pytest.approx(1.2)
    assert record["canonical_unit"] == "g/L"
    assert record["interpretation"] == "high"
    assert record["qualifiers"] == ["fasting"]
    assert record["source_offsets"] == {"start": 18, "end": 44}
    assert record["reference_range"] == {
        "low": 70.0,
        "high": 99.0,
        "low_inclusive": True,
        "high_inclusive": True,
        "unit": "mg/dL",
        "canonical_low": pytest.approx(0.7),
        "canonical_high": pytest.approx(0.99),
        "canonical_unit": "g/L",
        "status": "ok",
        "unit_status": "known",
    }


def test_embedded_unit_and_unitless_attached_range_are_supported() -> None:
    record = normalize_lab_measurement(
        "4.2 mmol/L",
        reference_range="3.5-5.1",
        source_offsets=(2, 14),
    )

    assert record["status"] == "ok"
    assert record["value"] == pytest.approx(4.2)
    assert record["unit"] == "mmol/L"
    assert record["canonical_unit"] == "mol/L"
    assert record["reference_range"]["unit"] is None
    assert record["reference_range"]["canonical_unit"] == "mol/L"
    assert record["interpretation"] == "normal"
    assert record["source_offsets"] == {"start": 2, "end": 14}


@pytest.mark.parametrize("unit", ["mystery-unit", "units", None])
def test_unknown_or_missing_unit_fails_closed_without_guessing(unit: object) -> None:
    record = normalize_lab_measurement(
        {
            "value": 7.5,
            "unit": unit,
            "reference_range": "4-11",
            "source_offsets": [4, 7],
        }
    )

    assert record["status"] == "unknown_unit"
    assert record["unit_status"] in {"missing", "unknown", "ambiguous"}
    assert record["canonical_value"] is None
    assert record["canonical_unit"] is None
    assert record["interpretation"] == "unknown"


def test_unknown_reference_unit_is_explicit_and_does_not_leak_source_text() -> None:
    record = normalize_lab_measurement(
        {
            "value": 3.2,
            "unit": "mg/dL",
            "reference_range": {"low": 1, "high": 2, "unit": "private-unit"},
            "source_offsets": (9, 22),
        }
    )

    serialized = json.dumps(record, sort_keys=True)

    assert record["status"] == "unknown_unit"
    assert record["reference_range"]["status"] == "unknown_unit"
    assert record["interpretation"] == "unknown"
    assert "input_value" not in serialized
    assert "reference_range_text" not in serialized
    assert "3.2 mg/dL" not in serialized


def test_explicit_flag_is_retained_when_numeric_comparison_is_unavailable() -> None:
    record = normalize_lab_measurement(
        8.0,
        "mystery-unit",
        "1-2 mystery-unit",
        flag="critical",
    )

    assert record["status"] == "unknown_unit"
    assert record["interpretation"] == "critical"
    assert record["provenance"]["explicit_flag_provided"] is True


def test_batch_normalization_preserves_input_order_and_is_deterministic() -> None:
    rows = [
        {"analyte": "Sodium", "value": 140, "unit": "mmol/L"},
        {"analyte": "Potassium", "value": 4.0, "unit": "mmol/L"},
    ]

    first = normalize_lab_measurements(rows)
    second = normalize_lab_measurements(rows)

    assert first == second
    assert [row["analyte"] for row in first] == ["Sodium", "Potassium"]
    assert all(row["advisory"] == LAB_MEASUREMENT_ADVISORY for row in first)
