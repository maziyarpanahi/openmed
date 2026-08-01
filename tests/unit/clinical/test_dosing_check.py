"""Tests for advisory medication dosing range checks."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import (
    DOSING_RANGE_CHECK_DISCLAIMER,
    check_dose_ranges,
)


def _fixture_ranges():
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "clinical"
        / "synthetic_dose_ranges.json"
    )
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_over_range_dose_returns_one_review_flag_with_observed_bound_and_disclaimer():
    result = check_dose_ranges(
        [{"drug": "democef", "route": "oral", "amount": 1000, "unit": "mg"}],
        _fixture_ranges(),
    )

    assert result["notes"] == []
    assert len(result["flags"]) == 1
    flag = result["flags"][0]
    assert flag["status"] == "above_range"
    assert flag["observed_value"] == 1000
    assert flag["observed_unit"] == "mg"
    assert flag["reference_bound"] == 500
    assert flag["reference_bound_type"] == "high"
    assert flag["disclaimer"] == DOSING_RANGE_CHECK_DISCLAIMER


def test_in_range_dose_returns_no_flags():
    result = check_dose_ranges(
        [{"drug": "democef", "route": "oral", "amount": 250, "unit": "mg"}],
        _fixture_ranges(),
    )

    assert result == {"flags": [], "notes": []}


def test_incompatible_units_yield_not_checked_note_without_false_flag():
    result = check_dose_ranges(
        [{"drug": "demodrip", "route": "iv", "amount": 120, "unit": "mmHg"}],
        _fixture_ranges(),
    )

    assert result["flags"] == []
    assert result["notes"][0]["reason"] == "unit_mismatch"
    assert result["notes"][0]["message"] == "unit mismatch, not checked"


def test_missing_reference_range_yields_not_checked_note_without_implicit_pass():
    result = check_dose_ranges(
        [{"drug": "unknownmed", "route": "oral", "amount": 1000, "unit": "mg"}],
        _fixture_ranges(),
    )

    assert result["flags"] == []
    assert result["notes"][0]["reason"] == "missing_reference_range"
    assert result["notes"][0]["message"] == "no reference range supplied, not checked"


def test_dosing_flags_do_not_include_corrected_or_recommended_dose():
    result = check_dose_ranges(
        [{"drug": "democef", "route": "oral", "amount": 1000, "unit": "mg"}],
        _fixture_ranges(),
    )

    flag = result["flags"][0]
    assert "corrected_dose" not in flag
    assert "recommended_dose" not in flag
    assert "recommend" not in flag["message"].lower()
