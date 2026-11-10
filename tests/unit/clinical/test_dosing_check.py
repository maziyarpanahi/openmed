"""Focused tests for caller-supplied, advisory dose-range checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    CLINICAL_DECISION_SUPPORT_DISCLAIMER,
    DOSE_RANGE_ADVISORY,
    GuardedSuggestion,
    check_dose_ranges,
    normalize_dose,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "synthetic_dose_ranges.json"
)


def _dose(
    value: object,
    unit: str = "mg",
    *,
    drug: str = "Synthetic Medication Alpha",
    route: str = "oral",
    start: int = 10,
    end: int = 24,
) -> dict[str, object]:
    return {
        "drug": drug,
        "route": route,
        "dose": value,
        "unit": unit,
        "start": start,
        "end": end,
    }


def test_synthetic_fixture_documents_user_supplied_production_ranges() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["metadata"]["synthetic"] is True
    assert payload["metadata"]["clinical_use"] is False
    assert "production" in payload["metadata"]["notice"]
    assert "user-supplied" in payload["metadata"]["notice"]


def test_over_range_dose_returns_guarded_review_flag_with_exceeded_bound() -> None:
    flags = check_dose_ranges([_dose(1000)], FIXTURE)

    assert len(flags) == 1
    flag = flags[0]
    assert isinstance(flag, GuardedSuggestion)
    assert flag.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert flag.requires_clinician_review is True
    assert flag.autonomous_decision is False
    assert [(span.start, span.end) for span in flag.source_spans] == [(10, 24)]

    suggestion = flag.suggestion
    assert suggestion["kind"] == "dose_range_flag"
    assert suggestion["status"] == "above_range"
    assert suggestion["observed_value"] == 1000
    assert suggestion["observed_unit"] == "mg"
    assert suggestion["reference_bound"] == 500
    assert suggestion["reference_bound_unit"] == "mg"
    assert suggestion["bound"] == "high"
    assert suggestion["advisory"] == DOSE_RANGE_ADVISORY
    assert "corrected_dose" not in suggestion
    assert "recommended_dose" not in suggestion


def test_below_range_dose_identifies_the_lower_bound() -> None:
    flags = check_dose_ranges([_dose(50)], FIXTURE)

    assert len(flags) == 1
    assert flags[0].suggestion["status"] == "below_range"
    assert flags[0].suggestion["reference_bound"] == 100
    assert flags[0].suggestion["bound"] == "low"


def test_in_range_dose_produces_no_flag() -> None:
    assert check_dose_ranges([_dose(250)], FIXTURE) == []


def test_unit_conversion_is_dimension_checked_before_comparison() -> None:
    normalized = normalize_dose(1, "g")
    assert normalized["recognized"] is True
    assert normalized["canonical_value"] == pytest.approx(1.0)
    assert normalized["canonical_unit"] == "g"

    flags = check_dose_ranges([_dose(0.25, "g")], FIXTURE)
    assert flags == []


def test_incompatible_unit_returns_not_checked_note_never_false_flag() -> None:
    notes = check_dose_ranges([_dose(250, "mL")], FIXTURE)

    assert len(notes) == 1
    note = notes[0]
    assert note.suggestion["kind"] == "dose_range_note"
    assert note.suggestion["status"] == "not_checked"
    assert note.suggestion["note"] == "unit mismatch, not checked"
    assert "dose_range_flag" not in note.suggestion["kind"]


def test_missing_reference_returns_not_checked_note_not_implicit_pass() -> None:
    notes = check_dose_ranges(
        [_dose(250, drug="Synthetic Medication Without Range")],
        FIXTURE,
    )

    assert len(notes) == 1
    assert notes[0].suggestion["status"] == "not_checked"
    assert notes[0].suggestion["note"] == ("no reference range supplied, not checked")


def test_nested_mapping_reference_table_is_supported() -> None:
    table = {
        "Synthetic Medication Alpha": {"oral": {"low": 100, "high": 500, "unit": "mg"}}
    }

    assert check_dose_ranges([_dose(250)], table) == []
