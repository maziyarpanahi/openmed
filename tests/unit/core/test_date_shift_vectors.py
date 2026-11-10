"""Golden vectors that pin the exact outputs of the date-shift offset helpers.

``test_date_shift.py`` checks that the offsets are stable, bounded and non-zero.
These vectors pin the values themselves, so a port or a refactor that changes
every shifted date cannot pass unnoticed. All fixture material is synthetic.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from openmed.core.date_shift import stable_offset_for, stable_offset_from_seed

_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "parity" / "date_shift_v1.json"
_VECTORS = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_CASES = {
    case["id"]: case
    for group in ("patient_key_cases", "seed_cases", "gregorian_cases")
    for case in _VECTORS[group]
}


def _ids(group: str) -> list[str]:
    return [case["id"] for case in _VECTORS[group]]


def _keyed_offset(case: dict) -> int:
    return stable_offset_for(
        case["patient_key"],
        max_days=case["max_days"],
        secret=bytes.fromhex(case["secret_hex"]),
    )


def _require(matches: bool, case_id: str) -> None:
    # Report the case identifier only: never echo keys, secrets or dates.
    if not matches:
        pytest.fail(f"date-shift vector mismatch: {case_id}", pytrace=False)


def test_fixture_is_labelled_synthetic_and_has_unique_case_ids():
    assert _VECTORS["schema_version"] == 1
    assert _VECTORS["synthetic"] is True
    total = sum(len(_VECTORS[group]) for group in _VECTORS if group.endswith("_cases"))
    assert len(_CASES) == total
    assert {case["max_days"] for case in _CASES.values()} == {1, 30, 365}


@pytest.mark.parametrize("case_id", _ids("patient_key_cases"))
def test_patient_keyed_offset_matches_vector(case_id):
    case = _CASES[case_id]
    _require(_keyed_offset(case) == case["offset"], case_id)


@pytest.mark.parametrize("case_id", _ids("seed_cases"))
def test_seeded_offset_matches_vector(case_id):
    case = _CASES[case_id]
    actual = stable_offset_from_seed(case["seed"], max_days=case["max_days"])
    _require(actual == case["offset"], case_id)


@pytest.mark.parametrize("case_id", _ids("gregorian_cases"))
def test_offset_applied_on_the_gregorian_calendar_matches_vector(case_id):
    case = _CASES[case_id]
    shifted = date.fromisoformat(case["date"]) + timedelta(days=_keyed_offset(case))
    _require(shifted.isoformat() == case["shifted"], case_id)
