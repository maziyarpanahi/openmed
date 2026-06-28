"""Tests for medication sig frequency and duration normalization."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    MEDICATION_SIG_ADVISORY,
    normalize_duration,
    normalize_frequency,
)


@pytest.mark.parametrize(
    ("text", "expected_per_day"),
    [
        ("qd", 1.0),
        ("daily", 1.0),
        ("BID", 2.0),
        ("TID", 3.0),
        ("QID", 4.0),
        ("q.h.s.", 1.0),
    ],
)
def test_normalize_frequency_common_sig_cues(text, expected_per_day):
    normalized = normalize_frequency(text)
    assert normalized["recognized"] is True
    assert normalized["frequency_per_day"] == expected_per_day
    assert normalized["confidence"] > 0


def test_normalize_frequency_interval_q8h():
    normalized = normalize_frequency("q8h")
    assert normalized["recognized"] is True
    assert normalized["period"] == 8
    assert normalized["period_unit"] == "h"
    assert normalized["frequency_per_day"] == 3.0


def test_normalize_frequency_interval_every_twelve_hours():
    normalized = normalize_frequency("every 12 hours")
    assert normalized["recognized"] is True
    assert normalized["period"] == 12
    assert normalized["period_unit"] == "h"
    assert normalized["frequency_per_day"] == 2.0


def test_normalize_frequency_weekly_interval():
    normalized = normalize_frequency("weekly")
    assert normalized["recognized"] is True
    assert normalized["period"] == 1
    assert normalized["period_unit"] == "wk"
    assert normalized["frequency_per_day"] == 1.0 / 7.0


def test_normalize_frequency_prn_is_flagged_without_numeric_frequency():
    normalized = normalize_frequency("PRN")
    assert normalized["recognized"] is True
    assert normalized["as_needed"] is True
    assert normalized["frequency_per_day"] is None
    assert normalized["period"] is None
    assert normalized["period_unit"] is None


def test_normalize_frequency_scheduled_prn_keeps_schedule_and_flag():
    normalized = normalize_frequency("BID PRN")
    assert normalized["recognized"] is True
    assert normalized["as_needed"] is True
    assert normalized["frequency_per_day"] == 2.0


def test_normalize_frequency_unrecognized_preserves_raw_text():
    normalized = normalize_frequency("with meals on alternating clinic days")
    assert normalized["recognized"] is False
    assert normalized["confidence"] == 0.0
    assert normalized["raw"] == "with meals on alternating clinic days"
    assert normalized["frequency_per_day"] is None


@pytest.mark.parametrize(
    ("text", "value", "unit", "days"),
    [
        ("x 7 days", 7, "d", 7),
        ("x7d", 7, "d", 7),
        ("for 2 weeks", 2, "wk", 14),
        ("10/7", 10, "d", 10),
        ("2/52", 2, "wk", 14),
    ],
)
def test_normalize_duration_common_cues(text, value, unit, days):
    normalized = normalize_duration(text)
    assert normalized["recognized"] is True
    assert normalized["value"] == value
    assert normalized["unit"] == unit
    assert normalized["days"] == days
    assert normalized["confidence"] == 1.0


def test_normalize_duration_unrecognized_preserves_raw_text():
    normalized = normalize_duration("until next visit")
    assert normalized["recognized"] is False
    assert normalized["confidence"] == 0.0
    assert normalized["raw"] == "until next visit"
    assert normalized["days"] is None


def test_medication_sig_advisory_documents_prn_and_review_scope():
    assert "PRN" in MEDICATION_SIG_ADVISORY
    assert "clinician review" in MEDICATION_SIG_ADVISORY
