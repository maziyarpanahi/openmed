"""Tests for laboratory reference-range helpers."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    LAB_FLAG_ADVISORY,
    derive_abnormal_flag,
    parse_reference_range,
)


def test_parse_closed_reference_range():
    parsed = parse_reference_range("135-145")
    assert parsed == {
        "low": 135.0,
        "high": 145.0,
        "low_inclusive": True,
        "high_inclusive": True,
    }


def test_parse_decimal_reference_range_with_spaces():
    parsed = parse_reference_range("0.5 - 1.2")
    assert parsed["low"] == 0.5
    assert parsed["high"] == 1.2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<5", {"low": None, "high": 5.0, "high_inclusive": False}),
        ("<=5", {"low": None, "high": 5.0, "high_inclusive": True}),
        (">10", {"low": 10.0, "high": None, "low_inclusive": False}),
        (">=10", {"low": 10.0, "high": None, "low_inclusive": True}),
    ],
)
def test_parse_one_sided_reference_ranges(text, expected):
    parsed = parse_reference_range(text)
    for key, value in expected.items():
        assert parsed[key] == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a range",
        "1..2 - 3",
        "145-135",
        "5 mg/dL",
        "5-",
    ],
)
def test_parse_unparseable_reference_range_returns_empty_bounds(text):
    assert parse_reference_range(text) == {
        "low": None,
        "high": None,
        "low_inclusive": True,
        "high_inclusive": True,
    }


def test_derive_abnormal_flag_from_parsed_bounds():
    parsed_range = {
        "low": 135,
        "high": 145,
        "low_inclusive": True,
        "high_inclusive": True,
    }
    assert derive_abnormal_flag(130, parsed_range) == "low"
    assert derive_abnormal_flag(135, parsed_range) == "normal"
    assert derive_abnormal_flag(140, parsed_range) == "normal"
    assert derive_abnormal_flag(145, parsed_range) == "normal"
    assert derive_abnormal_flag(150, parsed_range) == "high"


def test_derive_abnormal_flag_from_raw_range_text():
    assert derive_abnormal_flag(130, "135-145") == "low"
    assert derive_abnormal_flag(140, "135-145") == "normal"
    assert derive_abnormal_flag(150, "135-145") == "high"


def test_exclusive_one_sided_bounds_are_honored():
    assert derive_abnormal_flag(5, "<5") == "high"
    assert derive_abnormal_flag(4.9, "<5") == "normal"
    assert derive_abnormal_flag(10, ">10") == "low"
    assert derive_abnormal_flag(10.1, ">10") == "normal"


@pytest.mark.parametrize(
    ("explicit_flag", "expected"),
    [
        ("H", "high"),
        ("high", "high"),
        ("L", "low"),
        ("low", "low"),
        ("C", "critical"),
        ("critical", "critical"),
    ],
)
def test_explicit_flags_override_derived_value(explicit_flag, expected):
    assert derive_abnormal_flag(140, "135-145", explicit_flag=explicit_flag) == expected


@pytest.mark.parametrize(
    ("value", "reference_range"),
    [
        ("invalid_numeric", {"low": 135, "high": 145}),
        ("5 mg/dL", "<5"),
        (140, "not a range"),
        (140, {"low": None, "high": None}),
        (140, {"low": "not numeric", "high": 145}),
    ],
)
def test_derive_abnormal_flag_returns_unknown_when_it_cannot_derive(
    value,
    reference_range,
):
    assert derive_abnormal_flag(value, reference_range) == "unknown"


def test_lab_flag_advisory_documents_heuristic_scope():
    assert "heuristic" in LAB_FLAG_ADVISORY
    assert "originating laboratory" in LAB_FLAG_ADVISORY
