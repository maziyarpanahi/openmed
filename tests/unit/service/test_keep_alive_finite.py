"""Finite timeout regressions using synthetic duration strings only."""

import math

import pytest

from openmed.service.keep_alive import parse_keep_alive


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "NaN",
        "+inf",
        "inf",
        "-inf",
        "1e999",
        "-1e999",
        10**400,
        "9" * 400 + "s",
        "9" * 307 + "d",
        "9" * 308 + "s" + "9" * 308 + "s",
    ],
    ids=[
        "float-nan",
        "float-inf",
        "float-negative-inf",
        "nan-string",
        "mixed-case-nan",
        "signed-inf",
        "inf-string",
        "negative-inf-string",
        "positive-exponent-overflow",
        "negative-exponent-overflow",
        "huge-int",
        "component-overflow",
        "unit-overflow",
        "compound-overflow",
    ],
)
def test_nonfinite_durations_are_rejected(value):
    with pytest.raises(ValueError, match="keep_alive must be finite"):
        parse_keep_alive(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "  ",
        "off",
        "none",
        "never",
        "forever",
        "infinite",
        "infinity",
        " INFINITY ",
    ],
)
def test_explicit_opt_out_values_are_preserved(value):
    assert parse_keep_alive(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (-0.0, 0.0),
        (2.5, 2.5),
        ("1e2", 100),
        ("250ms", 0.25),
        ("1h30m", 5400),
        ("2d", 172800),
        (" 30S ", 30),
    ],
)
def test_valid_finite_durations_are_unchanged(value, expected):
    result = parse_keep_alive(value)
    assert math.isfinite(result)
    assert result == expected


@pytest.mark.parametrize("value", [True, False, -1, "-0.1", [], "one hour", "1h 30m"])
def test_invalid_durations_remain_rejected(value):
    with pytest.raises(ValueError):
        parse_keep_alive(value)
