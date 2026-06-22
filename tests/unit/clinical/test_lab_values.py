import pytest

from openmed.clinical.lab_values import derive_abnormal_flag, parse_reference_range


def test_parse_reference_range_dash_range() -> None:
    result = parse_reference_range("135-145")
    assert result["low"] == 135
    assert result["high"] == 145


def test_parse_reference_range_decimal_range() -> None:
    result = parse_reference_range("0.5 - 1.2")
    assert result == {
        "low": 0.5,
        "high": 1.2,
        "low_inclusive": True,
        "high_inclusive": True,
    }


def test_parse_reference_range_less_than() -> None:
    result = parse_reference_range("<5")
    assert result["low"] is None
    assert result["high"] == 5
    assert result["high_inclusive"] is False


def test_parse_reference_range_greater_than_or_equal() -> None:
    result = parse_reference_range(">=10")
    assert result["low"] == 10
    assert result["low_inclusive"] is True
    assert result["high"] is None


def test_derive_abnormal_flag_low() -> None:
    assert derive_abnormal_flag(130, {"low": 135, "high": 145}) == "low"


def test_derive_abnormal_flag_normal() -> None:
    assert derive_abnormal_flag(140, {"low": 135, "high": 145}) == "normal"


def test_derive_abnormal_flag_explicit_high_wins() -> None:
    assert (
        derive_abnormal_flag(140, {"low": 135, "high": 145}, explicit_flag="H")
        == "high"
    )


@pytest.mark.parametrize(
    ("explicit_flag", "expected"),
    [
        ("L", "low"),
        ("low", "low"),
        ("H", "high"),
        ("high", "high"),
        ("critical", "critical"),
        ("C", "critical"),
        ("N", "normal"),
        ("normal", "normal"),
    ],
)
def test_derive_abnormal_flag_explicit_flags_win(explicit_flag, expected) -> None:
    assert (
        derive_abnormal_flag(140, {"low": 135, "high": 145}, explicit_flag) == expected
    )


def test_derive_abnormal_flag_non_numeric_unknown() -> None:
    assert derive_abnormal_flag("not numeric", {"low": 135, "high": 145}) == "unknown"


def test_derive_abnormal_flag_does_not_extract_embedded_numbers() -> None:
    assert derive_abnormal_flag("about 140", {"low": 135, "high": 145}) == "unknown"


def test_derive_abnormal_flag_unparseable_range_unknown() -> None:
    assert derive_abnormal_flag(140, "not a range") == "unknown"


def test_derive_abnormal_flag_unparseable_parsed_range_unknown() -> None:
    assert derive_abnormal_flag(140, parse_reference_range("not a range")) == "unknown"
