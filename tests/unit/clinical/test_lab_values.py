from openmed.clinical.lab_values import derive_abnormal_flag, parse_reference_range


def test_parse_reference_range_dash_range() -> None:
    result = parse_reference_range("135-145")
    assert result["low"] == 135
    assert result["high"] == 145


def test_parse_reference_range_less_than() -> None:
    result = parse_reference_range("<5")
    assert result["low"] is None
    assert result["high"] == 5


def test_derive_abnormal_flag_low() -> None:
    assert derive_abnormal_flag(130, {"low": 135, "high": 145}) == "low"


def test_derive_abnormal_flag_normal() -> None:
    assert derive_abnormal_flag(140, {"low": 135, "high": 145}) == "normal"


def test_derive_abnormal_flag_explicit_high_wins() -> None:
    assert (
        derive_abnormal_flag(140, {"low": 135, "high": 145}, explicit_flag="H")
        == "high"
    )


def test_derive_abnormal_flag_non_numeric_unknown() -> None:
    assert derive_abnormal_flag("not numeric", {"low": 135, "high": 145}) == "unknown"


def test_derive_abnormal_flag_unparseable_range_unknown() -> None:
    assert derive_abnormal_flag(140, "not a range") == "unknown"
