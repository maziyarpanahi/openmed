import pytest
from openmed.clinical.lab_values import parse_reference_range, derive_abnormal_flag

def test_lab_value_parsing_and_flagging():
    # Test Range Parsing
    assert parse_reference_range('135-145')["low"] == 135
    assert parse_reference_range('135-145')["high"] == 145
    assert parse_reference_range('<5')["high"] == 5
    assert parse_reference_range('<5')["low"] is None
    assert parse_reference_range('0.5 - 1.2')["low"] == 0.5

    # Test Flag Derivation from Bounds
    parsed_range = {"low": 135, "high": 145, "low_inclusive": True, "high_inclusive": True}
    assert derive_abnormal_flag(130, parsed_range) == 'low'
    assert derive_abnormal_flag(140, parsed_range) == 'normal'
    assert derive_abnormal_flag(150, parsed_range) == 'high'

    # Test Explicit Flags overriding values
    assert derive_abnormal_flag(140, parsed_range, explicit_flag='H') == 'high'

    # Test Fallbacks to Unknown
    assert derive_abnormal_flag('invalid_numeric', parsed_range) == 'unknown'
    assert derive_abnormal_flag(140, {"low": None, "high": None}) == 'unknown'