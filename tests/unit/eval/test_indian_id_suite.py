"""Tests for the synthetic Indian multi-identifier recognizer gate."""

from __future__ import annotations

import json

from openmed.eval.suites import (
    DEFAULT_SUITES,
    INDIAN_MULTI_ID,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.indian_ids import (
    load_indian_id_fixtures,
    run_indian_id_evaluation,
)


def test_suite_is_registered_and_bundles_no_registry_data():
    assert INDIAN_MULTI_ID in DEFAULT_SUITES
    assert load_suite_fixtures(INDIAN_MULTI_ID) == load_indian_id_fixtures()

    metadata = suite_metadata(INDIAN_MULTI_ID)
    assert metadata["synthetic"] is True
    assert metadata["bundles_registry_data"] is False
    assert metadata["report_fields"] == [
        "offsets",
        "hmac_hashes",
        "identifier_types",
    ]


def test_each_identifier_has_valid_and_invalid_synthetic_fixture_coverage():
    fixtures = load_indian_id_fixtures()
    assert len(fixtures) == 8
    assert {
        span.identifier_type for fixture in fixtures for span in fixture.expected
    } == {
        "abha",
        "gstin",
        "ifsc",
        "indian_driving_licence",
        "indian_passport",
        "pan",
        "vehicle_registration",
        "voter_id_epic",
    }
    for fixture in fixtures:
        assert fixture.expected
        assert fixture.hard_negatives


def test_recognizer_gate_has_zero_leakage_and_zero_false_accepts():
    result = run_indian_id_evaluation()
    assert result.passed is True
    assert result.fixture_count == 8
    assert result.expected_span_count == 8
    assert result.detected_span_count == 8
    assert result.entity_leakage_count == 0
    assert result.false_accept_count == 0
    assert result.failures == ()


def test_eval_report_contains_no_raw_identifier_values():
    fixtures = load_indian_id_fixtures()
    report_json = json.dumps(run_indian_id_evaluation().to_dict(), sort_keys=True)
    for fixture in fixtures:
        for expected in fixture.expected:
            assert expected.text not in report_json
        for negative in fixture.hard_negatives:
            assert negative.text not in report_json
