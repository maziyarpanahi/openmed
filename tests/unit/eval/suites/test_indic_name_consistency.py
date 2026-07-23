"""Tests for the synthetic Indic name consistency evaluation suite."""

from openmed.eval.suites import (
    INDIC_NAME_CONSISTENCY,
    evaluate_indic_name_consistency,
    load_indic_name_fixtures,
    load_suite_fixtures,
    suite_metadata,
)


def test_indic_name_fixture_loader_and_registry():
    fixtures = load_indic_name_fixtures()

    assert len(fixtures) == 3
    assert load_suite_fixtures(INDIC_NAME_CONSISTENCY) == fixtures
    metadata = suite_metadata(INDIC_NAME_CONSISTENCY)
    assert metadata["suite"] == INDIC_NAME_CONSISTENCY
    assert metadata["synthetic"] is True
    assert metadata["leakage_gate"] == "zero"


def test_indic_name_consistency_gate_passes_stdlib_fallback():
    result = evaluate_indic_name_consistency()

    assert result.passed is True
    assert result.group_count == 3
    assert result.variant_count == 9
    assert result.surrogate_identity_count == 3
    assert result.collision_count == 0
    assert result.leakage_count == 0
    assert result.script_mismatch_count == 0
    assert result.deterministic is True
    assert result.failures == ()
