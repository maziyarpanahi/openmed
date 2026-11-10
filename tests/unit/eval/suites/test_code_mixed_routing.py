from __future__ import annotations

import pytest

from openmed.eval.suites import (
    CODE_MIXED_ROUTING,
    DEFAULT_SUITES,
    load_suite_fixtures,
    suite_metadata,
    validate_suite_name,
)
from openmed.eval.suites.code_mixed_routing import (
    MIN_TOKEN_LID_ACCURACY,
    evaluate_code_mixed_routing,
    load_code_mixed_routing_fixtures,
    run_code_mixed_routing,
)


def test_code_mixed_suite_is_registered():
    assert CODE_MIXED_ROUTING in DEFAULT_SUITES
    assert validate_suite_name(CODE_MIXED_ROUTING) == CODE_MIXED_ROUTING
    assert load_suite_fixtures(CODE_MIXED_ROUTING)
    metadata = suite_metadata(CODE_MIXED_ROUTING)
    assert metadata["minimum_token_lid_accuracy"] == 0.80
    assert metadata["requires_zero_entity_leakage"] is True


def test_synthetic_hinglish_fixture_is_offset_only_and_aligned():
    fixtures = load_code_mixed_routing_fixtures()

    assert fixtures
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(fixture.gold_tokens for fixture in fixtures)
    for fixture in fixtures:
        for token in fixture.gold_tokens:
            assert 0 <= token.start < token.end <= len(fixture.text)
            assert not hasattr(token, "text")


def test_code_mixed_eval_meets_accuracy_recall_and_leakage_gates():
    result = run_code_mixed_routing()

    assert result.token_lid_accuracy >= MIN_TOKEN_LID_ACCURACY
    assert result.code_mixed_deid_recall > result.baseline_deid_recall
    assert result.code_mixed_deid_recall == 1.0
    assert result.entity_leakage_count == 0
    assert result.named_entity_tokens_retained == result.named_entity_token_count
    assert result.deterministic is True
    assert result.passed is True
    assert "Patient" not in str(result.to_dict())


def test_eval_fails_closed_when_model_hook_drops_hindi_routes():
    fixtures = load_code_mixed_routing_fixtures()

    def english_hook(text, spans):
        return ["en"] * len(spans)

    result = evaluate_code_mixed_routing(fixtures, lid_model=english_hook)

    assert result.token_lid_accuracy < MIN_TOKEN_LID_ACCURACY
    assert result.passed is False


def test_fixture_loader_rejects_raw_token_surfaces(tmp_path):
    fixture_path = tmp_path / "invalid.jsonl"
    fixture_path.write_text(
        '{"id":"bad","text":"Ravi","gold_tokens":'
        '[{"start":0,"end":4,"label":"ne","text":"Ravi"}],'
        '"gold_spans":[],"metadata":{"synthetic":true}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="offset-only"):
        load_code_mixed_routing_fixtures(fixture_path)
