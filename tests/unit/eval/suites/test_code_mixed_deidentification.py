from __future__ import annotations

import json

from openmed.eval.suites.code_mixed_routing import (
    CODE_MIXED_ENTITY_LEAKAGE_GATE,
    CODE_MIXED_PHI_RECALL_GATE,
    evaluate_code_mixed_routing,
    load_code_mixed_fixtures,
)


def test_code_mixed_synthetic_gate_has_full_recall_and_zero_leakage() -> None:
    fixtures = load_code_mixed_fixtures()
    report = evaluate_code_mixed_routing(fixtures)

    assert all(fixture.synthetic for fixture in fixtures)
    assert report.recall >= CODE_MIXED_PHI_RECALL_GATE
    assert report.entity_leakage_count == CODE_MIXED_ENTITY_LEAKAGE_GATE
    assert report.english_false_positive_count == 0
    assert report.passed


def test_code_mixed_report_is_aggregate_only_and_deterministic() -> None:
    report = evaluate_code_mixed_routing()

    payload = report.to_dict()
    assert payload == report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    assert "Rahul" not in serialized
    assert "9876543210" not in serialized
    assert payload["gates"] == {
        "entity_leakage_max": 0,
        "phi_recall_min": 1.0,
    }
