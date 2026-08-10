"""Tests for the aggregate-only policy-impact simulator."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.core.audit import stable_hash
from openmed.risk import (
    PolicyVersion,
    TypedResource,
    evaluate_policy_impact,
)


def _policies() -> tuple[dict[str, object], dict[str, object]]:
    baseline = {
        "name": "baseline-v1",
        "actions": {"clinical_note": "keep", "lab_result": "mask"},
        "gates": {"clinical_note": "leakage", "lab_result": "quality"},
        "waivers": {"lab_result": False},
    }
    candidate = {
        "name": "candidate-v2",
        "actions": {"clinical_note": "redact", "lab_result": "mask"},
        "gates": {
            "clinical_note": ["budget", "leakage"],
            "lab_result": "quality",
        },
        "waivers": {"lab_result": True},
    }
    return baseline, candidate


def test_digest_reports_typed_action_gate_and_waiver_deltas_only() -> None:
    baseline, candidate = _policies()
    resources = [
        {
            "type": "clinical_note",
            "count": 2,
            "record_id": "synthetic-record-should-not-appear",
        },
        {"kind": "lab_result", "value": "synthetic-value-should-not-appear"},
        {"resource_type": "unchanged_type", "value": "synthetic-value"},
    ]

    impact = evaluate_policy_impact(baseline, candidate, resources)
    payload = impact.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["resources"] == {
        "total_count": 4,
        "type_counts": {
            "clinical_note": 2,
            "lab_result": 1,
            "unchanged_type": 1,
        },
    }
    assert payload["summary"] == {
        "action_changed_resource_count": 2,
        "changed_resource_count": 3,
        "gate_changed_resource_count": 2,
        "unchanged_resource_count": 1,
        "waiver_changed_resource_count": 1,
    }
    assert payload["action_deltas"] == [
        {
            "count": 2,
            "from": "keep",
            "resource_type": "clinical_note",
            "to": "redact",
        }
    ]
    assert payload["gate_deltas"] == [
        {
            "count": 2,
            "from": "leakage",
            "resource_type": "clinical_note",
            "to": ["budget", "leakage"],
        }
    ]
    assert payload["waiver_deltas"] == [
        {
            "count": 1,
            "from": False,
            "resource_type": "lab_result",
            "to": True,
        }
    ]
    assert "synthetic-record-should-not-appear" not in serialized
    assert "synthetic-value-should-not-appear" not in serialized
    assert impact.digest == stable_hash(impact.canonical_payload())


def test_digest_is_order_independent_and_inputs_are_not_mutated() -> None:
    baseline, candidate = _policies()
    original_baseline = copy.deepcopy(baseline)
    original_candidate = copy.deepcopy(candidate)
    resources = [TypedResource("clinical_note", count=2), "lab_result"]

    first = evaluate_policy_impact(baseline, candidate, resources)
    second = evaluate_policy_impact(
        baseline,
        candidate,
        list(reversed(resources)),
    )

    assert first.to_dict() == second.to_dict()
    assert baseline == original_baseline
    assert candidate == original_candidate


def test_policy_version_normalizes_compact_resource_rules_and_waiver_reasons() -> None:
    baseline = PolicyVersion.from_mapping(
        {
            "name": "v1",
            "resources": {
                "clinical_note": {
                    "action": "keep",
                    "gate": ["leakage"],
                    "waiver": False,
                }
            },
        }
    )
    candidate = PolicyVersion.from_mapping(
        {
            "name": "v2",
            "resources": {
                "clinical_note": {
                    "action": "redact",
                    "gate": ["leakage", "budget"],
                    "waiver": "synthetic review note",
                }
            },
        }
    )

    impact = evaluate_policy_impact(
        baseline,
        candidate,
        {"clinical_note": 3},
    )

    assert impact.changed_resource_count == 3
    assert impact.action_deltas[0].count == 3
    assert impact.gate_deltas[0].to_value == ("budget", "leakage")
    assert impact.waiver_deltas[0].to_value is True
    assert "synthetic review note" not in impact.to_json()


def test_equivalent_versions_produce_an_empty_stable_digest() -> None:
    policy = {"name": "stable", "default_action": "mask"}

    impact = evaluate_policy_impact(
        policy,
        copy.deepcopy(policy),
        {"clinical_note": 2},
    )

    assert impact.is_empty is True
    assert impact.changed_resource_count == 0
    assert impact.unchanged_resource_count == 2
    assert impact.action_deltas == ()
    assert impact.gate_deltas == ()
    assert impact.waiver_deltas == ()
    assert (
        impact.digest
        == evaluate_policy_impact(
            policy,
            policy,
            {"clinical_note": 2},
        ).digest
    )


def test_invalid_resource_type_does_not_echo_sensitive_input_in_exception() -> None:
    raw_value = "synthetic sensitive value"

    with pytest.raises(ValueError) as exc_info:
        evaluate_policy_impact(
            {"name": "baseline"},
            {"name": "candidate"},
            [{"resource_type": raw_value}],
        )

    assert raw_value not in str(exc_info.value)
