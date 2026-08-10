"""Focused tests for deterministic, value-free policy simulation."""

from __future__ import annotations

import copy
import json
import socket

import pytest

from openmed.risk.policy_simulation import (
    PolicySimulationSchemaError,
    PolicyVersion,
    render_policy_simulation_matrix,
    simulate_policy_matrix,
)


def _policies() -> tuple[dict[str, object], dict[str, object]]:
    base = {
        "version": "policy-v1",
        "actions": {"PERSON": "keep", "EMAIL": "mask"},
        "default_action": "keep",
        "blocking_gates": ["release"],
    }
    candidate = {
        "version": "policy-v2",
        "actions": {"PERSON": "mask", "EMAIL": "redact"},
        "default_action": "keep",
        "blocking_gates": ["release", "no_leak"],
    }
    return base, candidate


def _scenarios() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": "synthetic-sensitive-canary-001",
            "resource_class": "PERSON",
            "count": 2,
            "gate_outcomes": {"release": True, "no_leak": False},
        },
        {
            "scenario_id": "synthetic-email-case-002",
            "resource_class": "EMAIL",
            "count": 1,
            "gate_outcomes": {"release": True, "no_leak": True},
        },
    ]


def test_simulation_classifies_actions_counts_and_blocking_changes() -> None:
    base, candidate = _policies()
    matrix = simulate_policy_matrix(base, candidate, _scenarios())

    first, second = matrix.rows
    assert first.base_action == "keep"
    assert first.candidate_action == "mask"
    assert first.action_change == "stronger"
    assert first.count_change == "increased"
    assert first.base_gate_outcome == "pass"
    assert first.candidate_gate_outcome == "fail"
    assert first.blocking_change == "blocked"
    assert first.base_processed_count == 2
    assert first.candidate_processed_count == 0

    assert second.action_change == "stronger"
    assert second.count_change == "unchanged"
    assert second.blocking_change == "unchanged"

    summary = matrix.summary()
    assert summary["action_counts"] == {
        "base": {
            "keep": 2,
            "redact": 0,
            "replace": 0,
            "mask": 1,
            "hash": 0,
            "format_preserve": 0,
        },
        "candidate": {
            "keep": 0,
            "redact": 1,
            "replace": 0,
            "mask": 2,
            "hash": 0,
            "format_preserve": 0,
        },
        "delta": {
            "keep": -2,
            "redact": 1,
            "replace": 0,
            "mask": 1,
            "hash": 0,
            "format_preserve": 0,
        },
    }
    assert summary["count_change"]["affected"] == "increased"
    assert summary["count_change"]["base_affected"] == 1
    assert summary["count_change"]["candidate_affected"] == 3
    assert summary["blocking"]["base_blocked_count"] == 0
    assert summary["blocking"]["candidate_blocked_count"] == 2


def test_matrix_is_deterministic_and_serialization_is_value_free() -> None:
    base, candidate = _policies()
    scenarios = _scenarios()

    first = simulate_policy_matrix(base, candidate, scenarios)
    second = simulate_policy_matrix(base, candidate, scenarios)

    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    payload = json.loads(first.to_json())
    serialized = first.to_json() + first.to_markdown() + repr(first)
    assert "synthetic-sensitive-canary-001" not in serialized
    assert "synthetic-email-case-002" not in serialized
    assert payload["artifact"] == "openmed.risk.policy_simulation"
    assert payload["rows"][0]["scenario_fingerprint"].startswith("sha256:")
    assert "resource_class" in payload["rows"][0]


def test_simulation_does_not_mutate_policy_or_scenario_mappings() -> None:
    base, candidate = _policies()
    scenarios = _scenarios()
    original_base = copy.deepcopy(base)
    original_candidate = copy.deepcopy(candidate)
    original_scenarios = copy.deepcopy(scenarios)

    simulate_policy_matrix(base, candidate, scenarios)

    assert base == original_base
    assert candidate == original_candidate
    assert scenarios == original_scenarios


def test_bundled_profiles_are_supported_without_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("policy simulation must not open a network socket")

    monkeypatch.setattr(socket, "socket", fail_socket)
    matrix = simulate_policy_matrix(
        "clinical_minimal_redaction",
        "strict_no_leak",
        [
            {
                "resource_class": "LOCATION",
                "count": 1,
                "gate_outcomes": {"no_leak": True, "safety_sweep": True},
            }
        ],
    )

    assert matrix.rows[0].base_action == "keep"
    assert matrix.rows[0].candidate_action == "mask"
    assert matrix.rows[0].candidate_gate_outcome == "pass"
    assert matrix.rows[0].candidate_blocked is False


def test_render_helper_supports_markdown_json_and_dict() -> None:
    base, candidate = _policies()
    matrix = simulate_policy_matrix(base, candidate, _scenarios())

    assert render_policy_simulation_matrix(matrix).startswith(
        "# Privacy policy simulation matrix"
    )
    assert json.loads(render_policy_simulation_matrix(matrix, fmt="json")) == (
        matrix.to_dict()
    )
    assert render_policy_simulation_matrix(matrix, fmt="dict") == matrix.to_dict()


def test_payload_bearing_scenario_is_rejected_without_echoing_input() -> None:
    base, candidate = _policies()
    canary = "synthetic-sensitive-value-canary"
    with pytest.raises(PolicySimulationSchemaError) as error:
        simulate_policy_matrix(
            base,
            candidate,
            [
                {
                    "resource_class": "PERSON",
                    "count": 1,
                    "payload": canary,
                }
            ],
        )

    assert canary not in str(error.value)


def test_policy_version_and_scenario_are_immutable() -> None:
    base, _ = _policies()
    policy = PolicyVersion.from_mapping(base)

    with pytest.raises(TypeError):
        policy.actions["PERSON"] = "redact"  # type: ignore[index]
