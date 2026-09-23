"""Tests for synthetic, metadata-only governed-agent trace fixtures."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pytest

from openmed.agent.outcomes import OutcomeClass
from openmed.eval.agent_trace_fixtures import (
    AGENT_TRACE_FIXTURE_SCHEMA_VERSION,
    DEFAULT_AGENT_TRACE_FIXTURE_PATH,
    AgentTraceFixtureError,
    GovernedAgentScenario,
    GovernedAgentTraceFixture,
    load_governed_agent_trace_fixtures,
)

EXPECTED_CASES = {
    "read_only_allow": ("success", "completed"),
    "minimum_data_projection": ("success", "completed"),
    "missing_capability": ("abstained", "out_of_scope"),
    "purpose_mismatch": ("policy_denied", "purpose_mismatch"),
    "expired_consent": ("policy_denied", "consent_required"),
    "human_review": ("review_required", "human_gate"),
    "bounded_failure": ("failed", "timeout"),
}


def _valid_payload(case_id: str = "synthetic_case") -> dict[str, object]:
    return {
        "schema_version": AGENT_TRACE_FIXTURE_SCHEMA_VERSION,
        "case_id": case_id,
        "scenario": "read_only_allow",
        "run_id": "run_" + "1" * 32,
        "action_ids": ["act_" + "2" * 32],
        "trace_digest": "sha256:" + "3" * 64,
        "expected_outcome": "success",
        "expected_reason_code": "completed",
    }


def _write_jsonl(path: Path, *payloads: dict[str, object]) -> Path:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    return path


def test_bundled_cases_load_deterministically_with_declared_expectations() -> None:
    first = load_governed_agent_trace_fixtures()
    second = load_governed_agent_trace_fixtures(DEFAULT_AGENT_TRACE_FIXTURE_PATH)

    assert first == second
    assert {fixture.case_id for fixture in first} == set(EXPECTED_CASES)
    assert [fixture.scenario.value for fixture in first] == list(EXPECTED_CASES)
    assert {
        fixture.case_id: (
            fixture.expected_outcome.value,
            fixture.expected_reason_code,
        )
        for fixture in first
    } == EXPECTED_CASES
    assert all(
        fixture.schema_version == AGENT_TRACE_FIXTURE_SCHEMA_VERSION
        for fixture in first
    )


def test_every_declared_scenario_has_exactly_one_coverage_case() -> None:
    fixtures = load_governed_agent_trace_fixtures()

    assert {fixture.scenario for fixture in fixtures} == set(GovernedAgentScenario)
    assert len(fixtures) == len(GovernedAgentScenario)


def test_fixture_serialization_is_metadata_only_and_stable() -> None:
    fixtures = load_governed_agent_trace_fixtures()

    for fixture in fixtures:
        payload = fixture.to_dict()
        assert list(payload) == [
            "schema_version",
            "case_id",
            "scenario",
            "run_id",
            "action_ids",
            "trace_digest",
            "expected_outcome",
            "expected_reason_code",
        ]
        assert GovernedAgentTraceFixture.from_dict(payload) == fixture
        assert isinstance(fixture.expected_outcome, OutcomeClass)
        assert payload.keys().isdisjoint(
            {
                "name",
                "text",
                "prompt",
                "arguments",
                "credentials",
                "tool_input",
                "tool_output",
            }
        )


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ({"unexpected": "sentinel"}, "unknown_field"),
        ({"trace_digest": "sha256:not-a-digest"}, "invalid_digest"),
        ({"expected_outcome": "undeclared"}, "invalid_expected_outcome"),
        ({"expected_reason_code": "undeclared"}, "invalid_expected_outcome"),
    ],
)
def test_invalid_cases_fail_with_phi_safe_errors(
    tmp_path: Path,
    mutation: dict[str, object],
    error_code: str,
) -> None:
    sentinel = "Patient Example chart text bearer-secret tool-argument"
    payload = _valid_payload()
    payload.update(mutation)
    if "unexpected" in payload:
        payload["unexpected"] = sentinel

    with pytest.raises(AgentTraceFixtureError) as caught:
        load_governed_agent_trace_fixtures(
            _write_jsonl(tmp_path / "invalid.jsonl", payload)
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert caught.value.code == error_code
    assert sentinel not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    payload = _valid_payload("duplicate_case")
    path = _write_jsonl(tmp_path / "duplicate.jsonl", payload, payload)

    with pytest.raises(AgentTraceFixtureError) as caught:
        load_governed_agent_trace_fixtures(path)

    assert caught.value.code == "duplicate_case_id"
    assert caught.value.line_number == 2


def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    payload = json.dumps(_valid_payload()).removesuffix("}")
    path = tmp_path / "duplicate-field.jsonl"
    path.write_text(payload + ',"case_id":"other"}\n', encoding="utf-8")

    with pytest.raises(AgentTraceFixtureError) as caught:
        load_governed_agent_trace_fixtures(path)

    assert caught.value.code == "duplicate_field"
    assert caught.value.line_number == 1


def test_committed_fixture_contains_no_payload_or_identity_fields() -> None:
    forbidden_fields = {
        "name",
        "patient_id",
        "clinician_id",
        "chart_text",
        "prompt",
        "credentials",
        "tool_arguments",
        "tool_input",
        "tool_output",
    }

    for line in DEFAULT_AGENT_TRACE_FIXTURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        payload = json.loads(line)
        assert set(payload).isdisjoint(forbidden_fields)
        assert set(payload) == {
            "schema_version",
            "case_id",
            "scenario",
            "run_id",
            "action_ids",
            "trace_digest",
            "expected_outcome",
            "expected_reason_code",
        }
