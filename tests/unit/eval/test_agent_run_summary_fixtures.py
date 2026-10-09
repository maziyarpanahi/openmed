"""Metadata-only golden vectors and independent fail-closed mutations."""

from __future__ import annotations

import hashlib
import json
import socket
import time
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.agent.run_commitment import compute_run_summary_commitment
from openmed.agent.run_summary import RunSummary
from openmed.eval.agent_run_summary_fixtures import (
    DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH,
    MAX_RUN_SUMMARY_FIXTURE_BYTES,
    AgentRunSummaryFixtureError,
    load_agent_run_summary_fixtures,
)

EXPECTED_CASES = {
    "empty",
    "success",
    "abstention",
    "denial",
    "review",
    "failure",
    "mixed",
}


def _payload():
    return json.loads(DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH.read_text())


def _load(tmp_path, payload):
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(payload))
    return load_agent_run_summary_fixtures(path)


def test_every_declared_vector_rebuilds_byte_for_byte():
    fixtures = load_agent_run_summary_fixtures()
    assert {case.case_id for case in fixtures} == EXPECTED_CASES
    assert len(fixtures) == 7
    assert sum(len(case.events) for case in fixtures) == 11
    assert fixtures == load_agent_run_summary_fixtures(
        DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH
    )
    for case in fixtures:
        summary = case.build_summary()
        assert summary.to_json().encode() == case.expected_json.encode()
        assert RunSummary.from_json(case.expected_json).to_json() == case.expected_json
        assert (
            "sha256:" + hashlib.sha256(summary.to_markdown().encode()).hexdigest()
            == case.expected_markdown_digest
        )
        assert compute_run_summary_commitment(summary) == case.expected_commitment
        assert (
            RunSummary.from_events(reversed(case.events)).to_json()
            == case.expected_json
        )


def test_bundle_contains_only_closed_metadata_and_opaque_identifiers():
    payload = _payload()
    assert [case.to_dict() for case in load_agent_run_summary_fixtures()] == payload[
        "cases"
    ]
    for case in payload["cases"]:
        assert set(case) == {
            "case_id",
            "events",
            "expected_json",
            "expected_markdown_digest",
            "expected_commitment",
        }
        for event in case["events"]:
            assert event["workflow_id"].startswith("wf_")
            assert len(event["workflow_id"]) == 35
            assert set(event) == {
                "workflow_id",
                "outcome",
                "tool_call_count",
                "duration_seconds",
                "artifact_digests",
            }


def test_loader_needs_no_network_clock_models_or_credentials(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("an offline fixture loader attempted external work")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    for key in ("HF_TOKEN", "OPENAI_API_KEY", "OPENMED_MODEL", "OPENMED_EHR_URL"):
        monkeypatch.delenv(key, raising=False)
    assert len(load_agent_run_summary_fixtures()) == 7


@pytest.mark.parametrize(
    "version",
    [
        "schema_version",
        "run_summary_schema_version",
        "outcome_schema_version",
        "commitment_version",
    ],
)
def test_each_version_mutation_fails_independently(tmp_path, version):
    payload = _payload()
    payload[version] = "unsupported"
    with pytest.raises(AgentRunSummaryFixtureError, match="^unsupported_version$"):
        _load(tmp_path, payload)


@pytest.mark.parametrize("location", ["root", "case", "event", "outcome"])
def test_unknown_fields_never_echo_rejected_content(tmp_path, location):
    payload = _payload()
    objects = {
        "root": payload,
        "case": payload["cases"][1],
        "event": payload["cases"][1]["events"][0],
        "outcome": payload["cases"][1]["events"][0]["outcome"],
    }
    sentinel = "PRIVATE_SENTINEL chart text /private/path Bearer credential"
    objects[location][sentinel] = sentinel
    with pytest.raises(AgentRunSummaryFixtureError) as caught:
        _load(tmp_path, payload)
    assert str(caught.value) == "unknown_field"
    assert sentinel not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("workflow_id", "A Synthetic Name"),
        ("workflow_id", "/private/sensitive"),
        ("workflow_id", "Bearer-synthetic-token"),
        ("tool_call_count", True),
        ("tool_call_count", -1),
        ("duration_seconds", -1),
        ("duration_seconds", True),
        ("artifact_digests", ["not-a-digest"]),
        ("artifact_digests", "sha256:" + "1" * 64),
    ],
)
def test_invalid_event_mutations_fail_closed(tmp_path, field, value):
    payload = _payload()
    payload["cases"][1]["events"][0][field] = value
    with pytest.raises(AgentRunSummaryFixtureError) as caught:
        _load(tmp_path, payload)
    assert str(caught.value) in {
        "invalid_workflow_id",
        "invalid_event",
        "invalid_artifact_digests",
    }
    assert caught.value.__context__ is None


def test_unknown_outcome_and_duplicate_case_ids_fail_independently(tmp_path):
    payload = _payload()
    payload["cases"][1]["events"][0]["outcome"]["reason_code"] = "clinical_text"
    with pytest.raises(AgentRunSummaryFixtureError, match="^invalid_event$"):
        _load(tmp_path, payload)
    payload = _payload()
    payload["cases"][1] = payload["cases"][0]
    with pytest.raises(AgentRunSummaryFixtureError, match="^duplicate_case_id$"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    "field,code",
    [
        ("expected_json", "stale_json"),
        ("expected_markdown_digest", "stale_markdown"),
        ("expected_commitment", "stale_commitment"),
    ],
)
def test_stale_output_mutations_are_independent(tmp_path, field, code):
    payload = _payload()
    payload["cases"][1][field] = (
        "{}" if field == "expected_json" else "sha256:" + "0" * 64
    )
    with pytest.raises(AgentRunSummaryFixtureError, match=f"^{code}$"):
        _load(tmp_path, payload)


def test_event_mutation_is_not_silently_regenerated(tmp_path):
    payload = _payload()
    payload["cases"][1]["events"][0]["tool_call_count"] += 1
    with pytest.raises(AgentRunSummaryFixtureError, match="^stale_json$"):
        _load(tmp_path, payload)


@pytest.mark.parametrize("field", ["expected_markdown_digest", "expected_commitment"])
def test_malformed_expected_digests_fail_closed(tmp_path, field):
    payload = _payload()
    payload["cases"][0][field] = "not-a-digest"
    with pytest.raises(AgentRunSummaryFixtureError, match="^invalid_digest$"):
        _load(tmp_path, payload)


def test_typed_fixture_constructor_also_enforces_expectations():
    case = load_agent_run_summary_fixtures()[1]
    with pytest.raises(AgentRunSummaryFixtureError, match="^stale_commitment$"):
        replace(case, expected_commitment="sha256:" + "0" * 64)


@pytest.mark.parametrize(
    "content,code",
    [
        ('{"cases":[],"cases":[]}', "duplicate_field"),
        ('{"schema_version":NaN}', "non_finite_number"),
        ('{"schema_version":Infinity}', "non_finite_number"),
        ("[]", "invalid_object"),
        ("{", "invalid_json"),
    ],
)
def test_malformed_json_is_value_free(tmp_path, content, code):
    path = tmp_path / "vectors.json"
    path.write_text(content)
    with pytest.raises(AgentRunSummaryFixtureError, match=f"^{code}$") as caught:
        load_agent_run_summary_fixtures(path)
    assert caught.value.__context__ is None


def test_missing_oversized_and_invalid_paths_are_value_free(tmp_path):
    private_path = tmp_path / "PRIVATE_PATH_SENTINEL"
    with pytest.raises(
        AgentRunSummaryFixtureError, match="^fixture_read_failed$"
    ) as caught:
        load_agent_run_summary_fixtures(private_path)
    assert "PRIVATE_PATH_SENTINEL" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert caught.value.__context__ is None
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (MAX_RUN_SUMMARY_FIXTURE_BYTES + 1))
    with pytest.raises(AgentRunSummaryFixtureError, match="^fixture_too_large$"):
        load_agent_run_summary_fixtures(path)
    with pytest.raises(AgentRunSummaryFixtureError, match="^invalid_path$"):
        load_agent_run_summary_fixtures(object())


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda p: p.pop("commitment_version"), "missing_field"),
        (lambda p: p.update(cases=[]), "invalid_cases"),
        (lambda p: p["cases"][0].update(case_id="undeclared_case"), "invalid_case_id"),
        (lambda p: p["cases"][1]["events"][0].pop("duration_seconds"), "missing_field"),
        (
            lambda p: p["cases"][1]["events"][0]["outcome"].pop("schema_version"),
            "missing_field",
        ),
        (lambda p: p["cases"][0].update(events=[{}] * 129), "invalid_events"),
    ],
)
def test_missing_fields_and_case_bounds(tmp_path, change, code):
    payload = _payload()
    change(payload)
    with pytest.raises(AgentRunSummaryFixtureError, match=f"^{code}$"):
        _load(tmp_path, payload)
