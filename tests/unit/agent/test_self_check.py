"""Offline, synthetic contract and mutation tests for the governance runner."""

from __future__ import annotations

import copy
import io
import json
import socket
import time
import urllib.request
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import openmed.agent as agent
import openmed.agent.schemas as schemas
import openmed.agent.self_check as check
import openmed.eval.agent_run_summary_fixtures as fixtures
from openmed.cli.agent_self_check import run_from_args
from openmed.cli.main import build_parser, main

_MARKER = "SYNTHETIC-PHI-PROMPT-ENV-/private/fixture/path"
_NAMES = [n.value for n in check.SelfCheckName]


def _failures(report: check.AgentSelfCheckReport) -> set[str]:
    return {
        r.name.value for r in report.checks if r.status is check.SelfCheckStatus.FAILED
    }


def _mutate_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutate: Any
) -> None:
    document = json.loads(fixtures.DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH.read_bytes())
    mutate(document)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(fixtures, "DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH", path)


def _assert_closed(report: check.AgentSelfCheckReport) -> None:
    data = report.to_dict()
    assert set(data) == {"schema_version", "passed", "versions", "checks"}
    assert [r["name"] for r in data["checks"]] == _NAMES
    for row in data["checks"]:
        assert set(row) == {
            "name",
            "status",
            "reason",
            "case_count",
            "control_count",
            "digest",
        }
        assert row["status"] in {"passed", "failed"}
        assert row["reason"] in {"ok", "contract_mismatch", "unavailable"}
        assert type(row["case_count"]) is int and 0 <= row["case_count"] <= 2048
        assert type(row["control_count"]) is int and 0 <= row["control_count"] <= 2048
        if row["status"] == "failed":
            assert row["digest"] is None
    for output in (report.to_json(), report.to_text(), repr(report)):
        assert _MARKER not in output
        assert "wf_111" not in output
        assert "act_222" not in output
        assert "expected_json" not in output


def test_bundled_contracts_pass_deterministically() -> None:
    first, second = check.run_agent_self_check(), check.run_agent_self_check()
    assert first.passed and first == second
    assert first.to_json() == second.to_json()
    assert first.to_text() == second.to_text()
    assert [r.case_count for r in first.checks] == [4, 7, 7, 24, 2, 2, 7]
    assert all(r.control_count > 0 for r in first.checks)
    assert {r.name.value: r.digest for r in first.checks} == check._RESULT_DIGESTS
    assert len(fixtures.load_agent_run_summary_fixtures()) == 7
    _assert_closed(first)


def test_public_exports_and_frozen_report() -> None:
    for name in check.__all__:
        assert getattr(agent, name) is getattr(check, name)
        assert name in agent.__all__
    report = check.run_agent_self_check()
    with pytest.raises(FrozenInstanceError):
        report.checks = ()
    with pytest.raises(FrozenInstanceError):
        report.checks[0].digest = _MARKER
    data = report.to_dict()
    data["versions"]["outcome"] = _MARKER
    data["checks"][0]["status"] = _MARKER
    assert _MARKER not in report.to_json()


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_RUN_SUMMARY_FIXTURE_BYTES", 1_048_577),
        ("MAX_RUN_SUMMARY_FIXTURE_EVENTS", 129),
    ],
)
def test_fixture_bound_drift_fails_independently(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: int,
) -> None:
    monkeypatch.setattr(fixtures, name, value)
    report = check.run_agent_self_check()
    assert _failures(report) == {"golden"}
    _assert_closed(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", _MARKER),
        ("status", "passed"),
        ("reason", _MARKER),
        ("case_count", True),
        ("case_count", -1),
        ("control_count", 2049),
        ("digest", _MARKER),
        ("digest", None),
    ],
)
def test_result_rejects_uncontrolled_fields(field: str, value: Any) -> None:
    data = {
        "name": check.SelfCheckName.SCHEMA,
        "status": check.SelfCheckStatus.PASSED,
        "reason": check.SelfCheckReason.OK,
        "case_count": 1,
        "control_count": 1,
        "digest": "sha256:" + "1" * 64,
    }
    data[field] = value
    with pytest.raises(ValueError, match="self_check:") as exc:
        check.AgentSelfCheckResult(**data)
    assert _MARKER not in str(exc.value)


@pytest.mark.parametrize("shape", ["missing", "duplicate", "reversed", "list"])
def test_report_requires_every_check_once(shape: str) -> None:
    rows = check.run_agent_self_check().checks
    invalid = {
        "missing": rows[:-1],
        "duplicate": rows[:-1] + (rows[0],),
        "reversed": tuple(reversed(rows)),
        "list": list(rows),
    }[shape]
    with pytest.raises(ValueError, match="incomplete_report"):
        check.AgentSelfCheckReport(invalid)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("outcome", "additionalProperties", True),
        ("outcome", "description", _MARKER),
        ("correlation", "$id", _MARKER),
        ("timing", "$ref", "https://invalid.example/" + _MARKER),
        ("run_summary", "required", []),
    ],
)
def test_schema_mutations_fail_independently(
    monkeypatch: pytest.MonkeyPatch, name: str, field: str, value: Any
) -> None:
    catalog = schemas.build_agent_schema_catalog()
    catalog[name][field] = value
    monkeypatch.setattr(schemas, "build_agent_schema_catalog", lambda: catalog)
    report = check.run_agent_self_check()
    assert _failures(report) == {"schema"}
    _assert_closed(report)


@pytest.mark.parametrize("mutation", ["json", "markdown", "coherent"])
def test_golden_mutations_fail_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    def mutate(document: Any) -> None:
        case = document["cases"][1]
        if mutation == "markdown":
            case["expected_markdown_digest"] = "sha256:" + "0" * 64
        elif mutation == "json":
            data = json.loads(case["expected_json"])
            data["tool_call_count"] += 1
            case["expected_json"] = check._canonical(data)
        else:
            # Regenerating expectations must not silently redefine the gold.
            case["events"][0]["tool_call_count"] += 1
            summary = check.run_summary.RunSummary.from_events(check._events(case))
            case["expected_json"] = summary.to_json()
            case["expected_markdown_digest"] = (
                "sha256:"
                + check.hashlib.sha256(summary.to_markdown().encode()).hexdigest()
            )
            case["expected_commitment"] = (
                check.run_commitment.compute_run_summary_commitment(summary)
            )

    _mutate_fixture(monkeypatch, tmp_path, mutate)
    report = check.run_agent_self_check()
    assert "golden" in _failures(report)
    assert _failures(report) == (
        {"golden", "commitments"} if mutation == "coherent" else {"golden"}
    )
    _assert_closed(report)


@pytest.mark.parametrize("boundary", ["root", "case", "event", "outcome", "summary"])
def test_unsafe_fields_fail_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, boundary: str
) -> None:
    def mutate(document: Any) -> None:
        case = document["cases"][1]
        targets = {
            "root": document,
            "case": case,
            "event": case["events"][0],
            "outcome": case["events"][0]["outcome"],
        }
        if boundary == "summary":
            data = json.loads(case["expected_json"])
            data[_MARKER] = _MARKER
            case["expected_json"] = check._canonical(data)
        else:
            targets[boundary][_MARKER] = _MARKER

    _mutate_fixture(monkeypatch, tmp_path, mutate)
    report = check.run_agent_self_check()
    assert _failures(report) == {"unsafe_fields"}
    _assert_closed(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "run_" + "4" * 32),
        ("action_id", _MARKER),
        ("parent_action_id", "act_" + "3" * 32),
        ("schema_version", _MARKER),
        ("prompt", _MARKER),
    ],
)
def test_correlation_mutations_fail_independently(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    cases = copy.deepcopy(check._CORRELATION_CASES)
    cases[1][field] = value
    monkeypatch.setattr(check, "_CORRELATION_CASES", cases)
    report = check.run_agent_self_check()
    assert _failures(report) == {"correlations"}
    assert report.checks[4].case_count == 2
    assert report.checks[4].control_count == 5
    _assert_closed(report)


@pytest.mark.parametrize(
    "mutation",
    ["duration", "negative", "bool", "outside", "cycle", "coherent", "unsafe"],
)
def test_timing_mutations_fail_independently(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    cases = copy.deepcopy(check._TIMING_CASES)
    case = cases[1]
    if mutation == "duration":
        case["run"]["duration_ns"] += 1
    elif mutation == "negative":
        case["run"]["start_ns"] = -1
    elif mutation == "bool":
        case["run"]["start_ns"] = True
    elif mutation == "outside":
        case["actions"][1].update(end_ns=111, duration_ns=81)
    elif mutation == "cycle":
        case["actions"][0]["parent_action_id"] = case["actions"][1]["action_id"]
    elif mutation == "coherent":
        case["run"].update(end_ns=111, duration_ns=101)
    else:
        case["actions"][1][_MARKER] = _MARKER
    monkeypatch.setattr(check, "_TIMING_CASES", cases)
    report = check.run_agent_self_check()
    assert _failures(report) == {"timings"}
    assert report.checks[5].case_count == 2
    assert report.checks[5].control_count == 10
    _assert_closed(report)


@pytest.mark.parametrize("candidate", ["sha256:" + "0" * 64, _MARKER, None, True])
def test_commitment_mutations_fail_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, candidate: Any
) -> None:
    _mutate_fixture(
        monkeypatch,
        tmp_path,
        lambda doc: doc["cases"][1].update(expected_commitment=candidate),
    )
    report = check.run_agent_self_check()
    assert _failures(report) == {"commitments"}
    _assert_closed(report)


def test_outcome_mutation_is_reported_without_stopping_other_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mutate_fixture(
        monkeypatch,
        tmp_path,
        lambda doc: doc["cases"][1]["events"][0]["outcome"].update(reason_code=_MARKER),
    )
    report = check.run_agent_self_check()
    assert _failures(report) == {"golden", "unsafe_fields", "outcomes", "commitments"}
    _assert_closed(report)


def test_negative_controls_catch_broken_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check.run_commitment,
        "verify_run_summary_commitment",
        lambda *args: check.run_commitment.RunCommitmentVerificationResult.VERIFIED,
    )
    report = check.run_agent_self_check()
    assert _failures(report) == {"commitments"}
    assert report.checks[-1].control_count == 7


@pytest.mark.parametrize(
    "validator", ["schema", "outcomes", "timings", "unsafe_fields"]
)
def test_negative_controls_catch_permissive_validators(
    monkeypatch: pytest.MonkeyPatch,
    validator: str,
) -> None:
    if validator == "schema":
        from jsonschema import Draft202012Validator

        monkeypatch.setattr(Draft202012Validator, "is_valid", lambda *args: True)
    elif validator == "outcomes":
        monkeypatch.setattr(check.outcomes, "_validate_reason_code", lambda *args: None)
    elif validator == "timings":
        monkeypatch.setattr(check.timing, "_validate_actions", lambda *args: None)
    else:
        original = check.run_summary.RunSummary.from_dict
        monkeypatch.setattr(
            check.run_summary.RunSummary,
            "from_dict",
            classmethod(
                lambda cls, data: original(
                    {k: v for k, v in data.items() if k in check._SUMMARY_FIELDS}
                )
            ),
        )
    report = check.run_agent_self_check()
    assert _failures(report) == {validator}
    _assert_closed(report)


@pytest.mark.parametrize("field", ["expected_json", "expected_commitment"])
def test_missing_expectation_does_not_stop_unrelated_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    _mutate_fixture(monkeypatch, tmp_path, lambda doc: doc["cases"][1].pop(field))
    report = check.run_agent_self_check()
    owner = "golden" if field == "expected_json" else "commitments"
    assert _failures(report) == {owner, "unsafe_fields"}
    _assert_closed(report)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b"[]",
        b"\xff",
        b"x" * (1_048_576 + 1),
    ],
)
def test_bad_fixture_sources_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "synthetic.json"
    path.write_bytes(payload)
    monkeypatch.setattr(fixtures, "DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH", path)
    report = check.run_agent_self_check()
    assert _failures(report) == {"golden", "unsafe_fields", "outcomes", "commitments"}
    _assert_closed(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(schema_version=_MARKER),
        lambda doc: doc.update(cases=doc["cases"][:-1]),
        lambda doc: doc["cases"].__setitem__(1, doc["cases"][0]),
        lambda doc: doc["cases"][1]["events"][0].update(workflow_id=_MARKER),
    ],
)
def test_invalid_versions_case_coverage_and_identifiers_are_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Any,
) -> None:
    _mutate_fixture(monkeypatch, tmp_path, mutate)
    report = check.run_agent_self_check()
    assert not report.passed
    _assert_closed(report)


def test_exception_text_and_classes_never_leave_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> Any:
        raise RuntimeError(_MARKER)

    monkeypatch.setattr(check, "_load_fixture_document", fail)
    monkeypatch.setattr(schemas, "build_agent_schema_catalog", fail)
    report = check.run_agent_self_check()
    assert _failures(report) == {
        "schema",
        "golden",
        "unsafe_fields",
        "outcomes",
        "commitments",
    }
    _assert_closed(report)


def test_cli_json_and_text_are_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for flag in ([], ["--json"]):
        assert main(["agent", "self-check", *flag]) == 0
        first = capsys.readouterr()
        assert main(["agent", "self-check", *flag]) == 0
        second = capsys.readouterr()
        assert first == second and not first.err
        assert _MARKER not in first.out
        if flag:
            data = json.loads(first.out)
            assert data["command"] == "agent self-check" and data["ok"] is True
            assert data["data"] == check.run_agent_self_check().to_dict()
        else:
            assert "Agent governance self-check: passed" in first.out


def test_cli_failure_preserves_all_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check, "_CORRELATION_CASES", ())
    assert main(["agent", "self-check", "--json"]) == 1
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert data["data"]["passed"] is False
    assert len(data["data"]["checks"]) == 7
    assert not output.err and _MARKER not in output.out


def test_cli_unexpected_error_is_controlled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> Any:
        raise RuntimeError(_MARKER)

    monkeypatch.setattr(check, "run_agent_self_check", fail)
    assert main(["agent", "self-check", "--json"]) == 1
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert data["error"]["code"] == "agent_self_check_unavailable"
    assert _MARKER not in output.out + output.err


def test_handler_uses_caller_owned_stream() -> None:
    args = build_parser().parse_args(["agent", "self-check", "--json"])
    stream = io.StringIO()
    assert run_from_args(args, stdout=stream) == 0
    assert json.loads(stream.getvalue())["data"]["passed"] is True


def test_no_network_sleeps_models_or_config_reads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib
    import os

    cli_main = importlib.import_module("openmed.cli.main")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(_MARKER)

    with monkeypatch.context() as context:
        for obj, attr in (
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (urllib.request, "urlopen"),
            (time, "sleep"),
            (os, "getenv"),
            (cli_main, "_load_and_apply_config"),
            (cli_main, "get_config"),
            (cli_main, "prefetch_model"),
        ):
            context.setattr(obj, attr, forbidden)
        assert check.run_agent_self_check().passed
        assert main(["--config-path", _MARKER, "agent", "self-check", "--json"]) == 0
        output = capsys.readouterr()
        assert _MARKER not in output.out + output.err


@pytest.mark.parametrize(
    "payload", ["[" * 30 + "0" + "]" * 30, json.dumps([0] * 20_001)]
)
def test_parser_has_depth_and_node_budgets(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid_json_bounds"):
        check._bounded_json(payload)
