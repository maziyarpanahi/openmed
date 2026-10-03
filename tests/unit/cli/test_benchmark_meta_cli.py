"""Offline, content-free suite inspection and aggregate report comparison."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.cli import main_module
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites import DEFAULT_SUITES, REGISTERED_EVAL_SUITES, suite_metadata


def test_list_all_suites_offline(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("suite inspection must not open network connections")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert main_module.main(["benchmark", "list-suites", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["data"]["suites"]
    assert [r["suite"] for r in rows] == list(REGISTERED_EVAL_SUITES)
    assert all(r["task"] and r["license"] and r["access"] for r in rows)
    for row in rows:
        assert "fixture_path" not in json.dumps(row)
    by_name = {r["suite"]: r for r in rows}
    assert by_name["n2c2"]["access"] == "local-dua-required"
    assert by_name["i2b2"]["access"] == "local-dua-required"
    assert by_name["golden"]["access"] == "public-synthetic"


@pytest.mark.parametrize("suite", DEFAULT_SUITES)
def test_default_suite_task_is_explicit(suite):
    assert suite_metadata(suite)["task"] not in (None, "", "unknown")


def test_describe_n2c2_retains_current_adapter_map(capsys):
    assert main_module.main(["benchmark", "describe", "n2c2", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)["data"]
    assert row["task"] == "clinical_deidentification"
    assert row["categories"]["label_mapping"]["MEDICALRECORD"] == "ID_NUM"
    assert "DUA" in row["license"]["license_id"]
    assert "credentialed" in row["requirements"]["access"]


def test_unknown_suite_does_not_echo_private_input(capsys):
    assert main_module.main(["benchmark", "describe", "CaseySecret", "--json"]) == 2
    output = capsys.readouterr().out
    assert "CaseySecret" not in output
    assert json.loads(output)["error"]["code"] == "invalid_suite"


def write_report(path, metrics, *, suite="golden", count=2):
    return BenchmarkReport(
        suite=suite,
        model_name="CaseySecret",
        device="private-host",
        fixture_count=count,
        metrics=metrics,
        metadata={"raw_note": "CaseySecret"},
    ).write_json(path)


def test_compare_deltas_and_regression(tmp_path, capsys):
    a = write_report(
        tmp_path / "a.json",
        {"leakage": {"overall": 0.0}, "exact_span_f1": {"recall": 0.9, "f1": 0.8}},
    )
    b = write_report(
        tmp_path / "b.json",
        {"leakage": {"overall": 0.1}, "exact_span_f1": {"recall": 1.0, "f1": 0.7}},
    )
    assert main_module.main(["benchmark", "compare", str(a), str(b), "--json"]) == 0
    output = capsys.readouterr().out
    assert "CaseySecret" not in output and "private-host" not in output
    result = json.loads(output)["data"]
    assert result["verdict"] == "regression"
    assert result["regression_count"] == 2
    metrics = {r["metric"]: r for r in result["metrics"]}
    assert metrics["leakage.overall"]["delta"] == pytest.approx(0.1)
    assert metrics["exact_span_f1.recall"]["status"] == "improved"
    assert metrics["exact_span_f1.f1"]["delta"] == pytest.approx(-0.1)
    assert (
        main_module.main(
            ["benchmark", "compare", str(a), str(b), "--fail-on-regression"]
        )
        == 1
    )
    assert "Verdict: regression" in capsys.readouterr().out


def test_missing_evidence_is_not_zero_or_green(tmp_path, capsys):
    a = write_report(tmp_path / "a.json", {"recall": 1.0, "f1": 1.0})
    b = write_report(tmp_path / "b.json", {"f1": 1.0})
    assert (
        main_module.main(
            ["benchmark", "compare", str(a), str(b), "--json", "--fail-on-regression"]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)["data"]
    assert result["verdict"] == "incomplete"
    row = next(row for row in result["metrics"] if row["metric"] == "recall")
    assert row["candidate"] is None and row["delta"] is None


def test_equal_report_deterministic(tmp_path, capsys):
    a = write_report(tmp_path / "a.json", {"recall": 1.0, "f1": 1.0, "leakage": 0.0})
    args = ["benchmark", "compare", str(a), str(a), "--json", "--fail-on-regression"]
    assert main_module.main(args) == 0
    first = capsys.readouterr().out
    assert main_module.main(args) == 0
    assert capsys.readouterr().out == first
    assert json.loads(first)["data"]["verdict"] == "no-regression"


@pytest.mark.parametrize(
    "value", [True, "CaseySecret", None, -0.1, 1.1, float("nan"), float("inf")]
)
def test_invalid_metric_fails_safely(value, tmp_path, capsys):
    a = write_report(tmp_path / "a.json", {"f1": value})
    assert main_module.main(["benchmark", "compare", str(a), str(a), "--json"]) == 2
    output = capsys.readouterr().out
    assert "CaseySecret" not in output
    assert json.loads(output)["error"]["code"] == "invalid_reports"


@pytest.mark.parametrize("suite,count", [("shield", 2), ("golden", 3), ("golden", 0)])
def test_mismatched_or_empty_reports_rejected(suite, count, tmp_path, capsys):
    a = write_report(tmp_path / "a.json", {"f1": 1.0})
    b = write_report(tmp_path / "b.json", {"f1": 1.0}, suite=suite, count=count)
    assert main_module.main(["benchmark", "compare", str(a), str(b), "--json"]) == 2
    assert not json.loads(capsys.readouterr().out)["ok"]


@pytest.mark.parametrize(
    "data", ["{CaseySecret", "[]", "{}", '{"metrics": 1}', "x" * (8 * 1024**2 + 1)]
)
def test_malformed_or_oversized_file_rejected(data, tmp_path, capsys):
    path = tmp_path / "CaseySecret.json"
    path.write_text(data)
    assert (
        main_module.main(["benchmark", "compare", str(path), str(path), "--json"]) == 2
    )
    assert "CaseySecret" not in capsys.readouterr().out
