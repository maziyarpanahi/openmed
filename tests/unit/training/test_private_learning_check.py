"""Offline self-check category independence, evaluator failures and real CLI wiring."""

import importlib
import io
import json

import pytest

from openmed.training import private_learning_check as check


def test_every_mandatory_category_passes_deterministically_offline():
    first = check.run_private_learning_check()
    assert first.passed
    assert first.to_json() == check.run_private_learning_check().to_json()
    assert [c.name for c in first.checks] == list(check._NAMES)
    assert all(
        c.case_count > 0 and c.digest.startswith("sha256:") for c in first.checks
    )


@pytest.mark.parametrize("name", check._NAMES)
def test_each_contract_mutation_fails_independently_and_other_checks_run(
    name, monkeypatch
):
    monkeypatch.setattr(
        check,
        "_"
        + name.replace("schemas", "schema").replace(
            "forbidden_fields", "forbidden_fields"
        )
        + "_case",
        lambda: False,
    )
    report = check.run_private_learning_check()
    assert not report.passed
    assert [c.name for c in report.checks if not c.passed] == [name]


def test_unexpected_exception_never_escapes_or_enters_reports(monkeypatch):
    secret = "SYNTHETIC_PRIVATE_SENTINEL https://private.example /private/patient 4242 tensor_values"

    def fail():
        raise RuntimeError(secret)

    monkeypatch.setattr(check, "_update_case", fail)
    report = check.run_private_learning_check()
    assert not report.passed
    assert secret not in report.to_json() + report.to_text()
    assert len(report.checks) == 7


def test_schema_mutation_is_checked_by_actual_validator(monkeypatch):
    build = check.build_private_training_schemas

    def altered():
        catalog = build()
        catalog["federated_round_lifecycle"]["properties"]["schema_version"][
            "const"
        ] = "invalid_version"
        return catalog

    monkeypatch.setattr(check, "build_private_training_schemas", altered)
    assert [
        c.name for c in check.run_private_learning_check().checks if not c.passed
    ] == ["schemas"]


def test_actual_schedule_state_update_suppression_and_fingerprint_interventions(
    monkeypatch,
):
    monkeypatch.setattr(
        check.FederatedRoundSchedule, "active_phase_at", lambda *_: None
    )
    assert not next(
        c for c in check.run_private_learning_check().checks if c.name == "schedule"
    ).passed
    monkeypatch.undo()
    monkeypatch.setattr(check, "same_update_schema", lambda *_: True)
    assert not next(
        c for c in check.run_private_learning_check().checks if c.name == "fingerprint"
    ).passed


def test_json_export_is_allowlisted_and_contains_no_underlying_payload_values():
    result = check.run_private_learning_check().to_dict()
    assert set(result) == {"version", "contract_versions", "passed", "checks"}
    assert result["contract_versions"] == {
        "lifecycle": "v1",
        "schedule": "v1",
        "status": "v1",
        "update": "v1",
        "aggregate": "v1",
    }
    assert all(
        set(c) == {"name", "passed", "fixture_case_count", "digest", "reason_code"}
        for c in result["checks"]
    )
    rendered = json.dumps(result)
    for forbidden in (
        "model_digest",
        "update_digest",
        "aggregate_value",
        "patient_count",
        "adapter.lora",
        "endpoint",
        "parameters",
        "clipping_lower_bound",
        "participant_count",
    ):
        assert forbidden not in rendered


def test_real_registered_argparse_handler_json_and_failure_exit(monkeypatch):
    main = importlib.import_module("openmed.cli.main")
    parser = main.build_parser()
    args = parser.parse_args(["training", "self-check", "--json"])
    stream = io.StringIO()
    cli = importlib.import_module("openmed.cli.private_learning_check")
    assert cli.run_from_args(args, stdout=stream) == 0
    envelope = json.loads(stream.getvalue())
    assert envelope["command"] == "training self-check"
    assert envelope["data"]["passed"]
    stream = io.StringIO()
    monkeypatch.setattr(check, "_update_case", lambda: False)
    assert cli.run_from_args(args, stdout=stream) == 1
    assert json.loads(stream.getvalue())["outcome"] == "check_failed"


def test_no_models_network_or_sleeps_are_needed(monkeypatch):
    import socket
    import time

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network_or_sleep_used")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    assert check.run_private_learning_check().passed


@pytest.mark.parametrize(
    "name", ["lifecycle", "update", "suppression", "forbidden_fields"]
)
def test_real_contract_interventions_are_local_to_the_expected_check(name, monkeypatch):
    if name == "lifecycle":
        monkeypatch.setattr(
            check.FederatedRoundLifecycle, "can_transition_to", lambda *_: True
        )
    elif name == "update":
        original = check.FederatedUpdateMetadata.from_dict
        metadata, _policy = check._update()

        def accept_wrong_count(_cls, payload, *, policy):
            return (
                metadata
                if payload.get("total_elements") == 5
                else original(payload, policy=policy)
            )

        monkeypatch.setattr(
            check.FederatedUpdateMetadata, "from_dict", classmethod(accept_wrong_count)
        )
    elif name == "suppression":
        original = check.build_federated_metric_envelope

        def release_small_group(**kwargs):
            if kwargs["participant_count"] == 1:
                kwargs["participant_count"] = 100
            return original(**kwargs)

        monkeypatch.setattr(
            check, "build_federated_metric_envelope", release_small_group
        )
    else:
        original = check.FederatedRoundLifecycle.from_dict

        def accept_forbidden_field(_cls, payload):
            return original({k: v for k, v in payload.items() if k != "patient_count"})

        monkeypatch.setattr(
            check.FederatedRoundLifecycle,
            "from_dict",
            classmethod(accept_forbidden_field),
        )
    assert [
        result.name
        for result in check.run_private_learning_check().checks
        if not result.passed
    ] == [name]


def test_unknown_result_metadata_and_duplicate_checks_cannot_be_exported():
    report = check.run_private_learning_check()
    with pytest.raises(ValueError, match="invalid private-learning check result"):
        check.PrivateLearningCheckResult(
            "private raw value", True, 1, "sha256:" + "a" * 64
        )
    with pytest.raises(ValueError, match="invalid private-learning check report"):
        check.PrivateLearningCheckReport((report.checks[0],) * 7)


def test_cli_unexpected_failure_has_no_retained_private_exception_context(monkeypatch):
    import argparse

    cli = importlib.import_module("openmed.cli.private_learning_check")

    def fail():
        raise RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")

    monkeypatch.setattr(check, "run_private_learning_check", fail)
    with pytest.raises(cli.CliError) as error:
        cli.run_from_args(
            argparse.Namespace(json_output=True, command_path="training self-check")
        )
    assert error.value.__context__ is None
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in str(error.value)


def test_installed_console_entrypoint_runs_the_registered_command(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    executable = Path(sys.executable).with_name("openmed")
    environment = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(
        [str(executable), "training", "self-check", "--json"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["command"] == "training self-check"
    assert envelope["data"]["passed"] is True
    assert result.stderr == ""
    repeated = subprocess.run(
        [str(executable), "training", "self-check", "--json"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert repeated.returncode == 0
    assert repeated.stdout == result.stdout


def test_valid_format_wrong_fingerprint_fails_golden_oracle_independently(monkeypatch):
    monkeypatch.setattr(
        check, "fingerprint_update_schema", lambda _: "sha256:" + "0" * 64
    )
    report = check.run_private_learning_check()
    assert [item.name for item in report.checks if not item.passed] == ["fingerprint"]
    assert len(report.checks) == 7
    assert report.passed is False
