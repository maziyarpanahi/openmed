"""Tests for the reference v4.0 service recovery drill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.eval import service_recovery_drill as drill
from openmed.eval.service_release import (
    load_service_recovery_drill_schema,
    scan_for_sentinels,
)
from tests.fixtures import service_release as fx


def _run(**kwargs: Any) -> dict[str, Any]:
    return drill.run_recovery_drill(
        candidate_commit=fx.CANDIDATE, generated_at=fx.GENERATED_AT, **kwargs
    )


def _steps(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (scenario["category"], row["phase"]): row
        for scenario in report["scenarios"]
        for row in scenario["steps"]
    }


def test_master_drill_exercises_available_recovery_and_records_gaps(
    tmp_path: Path,
) -> None:
    report = _run(work_dir=tmp_path / "drill")
    steps = _steps(report)

    assert report["state"] == drill.DRILL_INCOMPLETE
    assert [row["category"] for row in report["scenarios"]] == list(
        drill.DRILL_CATEGORIES
    )
    scenario_states = {row["category"]: row["state"] for row in report["scenarios"]}
    assert scenario_states[drill.STORAGE] == drill.DRILL_COMPLETE
    assert scenario_states[drill.PROCESS] == drill.DRILL_COMPLETE
    assert scenario_states[drill.NETWORK] == drill.DRILL_COMPLETE
    unavailable = {
        key: (row["owner"], row["check"])
        for key, row in steps.items()
        if row["state"] == drill.STEP_NOT_AVAILABLE
    }
    assert unavailable == dict(drill.UNAVAILABLE_DRILL_STEPS)
    assert all(
        row["state"] == drill.STEP_PASSED
        for key, row in steps.items()
        if key not in unavailable
    )
    objectives = {row["id"]: row for row in report["objectives"]}
    for name in (
        "admissions_during_open_circuit",
        "corrupted_bytes_served",
        "lost_committed_records",
        "partial_writes_visible",
    ):
        assert objectives[name] == {
            "id": name,
            "observed": 0,
            "state": drill.OBJECTIVE_MET,
            "target": 0,
        }
    assert objectives["unverified_restored_records"]["state"] == "not_measured"
    assert report["summary"]["steps_not_available"] == 2
    assert drill.validate_recovery_drill_report(report) == ()
    Draft202012Validator(load_service_recovery_drill_schema()).validate(report)


def test_report_is_value_free_and_reproducible(tmp_path: Path) -> None:
    first = _run(work_dir=tmp_path / "one")
    second = _run(work_dir=tmp_path / "two")
    rendered = json.dumps(first, sort_keys=True)

    assert first == second
    assert first["report_digest"] == drill.recovery_drill_digest(first)
    assert str(tmp_path) not in rendered
    assert "job-drill" not in rendered and "synthetic recovery" not in rendered
    assert scan_for_sentinels(rendered) == ()


def test_injected_steps_complete_the_drill() -> None:
    report = _run(steps=fx.stand_in_drill_steps())
    assert report["state"] == drill.DRILL_COMPLETE
    assert report["summary"]["steps_passed"] == 24
    assert all(row["state"] == "met" for row in report["objectives"])


def test_storage_rollback_regression_is_detected(monkeypatch) -> None:
    from openmed.service.jobs import LocalJobStore

    def publish_without_rollback(self: Any, previous: Any) -> None:
        del previous
        self._persist_locked()

    monkeypatch.setattr(LocalJobStore, "_publish_locked", publish_without_rollback)
    report = _run()
    steps = _steps(report)

    assert steps[(drill.STORAGE, "contain")]["state"] == drill.STEP_FAILED
    assert steps[(drill.STORAGE, "contain")]["observed"] == {
        "partial_writes_visible": 1
    }
    objectives = {row["id"]: row for row in report["objectives"]}
    assert objectives["partial_writes_visible"]["state"] == "missed"
    assert report["state"] == drill.DRILL_FAILED


@pytest.mark.parametrize("operation", ["create", "verify", "restore"])
def test_storage_recovery_refusal_blocks_the_drill(monkeypatch, operation) -> None:
    from types import SimpleNamespace

    from openmed.service import recovery

    monkeypatch.setattr(
        recovery,
        f"{operation}_recovery_point",
        lambda *args, **kwargs: SimpleNamespace(completed=False),
    )
    report = _run()
    recovered = _steps(report)[(drill.STORAGE, "recover")]
    assert recovered["state"] == drill.STEP_FAILED
    assert recovered["observed"] == {"unverified_restored_records": 1}
    assert report["state"] == drill.DRILL_FAILED


def test_storage_recovery_does_not_trust_a_success_receipt_alone(monkeypatch) -> None:
    from types import SimpleNamespace

    from openmed.service import recovery

    monkeypatch.setattr(
        recovery,
        "restore_recovery_point",
        lambda *args, **kwargs: SimpleNamespace(completed=True),
    )
    report = _run()
    assert _steps(report)[(drill.STORAGE, "recover")]["state"] == drill.STEP_FAILED
    assert report["state"] == drill.DRILL_FAILED


def test_storage_recovery_reads_restored_bytes_and_preserves_live_state(tmp_path):
    from openmed.service.jobs import LocalJobStore
    from openmed.service.recovery import JOB_STORE_PATH

    root = tmp_path / "drill"
    report = _run(work_dir=root)
    assert _steps(report)[(drill.STORAGE, "recover")]["state"] == drill.STEP_PASSED
    original = LocalJobStore(root / "storage" / "live" / "jobs.json")
    restored = LocalJobStore(root / "storage" / "restored" / JOB_STORE_PATH)
    assert original.get("job-drill-pending") is None
    assert restored.get("job-drill-pending") is not None
    assert original.get("job-drill-committed") == restored.get("job-drill-committed")


@pytest.mark.parametrize("phase", ["detect", "contain"])
def test_process_drill_detects_missing_lease_or_quarantine_transition(
    monkeypatch, phase
):
    from openmed.service.jobs import DeidentifyJobQueue

    original = DeidentifyJobQueue._reconcile_running

    def ignore_transition(self, record, now, counts):
        if phase == "detect" or record["recovery"]["attempt"] >= 2:
            return None
        return original(self, record, now, counts)

    monkeypatch.setattr(DeidentifyJobQueue, "_reconcile_running", ignore_transition)
    report = _run()
    assert _steps(report)[(drill.PROCESS, phase)]["state"] == drill.STEP_FAILED
    assert report["state"] == drill.DRILL_FAILED


def test_process_replay_must_preserve_the_checkpoint(monkeypatch):
    from openmed.service.jobs import DeidentifyJobQueue

    original = DeidentifyJobQueue._process

    def replay_from_beginning(self, lease, record):
        if record["recovery"]["execution"] == 2:
            record["recovery"]["completed_documents"] = 0
        return original(self, lease, record)

    monkeypatch.setattr(DeidentifyJobQueue, "_process", replay_from_beginning)
    report = _run()
    assert _steps(report)[(drill.PROCESS, "recover")]["state"] == drill.STEP_FAILED
    assert report["state"] == drill.DRILL_FAILED


def test_process_replay_must_refuse_unauthorized_approval(monkeypatch):
    from openmed.service.jobs import DeidentifyJobQueue

    original = DeidentifyJobQueue.replay

    def ignore_authorization(self, job, approval):
        self.replay_authorizer = lambda *args: True
        return original(self, job, approval)

    monkeypatch.setattr(DeidentifyJobQueue, "replay", ignore_authorization)
    report = _run()
    assert _steps(report)[(drill.PROCESS, "recover")]["state"] == drill.STEP_FAILED
    assert report["state"] == drill.DRILL_FAILED


def test_corrupted_bytes_served_regression_is_detected(monkeypatch) -> None:
    from openmed.structured.store.local import LocalArtifactStore
    from openmed.structured.store.protocols import StoreResult

    monkeypatch.setattr(
        LocalArtifactStore,
        "get_bytes",
        lambda self, content_hash: StoreResult.success(b"served"),
    )
    report = _run()
    detect = _steps(report)[(drill.CORRUPTED_STATE, "detect")]

    assert detect["state"] == drill.STEP_FAILED
    assert detect["observed"] == {"corrupted_bytes_served": 1}
    assert report["state"] == drill.DRILL_FAILED


def test_open_circuit_admission_regression_is_detected(monkeypatch) -> None:
    from openmed.service.resilience import ResilienceManager

    monkeypatch.setattr(
        ResilienceManager,
        "execute",
        lambda self, key, operation: operation(),
        raising=True,
    )
    report = _run()
    steps = _steps(report)
    assert steps[(drill.NETWORK, "contain")]["state"] == drill.STEP_FAILED
    assert {row["id"]: row for row in report["objectives"]}[
        "admissions_during_open_circuit"
    ]["state"] == "missed"


def test_raising_or_invalid_steps_fail_without_leaking_text() -> None:
    def raising(context: drill.DrillContext) -> drill.DrillStepOutcome:
        raise RuntimeError(f"leaked {context.work_dir}")

    def invalid(context: drill.DrillContext) -> Any:
        del context
        return drill.DrillStepOutcome("passed", "Not A Code")

    report = _run(
        steps={(drill.NETWORK, "verify"): raising, (drill.NETWORK, "resume"): invalid}
    )
    steps = _steps(report)

    assert steps[(drill.NETWORK, "verify")]["check"] == "step_raised"
    assert steps[(drill.NETWORK, "resume")]["check"] == "step_result_invalid"
    assert report["state"] == drill.DRILL_FAILED
    assert "leaked" not in json.dumps(report)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"candidate_commit": "HEAD"}, "candidate_commit_invalid"),
        ({"generated_at": "2026-11-08 00:00"}, "generated_at_invalid"),
        ({"steps": {("network", "repair"): lambda _c: None}}, "drill_step_key"),
        ({"steps": {("network", "detect"): "not callable"}}, "drill_step_key"),
    ],
)
def test_invalid_arguments_are_refused(kwargs: dict[str, Any], code: str) -> None:
    options = {"candidate_commit": fx.CANDIDATE, "generated_at": fx.GENERATED_AT}
    options.update(kwargs)
    with pytest.raises(drill.RecoveryDrillError, match=code):
        drill.run_recovery_drill(**options)


def test_non_empty_work_dir_is_refused(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("x")
    with pytest.raises(drill.RecoveryDrillError, match="work_dir_not_empty"):
        _run(work_dir=tmp_path)


def test_not_available_requires_controlled_owner_and_check() -> None:
    with pytest.raises(drill.RecoveryDrillError):
        drill.not_available("3577", "encrypted_backup_restore")
    with pytest.raises(drill.RecoveryDrillError):
        drill.not_available("#3577", "Free text")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda r: r.update(report_digest="sha256:" + "0" * 64),
            "drill_digest_mismatch",
        ),
        (lambda r: r.update(schema="other"), "unknown_schema"),
        (lambda r: r.update(extra=True), "drill_report_invalid"),
        (lambda r: r["scenarios"].pop(), "drill_category_missing:corrupted_state"),
        (lambda r: r["scenarios"][0]["steps"].reverse(), "drill_report_invalid"),
        (
            lambda r: r["scenarios"][0]["steps"][0].update(owner="#1"),
            "drill_report_invalid",
        ),
        (lambda r: r["objectives"].pop(), "drill_report_invalid"),
        (lambda r: r.update(generated_at="yesterday"), "drill_report_invalid"),
        (lambda r: r["summary"].update(steps_passed=-1), "drill_report_invalid"),
    ],
)
def test_validator_rejects_tampered_reports(mutate: Any, code: str) -> None:
    report = _run()
    mutate(report)
    assert code in drill.validate_recovery_drill_report(report)


def test_cli_writes_report_and_refuses_overwrite(tmp_path: Path, capsys) -> None:
    output = tmp_path / "drill.json"
    argv = [
        "--output",
        str(output),
        "--candidate",
        fx.CANDIDATE,
        "--generated-at",
        fx.GENERATED_AT,
    ]
    assert drill.main(argv) == 1
    report = json.loads(output.read_text())
    assert drill.validate_recovery_drill_report(report) == ()
    printed = capsys.readouterr().out
    assert "not_available" in printed and "(#3577)" in printed
    assert str(tmp_path) not in printed

    assert drill.main(argv) == 2
    assert "output_exists" in capsys.readouterr().err
    assert drill.main([*argv, "--overwrite", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["state"] == "incomplete"
    assert drill.main(["--output", str(output), "--candidate", "HEAD"]) == 2


def test_validator_rederives_states_instead_of_trusting_the_summary() -> None:
    report = _run()
    report["state"] = drill.DRILL_COMPLETE
    report["summary"]["steps_not_available"] = 0
    report["objectives"] = [
        {**row, "observed": 0, "state": "met"} for row in report["objectives"]
    ]
    report["report_digest"] = drill.recovery_drill_digest(report)
    assert drill.validate_recovery_drill_report(report) == (
        "drill_report_inconsistent",
    )
