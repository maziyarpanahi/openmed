"""Focused synthetic tests for the nightly release orchestrator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateReport

ROOT = Path(__file__).resolve().parents[3]
FIXED_SHA = "0123456789abcdef0123456789abcdef01234567"
FIXED_TIME = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


orchestrate = _load_script(
    "nightly_release_orchestrate",
    "scripts/release/orchestrate.py",
)
smoke_test = _load_script(
    "nightly_release_smoke",
    "scripts/release/smoke_test.py",
)


def _candidate(seed: str) -> orchestrate.NightlyCandidate:
    return orchestrate.NightlyCandidate(
        candidate_id=f"candidate-{seed}",
        weekday="monday",
        theme="synthetic",
        source_model_id=f"OpenMed/source-{seed}",
        repo_id=f"OpenMed/target-{seed}",
        family="PII",
        tier="Small",
        param_count=44_000_000,
        format="onnx",
        fixture_path="synthetic-fixture.json",
        suite="synthetic-pii",
        device="cpu",
    )


def _report(
    candidate: orchestrate.NightlyCandidate,
    decision: str = RELEASABLE,
) -> GateReport:
    return GateReport(
        repo_id=candidate.repo_id,
        family=candidate.family,
        tier=candidate.tier,
        param_count=candidate.param_count,
        format=candidate.format,
        per_label_recall={"PERSON": 0.99},
        per_label_precision={"PERSON": 0.98},
        critical_leakage_count=0 if decision == RELEASABLE else 1,
        residual_leakage_rate=0.0 if decision == RELEASABLE else 1.0,
        quant_recall_delta=0.0,
        p50_ms=1.0,
        p95_ms=2.0,
        ram_mb=32.0,
        eval_set_hash="sha256:synthetic-eval",
        leakage_fixture_hash="sha256:synthetic-leakage",
        decision=decision,
    ).sign("synthetic-key")


class _FakeRuntime:
    def __init__(
        self,
        root: Path,
        *,
        decisions: dict[str, str] | None = None,
        failures: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.decisions = decisions or {}
        self.failures = failures or {}
        self.calls: list[tuple[str, str]] = []
        self.quarantines: list[tuple[str, str]] = []
        self.latest: dict[str, str] = {}

    def _call(self, stage: str, candidate: orchestrate.NightlyCandidate) -> None:
        self.calls.append((stage, candidate.candidate_id))
        if self.failures.get(candidate.candidate_id) == stage:
            raise RuntimeError("Patient Example 123-45-6789")

    def build(self, candidate: orchestrate.NightlyCandidate) -> Path:
        self._call("build", candidate)
        path = self.root / "artifacts" / candidate.candidate_id
        path.mkdir(parents=True)
        (path / "model.onnx").write_bytes(candidate.candidate_id.encode())
        return path

    def artifact_digest(self, artifact_dir: Path) -> str:
        seed = artifact_dir.name.encode()
        import hashlib

        return f"sha256:{hashlib.sha256(seed).hexdigest()}"

    def evaluate(self, candidate, artifact_dir, *, generated_at):
        del artifact_dir, generated_at
        self._call("eval", candidate)
        return {"candidate": candidate.candidate_id}

    def gate(self, candidate, benchmark_report):
        del benchmark_report
        self._call("gate", candidate)
        return _report(
            candidate, self.decisions.get(candidate.candidate_id, RELEASABLE)
        )

    def build_card(self, candidate, artifact_dir, report, *, git_sha, released):
        del artifact_dir, report, git_sha, released
        self._call("model-card", candidate)

    def publish(self, candidate, artifact_dir, report_path, *, git_sha, released):
        del artifact_dir, report_path, git_sha, released
        self._call("publish", candidate)

    def promote(self, candidate, report):
        del report
        self._call("promote", candidate)
        self.latest[candidate.family] = candidate.repo_id
        return candidate.repo_id

    def smoke(self, candidate):
        self._call("smoke", candidate)

    def mark_last_green(self, candidate, report):
        del report
        self._call("last-green", candidate)
        self.latest[candidate.family] = candidate.repo_id
        return candidate.repo_id

    def rollback(self, candidate):
        self._call("rollback", candidate)
        target = "OpenMed/last-green"
        self.latest[candidate.family] = target
        return target

    def failure_report(self, candidate, *, stage):
        self.calls.append(("failure-report", candidate.candidate_id))
        return _report(candidate, QUARANTINED)

    def report_quarantine(
        self,
        candidate,
        *,
        run_id,
        git_sha,
        stage,
        gate_report_hash,
    ):
        del run_id, git_sha, gate_report_hash
        self.quarantines.append((candidate.candidate_id, stage))


def _run(
    tmp_path: Path,
    candidates,
    runtime: _FakeRuntime,
    *,
    run_id: str = "run-1",
):
    reports = tmp_path / "gates" / "release_reports"
    ledger = tmp_path / "gates" / "release_runs.jsonl"
    results = orchestrate.orchestrate_nightly(
        candidates,
        run_id=run_id,
        git_sha=FIXED_SHA,
        runtime=runtime,
        ledger_path=ledger,
        reports_dir=reports,
        clock=lambda: FIXED_TIME,
    )
    return results, ledger


def test_releasable_candidate_runs_every_stage_and_audits_offline(
    tmp_path: Path,
) -> None:
    candidate = _candidate("green")
    runtime = _FakeRuntime(tmp_path)

    results, ledger = _run(tmp_path, [candidate], runtime)

    assert [stage for stage, _ in runtime.calls] == [
        "build",
        "eval",
        "gate",
        "model-card",
        "publish",
        "promote",
        "smoke",
        "last-green",
    ]
    assert results[0].outcome == orchestrate.OUTCOME_PUBLISHED
    outcome = orchestrate.audit_nightly_run(
        "run-1",
        ledger_path=ledger,
        repository_root=tmp_path,
    )
    assert outcome[candidate.candidate_id]["published"] is True
    assert outcome[candidate.candidate_id]["smoke_test"] == orchestrate.SMOKE_PASSED


def test_quarantined_candidate_never_publishes_and_batch_continues(
    tmp_path: Path,
) -> None:
    first = _candidate("first")
    blocked = _candidate("blocked")
    last = _candidate("last")
    runtime = _FakeRuntime(
        tmp_path,
        decisions={blocked.candidate_id: QUARANTINED},
    )

    results, ledger = _run(tmp_path, [first, blocked, last], runtime)

    by_id = {result.candidate.candidate_id: result for result in results}
    assert by_id[blocked.candidate_id].outcome == orchestrate.OUTCOME_QUARANTINED
    assert ("model-card", blocked.candidate_id) not in runtime.calls
    assert ("publish", blocked.candidate_id) not in runtime.calls
    assert ("publish", last.candidate_id) in runtime.calls
    assert runtime.quarantines == [(blocked.candidate_id, "gate")]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert {row["run_status"] for row in rows} == {orchestrate.RUN_PARTIAL}
    assert all(orchestrate.verify_record(row) for row in rows)


def test_smoke_failure_flips_latest_to_last_green_and_records_rollback(
    tmp_path: Path,
) -> None:
    candidate = _candidate("smoke-fail")
    runtime = _FakeRuntime(
        tmp_path,
        failures={candidate.candidate_id: "smoke"},
    )

    results, ledger = _run(tmp_path, [candidate], runtime)

    result = results[0]
    assert ("rollback", candidate.candidate_id) in runtime.calls
    assert runtime.latest["PII"] == "OpenMed/last-green"
    assert result.outcome == orchestrate.OUTCOME_ROLLED_BACK
    assert result.smoke_test == orchestrate.SMOKE_FAILED
    assert result.pointer_target == "OpenMed/last-green"
    outcome = orchestrate.audit_nightly_run(
        "run-1",
        ledger_path=ledger,
        repository_root=tmp_path,
    )
    assert outcome[candidate.candidate_id]["published"] is False
    assert outcome[candidate.candidate_id]["pointer_target"] == "OpenMed/last-green"


def test_failure_text_never_reaches_logs_ledger_or_gate_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = _candidate("private")
    runtime = _FakeRuntime(
        tmp_path,
        failures={candidate.candidate_id: "build"},
    )

    results, ledger = _run(tmp_path, [candidate], runtime)

    assert results[0].artifact_digest is None
    captured = capsys.readouterr()
    persisted = ledger.read_text(encoding="utf-8") + "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "gates" / "release_reports").rglob("*.json")
    )
    assert "Patient Example" not in captured.out + captured.err + persisted
    assert "123-45-6789" not in captured.out + captured.err + persisted


def test_audit_rejects_tampered_gate_report(tmp_path: Path) -> None:
    candidate = _candidate("tamper")
    runtime = _FakeRuntime(tmp_path)
    _, ledger = _run(tmp_path, [candidate], runtime)
    report_path = next((tmp_path / "gates" / "release_reports").rglob("*.json"))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["family"] = "forged"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(orchestrate.ReleaseManifestError):
        orchestrate.audit_nightly_run(
            "run-1",
            ledger_path=ledger,
            repository_root=tmp_path,
        )


def test_committed_queue_has_two_reviewed_candidates_per_weekday() -> None:
    manifest_repo_ids = {
        json.loads(line)["repo_id"]
        for line in (ROOT / "models.jsonl").read_text().splitlines()
        if line.strip()
    }
    for weekday in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        candidates = orchestrate.load_nightly_queue(
            ROOT / "gates" / "nightly_release_queue.json",
            weekday=weekday,
        )
        assert len(candidates) == 2
        assert all(candidate.weekday == weekday for candidate in candidates)
        assert all(
            candidate.source_model_id in manifest_repo_ids for candidate in candidates
        )


def test_workflow_schedules_batch_guards_publish_and_commits_audit_state() -> None:
    workflow = (ROOT / ".github" / "workflows" / "nightly-release.yml").read_text()

    assert "cron: '17 2 * * 1-5'" in workflow
    assert "environment:\n      name: hf-publish" in workflow
    assert "HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}" in workflow
    assert (
        "OPENMED_RELEASE_GATE_KEY: ${{ secrets.OPENMED_RELEASE_GATE_KEY }}" in workflow
    )
    assert "continue-on-error: true" in workflow
    assert "python scripts/release/orchestrate.py run" in workflow
    assert "gates/release_runs.jsonl" in workflow
    assert "gates/release_reports/" in workflow
    assert "peter-evans/create-pull-request@v8" in workflow
    assert "steps.orchestrate.outcome != 'success'" in workflow


def test_fresh_venv_smoke_installs_downloads_and_probes_without_output(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    smoke_test.run_fresh_venv_smoke(
        "OpenMed/synthetic-model",
        format_name="onnx",
        repository_root=tmp_path,
        runner=runner,
    )

    assert len(calls) == 3
    assert calls[0][1:3] == ["-m", "venv"]
    assert "pip" in calls[1]
    assert f"{tmp_path.resolve()}[onnx-runtime]" in calls[1]
    assert "--probe" in calls[2]
    assert "--download-dir" in calls[2]


def test_fresh_venv_smoke_sanitizes_child_failure_output(tmp_path: Path) -> None:
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return subprocess.CompletedProcess(
            command,
            1 if calls == 3 else 0,
            stdout=b"Patient Example 123-45-6789",
            stderr=b"private model output",
        )

    with pytest.raises(smoke_test.SmokeTestError) as exc_info:
        smoke_test.run_fresh_venv_smoke(
            "OpenMed/synthetic-model",
            format_name="onnx",
            repository_root=tmp_path,
            runner=runner,
        )

    assert "Patient Example" not in str(exc_info.value)
    assert "123-45-6789" not in str(exc_info.value)


def test_artifact_probe_calls_extract_and_deidentify_and_asserts_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openmed.core.models as models_module
    import openmed.core.pii as pii_module

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    note = smoke_test._SYNTHETIC_NOTE
    phone = smoke_test._SYNTHETIC_PHONE
    start = note.index(phone)
    calls: list[str] = []

    monkeypatch.setattr(models_module, "ModelLoader", lambda config: object())

    def fake_extract(text, **kwargs):
        del kwargs
        calls.append("extract_pii")
        assert text == note
        return SimpleNamespace(
            entities=[
                SimpleNamespace(start=start, end=start + len(phone), label="PHONE")
            ]
        )

    def fake_deidentify(text, **kwargs):
        del kwargs
        calls.append("deidentify")
        assert text == note
        return SimpleNamespace(deidentified_text="Synthetic patient contact [PHONE].")

    monkeypatch.setattr(pii_module, "extract_pii", fake_extract)
    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)

    result = smoke_test.probe_artifact(artifact, format_name="onnx")

    assert calls == ["extract_pii", "deidentify"]
    assert result.span_count == 1
    assert result.span_offsets_hash.startswith("sha256:")
