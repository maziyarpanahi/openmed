"""Tests for the fail-closed release-readiness gate (#1814)."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
)
from openmed.eval.release_readiness import (
    NOT_READY,
    READY,
    ReadinessReport,
    evaluate_readiness,
    main,
)

GATE_KEY = "test-release-gate-key"
READINESS_KEY = "test-readiness-key"


def _make_gate_report(*, decision: str = RELEASABLE) -> GateReport:
    gate_results: tuple[GateCheck, ...] = ()
    if decision != RELEASABLE:
        gate_results = (GateCheck("g1a_recall", False, reason="Recall 0.900 < 0.995"),)
    return GateReport(
        repo_id="test/model",
        family="pii",
        tier="t1",
        param_count=100,
        format="pytorch",
        per_label_recall={"PERSON": 0.999},
        per_label_precision={"PERSON": 0.999},
        critical_leakage_count=0 if decision == RELEASABLE else 5,
        residual_leakage_rate=0.001 if decision == RELEASABLE else 0.05,
        quant_recall_delta=None,
        p50_ms=10.0,
        p95_ms=20.0,
        ram_mb=256.0,
        eval_set_hash="abc123",
        leakage_fixture_hash="def456",
        decision=decision,
        gate_results=gate_results,
    ).sign(GATE_KEY)


def _make_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# OpenMed\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")

    migration = root / "docs" / "migration" / "1.9-to-2.0.md"
    migration.parent.mkdir(parents=True)
    migration.write_text("# Migration Guide\n", encoding="utf-8")

    gates = root / "gates"
    gates.mkdir()
    (gates / "api_compat_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "before_ref": "v1.9.0",
                "after_ref": "v2.0.0",
                "summary": {
                    "before_symbols": 100,
                    "after_symbols": 105,
                    "added": 5,
                    "deprecated": 0,
                    "breaking": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (gates / "e2e_golden_pass.json").write_text(
        json.dumps({"passed": True, "suite": "golden_v2"}),
        encoding="utf-8",
    )

    clinical = root / "openmed" / "clinical"
    clinical.mkdir(parents=True)
    (clinical / "__init__.py").write_text(
        "OPENMED_CLINICAL_DISCLAIMER = 'Assistive output; not a medical device.'\n"
        "__all__ = ['OPENMED_CLINICAL_DISCLAIMER']\n",
        encoding="utf-8",
    )
    return root


def _evaluate(root: Path, **kwargs) -> ReadinessReport:
    return evaluate_readiness(
        repo_root=root,
        gate_report=_make_gate_report(),
        gate_report_key=GATE_KEY,
        signing_key=READINESS_KEY,
        **kwargs,
    )


def test_ready_when_all_signed_inputs_are_satisfied(tmp_path):
    report = _evaluate(_make_repo_root(tmp_path))

    assert report.decision == READY
    assert len(report.checks) == 5
    assert all(check.passed for check in report.checks)


def test_report_signature_covers_hash_decision_and_checks(tmp_path):
    report = _evaluate(_make_repo_root(tmp_path))

    assert report.verify(READINESS_KEY)
    assert not report.verify("wrong-key")

    report.checks = (
        GateCheck("extraction_gates", False, reason="tampered"),
        *report.checks[1:],
    )
    assert not report.verify(READINESS_KEY)


def test_repro_hash_is_stable(tmp_path):
    root = _make_repo_root(tmp_path)

    first = _evaluate(root)
    second = _evaluate(root)

    assert first.repro_hash == second.repro_hash


def test_unsigned_extraction_report_fails_closed(tmp_path):
    root = _make_repo_root(tmp_path)
    unsigned = _make_gate_report()
    unsigned.signature = None

    report = evaluate_readiness(
        repo_root=root,
        gate_report=unsigned,
        gate_report_key=GATE_KEY,
    )

    assert report.decision == NOT_READY
    assert report.failing_checks()[0].gate == "extraction_gates"
    assert "signature" in report.failing_checks()[0].reason


def test_tampered_extraction_report_fails_closed(tmp_path):
    root = _make_repo_root(tmp_path)
    gate_report = _make_gate_report()
    gate_report.per_label_recall = {"PERSON": 0.1}

    report = evaluate_readiness(
        repo_root=root,
        gate_report=gate_report,
        gate_report_key=GATE_KEY,
    )

    assert report.decision == NOT_READY
    assert report.failing_checks()[0].gate == "extraction_gates"


def test_missing_gate_report_fails_closed(tmp_path):
    report = evaluate_readiness(repo_root=_make_repo_root(tmp_path))

    assert report.decision == NOT_READY
    assert report.failing_checks()[0].gate == "extraction_gates"


def test_quarantined_extraction_gate_propagates(tmp_path):
    report = evaluate_readiness(
        repo_root=_make_repo_root(tmp_path),
        gate_report=_make_gate_report(decision=QUARANTINED),
        gate_report_key=GATE_KEY,
    )

    assert report.decision == NOT_READY
    check = next(
        item for item in report.failing_checks() if item.gate == "extraction_gates"
    )
    assert "QUARANTINED" in check.reason
    assert check.details["failing_gates"] == ["g1a_recall"]


def test_missing_migration_guide_fails_closed(tmp_path):
    root = _make_repo_root(tmp_path)
    (root / "docs" / "migration" / "1.9-to-2.0.md").unlink()

    report = _evaluate(root)

    assert report.decision == NOT_READY
    check = next(
        item for item in report.failing_checks() if item.gate == "required_docs"
    )
    assert "docs/migration/1.9-to-2.0.md" in check.reason


def test_disclaimer_comment_does_not_satisfy_gate(tmp_path):
    root = _make_repo_root(tmp_path)
    (root / "openmed" / "clinical" / "__init__.py").write_text(
        "# OPENMED_CLINICAL_DISCLAIMER is intentionally absent\n",
        encoding="utf-8",
    )

    report = _evaluate(root)

    assert report.decision == NOT_READY
    check = next(
        item for item in report.failing_checks() if item.gate == "clinical_disclaimer"
    )
    assert "non-empty string" in check.reason


def test_disclaimer_must_be_publicly_exported(tmp_path):
    root = _make_repo_root(tmp_path)
    (root / "openmed" / "clinical" / "__init__.py").write_text(
        "OPENMED_CLINICAL_DISCLAIMER = 'Not a medical device.'\n__all__ = []\n",
        encoding="utf-8",
    )

    report = _evaluate(root)

    assert report.decision == NOT_READY
    check = next(
        item for item in report.failing_checks() if item.gate == "clinical_disclaimer"
    )
    assert "omit" in check.reason


def test_breaking_api_report_fails_closed(tmp_path):
    root = _make_repo_root(tmp_path)
    path = root / "gates" / "api_compat_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["breaking"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = _evaluate(root)

    assert report.decision == NOT_READY
    check = next(item for item in report.failing_checks() if item.gate == "api_compat")
    assert "2 breaking change" in check.reason


def test_non_boolean_or_unnamed_golden_result_fails_closed(tmp_path):
    root = _make_repo_root(tmp_path)
    (root / "gates" / "e2e_golden_pass.json").write_text(
        json.dumps({"passed": 1, "suite": ""}),
        encoding="utf-8",
    )

    report = _evaluate(root)

    assert report.decision == NOT_READY
    check = next(item for item in report.failing_checks() if item.gate == "e2e_golden")
    assert "named passing suite" in check.reason


def test_report_serialization_round_trip_remains_verifiable(tmp_path):
    original = _evaluate(_make_repo_root(tmp_path))

    restored = ReadinessReport.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.verify(READINESS_KEY)


def test_cli_writes_report_and_returns_success(tmp_path):
    root = _make_repo_root(tmp_path)
    gate_report_path = root / "release-gate-report.json"
    gate_report_path.write_text(
        json.dumps(_make_gate_report().to_dict()),
        encoding="utf-8",
    )
    output = root / "release-readiness-report.json"

    result = main(
        [
            "--repo-root",
            str(root),
            "--gate-report",
            str(gate_report_path),
            "--gate-report-key",
            GATE_KEY,
            "--signing-key",
            READINESS_KEY,
            "--output",
            str(output),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == READY


def test_release_workflow_gates_publish_on_readiness():
    workflow = Path(".github/workflows/release-gates.yml").read_text(encoding="utf-8")

    assert "Generate API compatibility evidence" in workflow
    assert "tests/integration/test_end_to_end.py" in workflow
    assert "Run release-readiness gate" in workflow
    assert "--gate-report release-gate-report.json" in workflow
    assert "--output release-readiness-report.json" in workflow
    assert "steps.readiness.outcome == 'success'" in workflow
    assert "steps.readiness.outcome != 'success'" in workflow
