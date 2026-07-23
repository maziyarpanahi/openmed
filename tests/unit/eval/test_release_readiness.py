"""Tests for the release-readiness gate (#1814)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openmed.core.audit import AuditSignature
from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateCheck, GateReport
from openmed.eval.release_readiness import (
    NOT_READY,
    READY,
    ReadinessReport,
    evaluate_readiness,
    _DISCLAIMER_CONSTANT as DISCLAIMER_CONSTANT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_releasable_report() -> GateReport:
    """Return a minimal RELEASABLE GateReport."""
    return GateReport(
        repo_id="test/model",
        family="pii",
        tier="t1",
        param_count=100,
        format="pytorch",
        per_label_recall={"PERSON": 0.999},
        per_label_precision={"PERSON": 0.999},
        critical_leakage_count=0,
        residual_leakage_rate=0.001,
        quant_recall_delta=None,
        p50_ms=10.0,
        p95_ms=20.0,
        ram_mb=256.0,
        eval_set_hash="abc123",
        leakage_fixture_hash="def456",
        decision=RELEASABLE,
    )


def _make_quarantined_report() -> GateReport:
    """Return a minimal QUARANTINED GateReport."""
    return GateReport(
        repo_id="test/model",
        family="pii",
        tier="t1",
        param_count=100,
        format="pytorch",
        per_label_recall={"PERSON": 0.900},
        per_label_precision={"PERSON": 0.900},
        critical_leakage_count=5,
        residual_leakage_rate=0.05,
        quant_recall_delta=None,
        p50_ms=10.0,
        p95_ms=20.0,
        ram_mb=256.0,
        eval_set_hash="abc123",
        leakage_fixture_hash="def456",
        decision=QUARANTINED,
        gate_results=(
            GateCheck("g1a_recall", False, reason="Recall 0.900 < 0.995"),
        ),
    )


def _make_repo_root(tmp_path: Path, *, include_docs=True, include_disclaimer=True,
                     include_baseline=True, include_e2e=True) -> Path:
    """Create a minimal repo structure for testing."""
    root = tmp_path / "repo"
    root.mkdir()

    if include_docs:
        (root / "README.md").write_text("# OpenMed\n")
        (root / "CHANGELOG.md").write_text("# Changelog\n")
        (root / "MIGRATION.md").write_text("# Migration Guide\n")

    if include_baseline:
        gates_dir = root / "gates"
        gates_dir.mkdir()
        (gates_dir / "baseline.json").write_text(
            json.dumps({"api_surface": {"exports": ["deid", "ner"]}}),
        )

    if include_e2e:
        gates_dir = root / "gates"
        gates_dir.mkdir(exist_ok=True)
        (gates_dir / "e2e_golden_pass.json").write_text(
            json.dumps({"passed": True, "suite": "golden_v2"}),
        )

    clinical_dir = root / "openmed" / "clinical"
    clinical_dir.mkdir(parents=True)
    disclaimer_content = f'{DISCLAIMER_CONSTANT} = "OpenMed is not a medical device."'
    (clinical_dir / "disclaimer.py").write_text(disclaimer_content + "\n")

    return root


# ---------------------------------------------------------------------------
# Tests: all checks pass
# ---------------------------------------------------------------------------


class TestReadinessGate:
    """Test the release-readiness gate evaluation."""

    def test_ready_when_all_inputs_satisfied(self, tmp_path):
        """All checks pass => READY."""
        repo_root = _make_repo_root(tmp_path)
        gate_report = _make_releasable_report()

        report = evaluate_readiness(
            repo_root=repo_root,
            gate_report=gate_report,
        )

        assert report.decision == READY
        assert all(c.passed for c in report.checks)
        assert len(report.checks) == 5

    def test_signed_and_verifiable(self, tmp_path):
        """Report is signed and verify() succeeds."""
        repo_root = _make_repo_root(tmp_path)
        gate_report = _make_releasable_report()
        key = "test-signing-key"

        report = evaluate_readiness(
            repo_root=repo_root,
            gate_report=gate_report,
            signing_key=key,
        )

        assert report.signature is not None
        assert report.verify(key)
        assert not report.verify("wrong-key")

    def test_repro_hash_is_stable(self, tmp_path):
        """Two evaluations with the same inputs produce the same repro_hash."""
        repo_root = _make_repo_root(tmp_path)
        gate_report = _make_releasable_report()

        r1 = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)
        r2 = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert r1.repro_hash == r2.repro_hash


# ---------------------------------------------------------------------------
# Tests: fail-closed on missing evidence
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Missing evidence yields NOT_READY, never a default-pass."""

    def test_not_ready_without_gate_report(self, tmp_path):
        """No gate report => NOT_READY (extraction_gates fails)."""
        repo_root = _make_repo_root(tmp_path)

        report = evaluate_readiness(repo_root=repo_root, gate_report=None)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        assert any(c.gate == "extraction_gates" for c in failing)

    def test_not_ready_when_migration_missing(self, tmp_path):
        """Missing MIGRATION.md => NOT_READY."""
        repo_root = _make_repo_root(tmp_path)
        (repo_root / "MIGRATION.md").unlink()
        gate_report = _make_releasable_report()

        report = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        docs_check = next(c for c in failing if c.gate == "required_docs")
        assert "MIGRATION.md" in docs_check.reason

    def test_not_ready_when_disclaimer_missing(self, tmp_path):
        """Missing disclaimer constant => NOT_READY."""
        repo_root = _make_repo_root(tmp_path)
        # Remove disclaimer file
        for f in (repo_root / "openmed" / "clinical").iterdir():
            if f.suffix == ".py":
                f.unlink()
        gate_report = _make_releasable_report()

        report = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        assert any(c.gate == "clinical_disclaimer" for c in failing)

    def test_not_ready_when_baseline_missing(self, tmp_path):
        """Missing API-compat baseline => NOT_READY."""
        repo_root = _make_repo_root(tmp_path, include_baseline=False)
        gate_report = _make_releasable_report()

        report = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        assert any(c.gate == "api_compat" for c in failing)

    def test_not_ready_when_e2e_missing(self, tmp_path):
        """Missing e2e golden marker => NOT_READY."""
        repo_root = _make_repo_root(tmp_path, include_e2e=False)
        gate_report = _make_releasable_report()

        report = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        assert any(c.gate == "e2e_golden" for c in failing)


# ---------------------------------------------------------------------------
# Tests: quarantined extraction gate propagates
# ---------------------------------------------------------------------------


class TestQuarantinedPropagation:
    """A QUARANTINED extraction gate propagates to NOT_READY."""

    def test_quarantined_gate_yields_not_ready(self, tmp_path):
        repo_root = _make_repo_root(tmp_path)
        gate_report = _make_quarantined_report()

        report = evaluate_readiness(repo_root=repo_root, gate_report=gate_report)

        assert report.decision == NOT_READY
        failing = report.failing_checks()
        ext_check = next(c for c in failing if c.gate == "extraction_gates")
        assert "QUARANTINED" in ext_check.reason


# ---------------------------------------------------------------------------
# Tests: serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    """ReadinessReport round-trips through to_dict/from_dict."""

    def test_round_trip(self, tmp_path):
        repo_root = _make_repo_root(tmp_path)
        gate_report = _make_releasable_report()

        original = evaluate_readiness(
            repo_root=repo_root,
            gate_report=gate_report,
            signing_key="round-trip-key",
        )

        data = original.to_dict()
        restored = ReadinessReport.from_dict(data)

        assert restored.version == original.version
        assert restored.decision == original.decision
        assert restored.repro_hash == original.repro_hash
        assert len(restored.checks) == len(original.checks)
        for orig_check, rest_check in zip(original.checks, restored.checks):
            assert rest_check.gate == orig_check.gate
            assert rest_check.passed == orig_check.passed
            assert rest_check.reason == orig_check.reason
        assert restored.signature is not None
        assert restored.verify("round-trip-key")
