from __future__ import annotations

import json
from pathlib import Path

from openmed.cli import main_module
from openmed.core.audit import manifest_hash
from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.release_gates import (
    RELEASABLE,
    GateCheck,
    GateReport,
    apply_flakiness_quarantine,
)


def _write(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _artifact(path: Path, artifact_id: str, gates: list[str]) -> dict[str, object]:
    return {"artifact_id": artifact_id, "path": str(path), "gates": gates}


def _gate_report(artifacts: list[dict[str, object]]) -> GateReport:
    by_gate: dict[str, list[dict[str, object]]] = {gate: [] for gate in _GATES}
    for artifact in artifacts:
        for gate in artifact["gates"]:
            by_gate.setdefault(str(gate), []).append(artifact)

    return GateReport(
        repo_id="OpenMed/unit-model",
        family="PII",
        tier="Tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"PERSON": 0.995, "DATE": 0.995, "API_KEY": 0.995},
        per_label_precision={"PERSON": 0.98, "DATE": 0.98, "API_KEY": 0.99},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=50.0,
        p95_ms=120.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=RELEASABLE,
        gate_results=tuple(
            GateCheck(gate, True, details={"evidence": by_gate.get(gate, [])})
            for gate in _GATES
        ),
    )


_GATES = ("G1a", "G1b", "G2", "G3", "G4", "G5", "G6", "G7", "G8")


def _complete_artifacts(tmp_path: Path) -> list[dict[str, object]]:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    return [
        _artifact(
            _write(evidence_dir / "eval-set.jsonl", "{}\n"),
            "eval_set",
            ["G1a", "G1b", "G2", "G7"],
        ),
        _artifact(
            _write(evidence_dir / "leakage-fixtures.json", "[]\n"),
            "leakage_fixtures",
            ["G3", "G7"],
        ),
        _artifact(
            _write(evidence_dir / "quant-delta.json", '{"max_delta": 0.0}\n'),
            "quant_recall_delta",
            ["G4"],
        ),
        _artifact(
            _write(evidence_dir / "performance.json", '{"p50_ms": 50}\n'),
            "performance_report",
            ["G5", "G6"],
        ),
        _artifact(
            _write(evidence_dir / "baseline.json", '{"entries": {}}\n'),
            "baseline_store",
            ["G7"],
        ),
        _artifact(
            _write(evidence_dir / "span-fixtures.json", "[]\n"),
            "span_fixtures",
            ["G8"],
        ),
    ]


def test_complete_evidence_bundle_hashes_artifacts_and_covers_g1_g8(
    tmp_path: Path,
) -> None:
    artifacts = _complete_artifacts(tmp_path)
    report = _gate_report(artifacts)

    result = bundle_gate_evidence(report, tmp_path / "bundle")

    assert result.missing_required == ()
    assert result.manifest_path.is_file()
    assert result.summary_path.is_file()
    assert len(result.manifest["artifacts"]) == len(artifacts)
    assert all(result.manifest["gates"][gate]["status"] == "covered" for gate in _GATES)

    eval_entry = next(
        entry
        for entry in result.manifest["artifacts"]
        if entry["artifact_id"] == "eval_set"
    )
    assert eval_entry["sha256"] == manifest_hash(Path(artifacts[0]["path"]))
    assert (tmp_path / "bundle" / eval_entry["bundle_path"]).is_file()


def test_missing_required_artifact_is_manifested_with_affected_gate(
    tmp_path: Path,
) -> None:
    artifacts = _complete_artifacts(tmp_path)
    missing = tmp_path / "evidence" / "missing-span-fixtures.json"
    artifacts[-1] = _artifact(missing, "span_fixtures", ["G8"])

    result = bundle_gate_evidence(_gate_report(artifacts), tmp_path / "bundle")

    missing_entries = [
        entry for entry in result.manifest["artifacts"] if entry["status"] == "missing"
    ]
    assert missing_entries == [
        {
            "artifact_id": "span_fixtures",
            "artifact_ids": ["span_fixtures"],
            "bundle_path": None,
            "description": "",
            "gates": ["G8"],
            "required": True,
            "sha256": None,
            "source_path": str(missing),
            "status": "missing",
        }
    ]
    assert result.manifest["gates"]["G8"]["status"] == "missing"
    assert result.manifest["gates"]["G8"]["missing_artifacts"] == ["span_fixtures"]


def test_manifest_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    report = _gate_report(_complete_artifacts(tmp_path))

    first = bundle_gate_evidence(report, tmp_path / "bundle-a")
    second = bundle_gate_evidence(report, tmp_path / "bundle-b")

    assert first.manifest == second.manifest
    assert first.manifest_path.read_text(encoding="utf-8") == (
        second.manifest_path.read_text(encoding="utf-8")
    )


def test_stability_summary_is_carried_into_evidence_bundle(tmp_path: Path) -> None:
    report = apply_flakiness_quarantine(
        _gate_report(_complete_artifacts(tmp_path)),
        {
            "quarantined_gates": ["G1a"],
            "unstable_gates": ["G1a"],
            "verdict": "quarantined",
        },
    )

    result = bundle_gate_evidence(report, tmp_path / "bundle")

    assert result.manifest["gate_report"]["stability_summary"]["verdict"] == (
        "quarantined"
    )
    assert result.manifest["summary"]["stability_verdict"] == "quarantined"
    assert result.manifest["summary"]["quarantined_stability_gates"] == ["G1a"]
    assert "Stability: quarantined" in result.summary


def test_gates_bundle_cli_is_strict_only_for_missing_required_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _complete_artifacts(tmp_path)
    artifacts[-1] = _artifact(
        tmp_path / "evidence" / "missing-span-fixtures.json",
        "span_fixtures",
        ["G8"],
    )
    report_path = tmp_path / "gate-report.json"
    report_path.write_text(
        json.dumps(_gate_report(artifacts).to_dict()),
        encoding="utf-8",
    )

    non_strict = main_module.main(
        [
            "gates",
            "bundle",
            "--gate-report",
            str(report_path),
            "--output-dir",
            str(tmp_path / "non-strict"),
        ]
    )
    strict = main_module.main(
        [
            "gates",
            "bundle",
            "--gate-report",
            str(report_path),
            "--output-dir",
            str(tmp_path / "strict"),
            "--strict",
        ]
    )

    assert non_strict == 0
    assert strict == 1
