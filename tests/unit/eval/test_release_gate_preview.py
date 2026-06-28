from __future__ import annotations

import json
from pathlib import Path

from openmed.cli import main_module
from openmed.eval import release_gates


def _candidate(*, critical_leakage_count: int = 0) -> dict[str, object]:
    return {
        "suite": "golden",
        "model_name": "unit-model",
        "device": "cpu",
        "fixture_count": 1,
        "generated_at": "2026-06-15T00:00:00+00:00",
        "metadata": {
            "repo_id": "OpenMed/unit-model",
            "family": "PII",
            "tier": "Tiny",
            "param_count": 44_000_000,
            "format": "mlx-fp",
            "eval_set_hash": "sha256:eval",
            "leakage_fixture_hash": "sha256:leakage",
            "policy": "hipaa_safe_harbor",
            "thresholds": {"PERSON": {"en": 0.9}},
            "calibration_report": {"schema_version": 1, "groups": []},
            "span_fixtures": [
                {
                    "text": "Patient John on 2026-01-02 has ID 123.",
                    "predicted_spans": [
                        {"start": 8, "end": 12, "label": "PERSON"},
                        {"start": 16, "end": 26, "label": "DATE"},
                        {"start": 34, "end": 37, "label": "ID_NUM"},
                    ],
                }
            ],
        },
        "metrics": {
            "per_label_recall": {
                "PERSON": 0.990,
                "DATE": 0.990,
                "ID_NUM": 0.990,
                "API_KEY": 0.995,
            },
            "per_label_precision": {
                "PERSON": 0.98,
                "DATE": 0.98,
                "ID_NUM": 0.98,
                "API_KEY": 0.99,
            },
            "critical_leakage_count": critical_leakage_count,
            "leakage": {
                "overall": 0.0,
                "leaked_chars_by_label": {},
                "total_chars_by_label": {
                    "PERSON": 4,
                    "DATE": 10,
                    "ID_NUM": 3,
                    "API_KEY": 8,
                },
            },
            "latency": {"p50_ms": 50.0, "p95_ms": 120.0},
            "resources": {"peak_rss_mib": 128.0},
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline_store(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "entries": {
            "pii::tiny::mlx-fp": {
                "key": "pii::tiny::mlx-fp",
                "family": "PII",
                "tier": "Tiny",
                "format": "mlx-fp",
                "metrics": {
                    "per_label_recall": {
                        "PERSON": 0.990,
                        "DATE": 0.990,
                        "ID_NUM": 0.990,
                        "API_KEY": 0.995,
                    },
                    "residual_leakage_rate": 0.0,
                },
                "reproducibility_hash": f"sha256:{'0' * 64}",
                "repo_id": "OpenMed/unit-model",
                "released": "2026-06-14",
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_preview_helper_returns_unsigned_passing_preview(tmp_path: Path) -> None:
    baseline_store = tmp_path / "missing-baseline.json"
    output = tmp_path / "release-gate-report.json"

    report = release_gates.preview(_candidate(), baseline_path=baseline_store)
    rendered = release_gates.format_preview(report)

    assert report.decision == release_gates.RELEASABLE
    assert report.signature is None
    assert "Overall verdict: would-pass" in rendered
    assert "No signed report emitted; no GateReport file written." in rendered
    assert all(check.passed for check in report.gate_results)
    assert not output.exists()
    assert not baseline_store.exists()


def test_preview_lists_failing_gates_with_reasons(tmp_path: Path) -> None:
    report = release_gates.preview(
        _candidate(critical_leakage_count=1),
        baseline_path=tmp_path / "missing-baseline.json",
    )
    rendered = release_gates.format_preview(report)

    assert report.decision == release_gates.QUARANTINED
    assert "Overall verdict: would-fail" in rendered
    assert "G3" in rendered
    assert "fail" in rendered
    assert "critical leakage must be exactly zero" in rendered


def test_openmed_gates_preview_strict_exit_codes_and_no_mutation(
    tmp_path: Path,
    capsys,
) -> None:
    passing = _write_json(tmp_path / "passing.json", _candidate())
    failing = _write_json(
        tmp_path / "failing.json",
        _candidate(critical_leakage_count=1),
    )
    baseline_store = _baseline_store(tmp_path / "baseline.json")
    baseline_before = baseline_store.read_text(encoding="utf-8")

    pass_code = main_module.main(
        [
            "gates",
            "preview",
            "--candidate",
            str(passing),
            "--baseline-store",
            str(baseline_store),
            "--strict",
        ]
    )
    pass_output = capsys.readouterr().out

    fail_code = main_module.main(
        [
            "gates",
            "preview",
            "--candidate",
            str(failing),
            "--baseline-store",
            str(baseline_store),
        ]
    )
    fail_output = capsys.readouterr().out

    strict_fail_code = main_module.main(
        [
            "gates",
            "preview",
            "--candidate",
            str(failing),
            "--baseline-store",
            str(baseline_store),
            "--strict",
        ]
    )
    strict_fail_output = capsys.readouterr().out

    assert pass_code == 0
    assert "Overall verdict: would-pass" in pass_output
    assert fail_code == 0
    assert "Overall verdict: would-fail" in fail_output
    assert strict_fail_code == 1
    assert "Overall verdict: would-fail" in strict_fail_output
    assert baseline_store.read_text(encoding="utf-8") == baseline_before
    assert not (tmp_path / "release-gate-report.json").exists()
