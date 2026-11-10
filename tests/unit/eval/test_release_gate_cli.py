from __future__ import annotations

import json
from pathlib import Path

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


def test_release_gate_cli_returns_zero_for_releasable_candidate(tmp_path: Path) -> None:
    candidate = _write_json(tmp_path / "candidate.json", _candidate())
    output = tmp_path / "gate-report.json"

    exit_code = release_gates.main(
        [
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--signing-key",
            "unit-key",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "RELEASABLE"


def test_release_gate_cli_returns_nonzero_for_failed_gate(tmp_path: Path) -> None:
    candidate = _write_json(
        tmp_path / "candidate.json",
        _candidate(critical_leakage_count=1),
    )
    output = tmp_path / "gate-report.json"

    exit_code = release_gates.main(
        [
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--signing-key",
            "unit-key",
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "QUARANTINED"
    assert any(
        check["gate"] == "G3" and check["passed"] is False
        for check in report["gate_results"]
    )
