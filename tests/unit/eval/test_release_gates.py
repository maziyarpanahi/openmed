from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval import release_gates
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateReport,
    ReleaseGate,
)
from openmed.eval.report import BenchmarkReport

SIGNING_KEY = "unit-release-key"


def _calibration_files(tmp_path: Path) -> tuple[Path, Path]:
    thresholds = tmp_path / "thresholds.json"
    calibration = tmp_path / "calibration_report.json"
    thresholds.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "openmed.calibration.thresholds",
                "thresholds": {"unit-model": {"PERSON": {"en": 0.9}}},
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps({"schema_version": 1, "groups": []}),
        encoding="utf-8",
    )
    return thresholds, calibration


def _report(
    tmp_path: Path,
    *,
    metadata_updates: dict[str, object] | None = None,
    metric_updates: dict[str, object] | None = None,
) -> BenchmarkReport:
    thresholds, calibration = _calibration_files(tmp_path)
    metadata: dict[str, object] = {
        "repo_id": "OpenMed/unit-model",
        "family": "PII",
        "tier": "Tiny",
        "param_count": 44_000_000,
        "format": "mlx-fp",
        "eval_set_hash": "sha256:eval",
        "leakage_fixture_hash": "sha256:leakage",
        "policy": "hipaa_safe_harbor",
        "thresholds_path": str(thresholds),
        "calibration_report_path": str(calibration),
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
    }
    if metadata_updates:
        metadata.update(metadata_updates)

    metrics: dict[str, object] = {
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
        "critical_leakage_count": 0,
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
        "quant_recall_delta": 0.0,
        "latency": {"p50_ms": 50.0, "p95_ms": 120.0},
        "resources": {"peak_rss_mib": 128.0},
    }
    if metric_updates:
        metrics.update(metric_updates)

    return BenchmarkReport(
        suite="golden",
        model_name="unit-model",
        device="cpu",
        fixture_count=1,
        generated_at="2026-06-15T00:00:00+00:00",
        metrics=metrics,
        metadata=metadata,
    )


def _baseline() -> dict[str, object]:
    return {
        "key": "pii::tiny::mlx-fp",
        "metrics": {
            "per_label_recall": {
                "PERSON": 0.990,
                "DATE": 0.990,
                "ID_NUM": 0.990,
                "API_KEY": 0.995,
            },
            "residual_leakage_rate": 0.0,
        },
    }


def _gate() -> ReleaseGate:
    return ReleaseGate(signing_key=SIGNING_KEY)


def _check(report, gate_name: str):
    return next(check for check in report.gate_results if check.gate == gate_name)


def test_release_gate_passes_and_emits_signed_section_64_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    original_load_policy = release_gates.policy_module.load_policy

    def spy_load_policy(name: str):
        calls.append(name)
        return original_load_policy(name)

    monkeypatch.setattr(release_gates.policy_module, "load_policy", spy_load_policy)

    result = _gate().evaluate(_report(tmp_path), _baseline())

    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert calls == ["hipaa_safe_harbor"]
    assert {
        "repo_id",
        "family",
        "tier",
        "param_count",
        "format",
        "per_label_recall",
        "per_label_precision",
        "critical_leakage_count",
        "residual_leakage_rate",
        "quant_recall_delta",
        "p50_ms",
        "p95_ms",
        "ram_mb",
        "eval_set_hash",
        "leakage_fixture_hash",
        "decision",
    }.issubset(result.to_dict())

    restored = GateReport.from_json(result.to_json())
    assert restored.verify(SIGNING_KEY)
    assert restored.to_json() == result.to_json()


def test_gate_report_from_json_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="Invalid JSON for GateReport"):
        GateReport.from_json("{")


def test_find_open_issue_returns_none_for_malformed_gh_json(monkeypatch) -> None:
    class Result:
        stdout = "{"

    monkeypatch.setattr(
        release_gates.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )

    assert (
        release_gates._find_open_issue(repo="owner/repo", title="Gate failure") is None
    )


@pytest.mark.parametrize(
    ("gate_name", "metric_updates", "metadata_updates"),
    [
        (
            "G1a",
            {"per_label_recall": {"PERSON": 0.989, "API_KEY": 0.995}},
            None,
        ),
        (
            "G1b",
            {"per_label_recall": {"PERSON": 0.990, "DATE": 0.990, "API_KEY": 0.994}},
            None,
        ),
        (
            "G2",
            {
                "per_label_recall": {
                    "PERSON": 0.990,
                    "DATE": 0.979,
                    "API_KEY": 0.995,
                }
            },
            None,
        ),
        ("G3", {"critical_leakage_count": 1}, None),
        ("G5", {"latency": {"p50_ms": 50.0, "p95_ms": 151.0}}, None),
        ("G6", {"latency": {"p50_ms": 50.0}}, None),
        (
            "G8",
            None,
            {
                "span_fixtures": [
                    {
                        "text": "Patient John",
                        "predicted_spans": [
                            {"start": 8, "end": 99, "label": "PERSON"},
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_release_gate_blocks_failed_gate_boundaries(
    tmp_path: Path,
    gate_name: str,
    metric_updates: dict[str, object] | None,
    metadata_updates: dict[str, object] | None,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates=metric_updates,
            metadata_updates=metadata_updates,
        ),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, gate_name).passed is False


def test_critical_leakage_forces_non_releasable(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(tmp_path, metric_updates={"critical_leakage_count": 2}),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, "G3").reason == "critical leakage must be exactly zero"


def test_g4_blocks_only_the_offending_quantized_format(tmp_path: Path) -> None:
    int8 = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-8bit"},
            metric_updates={"quant_recall_delta": 0.006},
        ),
        _baseline(),
    )
    fp = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-fp"},
            metric_updates={"quant_recall_delta": 0.006},
        ),
        _baseline(),
    )

    assert int8.decision == QUARANTINED
    assert int8.blocked_formats == ("mlx-8bit",)
    assert _check(int8, "G4").passed is False
    assert fp.decision == RELEASABLE
    assert fp.blocked_formats == ()
    assert _check(fp, "G4").passed is True


def test_g5_uses_nano_certificate_for_nano_declared_artifacts(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"tier": "Nano", "param_count": 44_000_000},
            metric_updates={
                "latency": {"p50_ms": 20.0, "p95_ms": 50.0},
                "resources": {"peak_rss_mib": 128.0},
            },
        ),
        _baseline(),
    )

    check = _check(result, "G5")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "Nano sub-tier budget not certified"
    assert check.details["failing_dimension"] == "param_count"
    assert check.details["parent_tier"] == "Tiny"


def test_missing_calibration_artifacts_fail_closed(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "thresholds_path": str(tmp_path / "missing-thresholds.json"),
                "calibration_report_path": str(tmp_path / "missing-report.json"),
            },
        ),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, "calibration_present").passed is False


def test_g7_blocks_recall_regression_and_residual_leakage(tmp_path: Path) -> None:
    baseline = {
        "metrics": {
            "per_label_recall": {"PERSON": 0.995, "API_KEY": 0.995},
            "residual_leakage_rate": 0.0,
        }
    }
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "per_label_recall": {"PERSON": 0.992, "API_KEY": 0.995},
                "leakage": {"overall": 0.001, "total_chars_by_label": {"PERSON": 4}},
            },
        ),
        baseline,
    )

    assert result.decision == QUARANTINED
    check = _check(result, "G7")
    assert check.passed is False
    assert "recall_drop" in check.details["violations"]
    assert "residual_leakage_regression" in check.details["violations"]


def test_zero_shot_language_gate_quarantines_transfer_floor_breach(
    tmp_path: Path,
) -> None:
    transfer_matrix = {
        "schema_version": 1,
        "artifact_type": "openmed.cross_lingual_transfer_matrix",
        "languages": ["en", "fr"],
        "leakage_floors": {"en": 0.10, "fr": 0.10},
        "matrix": {
            "en": {
                "en": {
                    "source_language": "en",
                    "target_language": "en",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": False,
                },
                "fr": {
                    "source_language": "en",
                    "target_language": "fr",
                    "leakage_rate": 0.25,
                    "leaked_chars": 25,
                    "total_chars": 100,
                    "zero_shot": True,
                },
            },
            "fr": {
                "en": {
                    "source_language": "fr",
                    "target_language": "en",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": True,
                },
                "fr": {
                    "source_language": "fr",
                    "target_language": "fr",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": False,
                },
            },
        },
    }

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"cross_lingual_transfer": transfer_matrix},
        ),
        _baseline(),
    )

    check = _check(result, "G9_zero_shot_language_leakage")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["violations"] == [
        {
            "source_language": "en",
            "target_language": "fr",
            "leakage_rate": 0.25,
            "leakage_floor": 0.10,
            "excess": 0.15,
            "leaked_chars": 25,
            "total_chars": 100,
        }
    ]


def test_membership_leakage_gate_blocks_leaky_configuration(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "membership_leakage": {
                    "attacker_auc": 0.91,
                    "attacker_advantage": 0.35,
                    "advantage_ceiling": 0.05,
                    "feature_hash": "sha256:features",
                    "per_label": {
                        "PERSON": {
                            "attacker_advantage": 0.35,
                            "feature_hash": "sha256:person",
                        }
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "membership_leakage")
    assert result.decision == QUARANTINED
    assert result.verify(SIGNING_KEY)
    assert check.passed is False
    assert "overall_advantage" in check.details["violations"]
    assert "PERSON" in check.details["violations"]["per_label_advantage"]


def test_membership_leakage_gate_passes_defended_configuration(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "membership_leakage": {
                    "attacker_auc": 0.5,
                    "attacker_advantage": 0.0,
                    "advantage_ceiling": 0.05,
                    "feature_hash": "sha256:features",
                    "defense": {"enabled": True, "clip_min": 0.5, "clip_max": 0.5},
                    "per_label": {
                        "PERSON": {
                            "attacker_advantage": 0.0,
                            "feature_hash": "sha256:person",
                        }
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "membership_leakage")
    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert check.passed is True
    assert check.details["defense"]["enabled"] is True


def test_g8_consumes_strict_quality_gate_output(tmp_path: Path, monkeypatch) -> None:
    calls = {"strict": 0}
    original = release_gates.quality_gates.validate_entity_spans_strict

    def strict(entities, text):
        calls["strict"] += 1
        return original(entities, text)

    monkeypatch.setattr(
        release_gates.quality_gates,
        "validate_entity_spans_strict",
        strict,
    )

    result = _gate().evaluate(_report(tmp_path), _baseline())

    assert result.decision == RELEASABLE
    assert calls == {"strict": 1}
    assert _check(result, "G8").details["spans_checked"] == 3


def test_k_floor_release_gate_passes_signed_enforcement_evidence(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "kanon_enforcement": {
                    "target_k": 2,
                    "kanon": {"k": 2},
                    "bounds": {
                        "max_reidentification_upper_bound": 0.5,
                        "numeric_self_check": {"passed": True},
                    },
                }
            },
        ),
        _baseline(),
    )

    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert _check(result, "k_floor").passed is True


def test_k_floor_release_gate_fails_realized_k_below_target(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "kanon_enforcement": {
                    "target_k": 3,
                    "kanon": {"k": 2},
                    "bounds": {
                        "max_reidentification_upper_bound": 0.5,
                        "numeric_self_check": {"passed": False},
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "k_floor")
    assert result.decision == QUARANTINED
    assert result.verify(SIGNING_KEY)
    assert check.passed is False
    assert check.details["violations"]["measured_k"] == {"observed": 2, "target": 3}


def test_g4_computes_quant_delta_from_fp_parent_recall(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-8bit"},
            metric_updates={
                "quant_recall_delta": None,
                "per_label_recall": {
                    "PERSON": 0.991,
                    "DATE": 0.990,
                    "ID_NUM": 0.990,
                    "API_KEY": 0.995,
                },
                "fp_parent_per_label_recall": {"PERSON": 0.996},
            },
        ),
        _baseline(),
    )

    check = _check(result, "G4")
    assert result.decision == QUARANTINED
    assert result.quant_recall_delta == pytest.approx(0.005)
    assert check.passed is False
    assert check.details["offending_labels"]["PERSON"]["limit"] == 0.005
    assert result.blocked_formats == ("mlx-8bit",)


def test_manifest_coherence_fails_when_readme_count_drifts(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/unit-model",
                "family": "PII",
                "task": "token-classification",
                "languages": ["en"],
                "tier": "Tiny",
                "param_count": 44_000_000,
                "formats": ["mlx-fp"],
                "license": "apache-2.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("2 models\n", encoding="utf-8")

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "manifest_path": str(manifest),
                "readme_path": str(readme),
            },
        ),
        _baseline(),
    )

    check = _check(result, "manifest_coherence")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["mismatches"]["readme"]["models"]["readme_floor"] == 2
