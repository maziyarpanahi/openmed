from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.core.repro_hash import build_training_provenance, write_training_provenance
from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.model_card_builder import (
    MODEL_DATASHEET_FILENAME,
    ModelCardBuilderError,
    build_model_card,
    validate_model_card_consistency,
)
from openmed.eval.release_gates import RELEASABLE, GateCheck, GateReport

SIGNING_KEY = "unit-model-card-key"


def test_model_card_builder_matches_gate_report_and_is_byte_stable(
    tmp_path: Path,
) -> None:
    row = _manifest_row(tmp_path)
    gate_path = _write_json(tmp_path / "gate-report.json", _gate_report().to_dict())
    provenance_path = _write_training_provenance(tmp_path, row["repo_id"])
    fairness_path = _write_json(
        tmp_path / "fairness-report.json",
        {
            "suite": "golden",
            "model_name": row["repo_id"],
            "leakage_disparity": 0.0,
            "worst_group_leakage": 0.0,
            "worst_group": None,
        },
    )
    calibration_path = _write_json(
        tmp_path / "calibration-report.json",
        {
            "schema_version": 1,
            "artifact_type": "openmed.calibration.report",
            "groups": [],
        },
    )
    quant_path = _write_json(
        tmp_path / "quant-delta.json",
        {
            "format": "mlx-fp",
            "quantized": False,
            "passed": True,
            "max_delta": None,
        },
    )

    first = build_model_card(
        row,
        gate_path,
        calibration_report=calibration_path,
        fairness_report=fairness_path,
        quant_delta=quant_path,
        training_provenance=provenance_path,
    )
    second = build_model_card(
        row,
        gate_path,
        calibration_report=calibration_path,
        fairness_report=fairness_path,
        quant_delta=quant_path,
        training_provenance=provenance_path,
    )

    assert first.markdown == second.markdown
    assert first.datasheet_json() == second.datasheet_json()

    datasheet = json.loads(first.datasheet_json())
    assert datasheet["gate_evidence"] == {
        "critical_leakage_count": 0,
        "eval_set_hash": "sha256:eval",
        "leakage_fixture_hash": "sha256:leakage",
        "per_label_recall": {"DATE": 0.992, "PERSON": 0.995},
        "quant_recall_delta": 0.0,
        "tier_fit": {
            "details": {
                "budget": {"p50_ms": 60.0, "p95_ms": 150.0, "ram_mb": 350.0},
                "tier": "tiny",
                "violations": {},
            },
            "gate": "G5",
            "passed": True,
            "reason": "ok",
        },
    }
    assert datasheet["training_data_lineage"]["license_tags"] == [
        "apache-2.0",
        "synthetic-openmed",
    ]
    assert "`synthetic-openmed`" in first.markdown
    assert "Patient John" not in first.markdown


def test_model_card_consistency_rejects_tampered_gate_evidence(
    tmp_path: Path,
) -> None:
    row = _manifest_row(tmp_path)
    result = build_model_card(
        row,
        _gate_report(),
        training_provenance=_write_training_provenance(tmp_path, row["repo_id"]),
    )
    tampered = copy.deepcopy(result.datasheet)
    tampered["gate_evidence"]["critical_leakage_count"] = 1

    with pytest.raises(ModelCardBuilderError, match="critical_leakage_count"):
        validate_model_card_consistency(
            result.markdown,
            row,
            _gate_report(),
            datasheet=tampered,
        )


def test_model_card_datasheet_can_be_archived_in_evidence_bundle(
    tmp_path: Path,
) -> None:
    row = _manifest_row(tmp_path)
    result = build_model_card(
        row,
        _gate_report(),
        training_provenance=_write_training_provenance(tmp_path, row["repo_id"]),
    )
    card_path = result.write_markdown(tmp_path / "README.md")
    datasheet_path = result.write_datasheet(tmp_path / MODEL_DATASHEET_FILENAME)

    bundle = bundle_gate_evidence(
        result.gate_report,
        tmp_path / "bundle",
        extra_artifacts={
            "model_card": card_path,
            "model_datasheet": datasheet_path,
        },
    )

    present = {
        artifact["artifact_id"]
        for artifact in bundle.manifest["artifacts"]
        if artifact["status"] == "present"
    }
    assert {"model_card", "model_datasheet"}.issubset(present)


def _manifest_row(tmp_path: Path) -> dict[str, object]:
    return {
        "repo_id": "OpenMed/OpenMed-PII-Tiny-44M-v1-mlx",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 44_000_000,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/OpenMed-PII-Tiny-44M",
        "formats": ["mlx-fp"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {
            "dataset": "synthetic-card-fixture",
            "micro_f1": 0.99,
            "recall": 0.995,
        },
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "released": "2026-07-05",
        "training_provenance": {"path": str(tmp_path / "training_provenance.json")},
    }


def _gate_report() -> GateReport:
    return GateReport(
        repo_id="OpenMed/OpenMed-PII-Tiny-44M-v1-mlx",
        family="PII",
        tier="Tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"PERSON": 0.995, "DATE": 0.992},
        per_label_precision={"PERSON": 0.994, "DATE": 0.991},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=32.0,
        p95_ms=81.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=RELEASABLE,
        gate_results=(
            GateCheck(
                "G5",
                True,
                reason="ok",
                details={
                    "tier": "tiny",
                    "budget": {"ram_mb": 350.0, "p50_ms": 60.0, "p95_ms": 150.0},
                    "violations": {},
                },
            ),
        ),
        policy="hipaa_safe_harbor",
        threshold_profile="strict",
        target_leakage_rate=0.005,
    ).sign(SIGNING_KEY)


def _write_training_provenance(tmp_path: Path, repo_id: object) -> Path:
    provenance = build_training_provenance(
        rng_seeds={"python": 123, "numpy": 456},
        data_manifest_hash="sha256:" + "2" * 64,
        recipe_config_hash="sha256:" + "3" * 64,
        env_lock_digest="sha256:" + "4" * 64,
        base_model="OpenMed/OpenMed-PII-Tiny-44M",
        base_model_revision="main",
        git_sha="abc123",
        repo_id=str(repo_id),
    )
    provenance["license_tags"] = ["synthetic-openmed"]
    return write_training_provenance(tmp_path, provenance)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
