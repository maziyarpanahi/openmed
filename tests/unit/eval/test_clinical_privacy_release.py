from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.core.repro_hash import build_training_provenance, compute_file_digest
from openmed.eval.clinical_privacy_release import (
    CLINICAL_PRIVACY_REQUIRED_GATES,
    ClinicalPrivacyGateFailure,
    ClinicalPrivacyReleaseError,
    build_clinical_privacy_gate_report,
    build_clinical_privacy_release,
)
from openmed.eval.datasets.clinical_phi import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PHI_MANIFEST_REF,
    CLINICAL_PRIVACY_MODEL_ID,
    clinical_phi_manifest_hash,
    load_clinical_phi_manifest,
)
from openmed.eval.release_gates import QUARANTINED, RELEASABLE
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites.shield import SHIELD_LABEL_TO_CANONICAL
from openmed.training.recipe import CONFIG_DIR

SIGNING_KEY = "synthetic-clinical-privacy-release-key"
RAW_SYNTHETIC_CANARY = "Synthetic Canary Person 0042"


def test_builds_signed_phi_free_release_artifacts(tmp_path: Path) -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    release = _build_release(checkpoint=checkpoint, provenance=provenance)

    assert release.gate_report.decision == RELEASABLE
    assert release.gate_report.verify(SIGNING_KEY) is True
    assert tuple(check.gate for check in release.gate_report.gate_results) == (
        CLINICAL_PRIVACY_REQUIRED_GATES
    )
    assert all(check.passed for check in release.gate_report.gate_results)
    assert release.model_manifest_entry["repo_id"] == CLINICAL_PRIVACY_MODEL_ID
    assert [
        benchmark["suite"] for benchmark in release.model_manifest_entry["benchmark"]
    ] == ["clinical-phi-held-out", "shield"]
    assert release.release_manifest["safety"] == {
        "assist_only": True,
        "clinical_decision_trigger": False,
        "local_offline_after_download": True,
        "raw_phi_in_artifacts": False,
        "shield_role": "comparison_only",
    }
    assert "Clinical Privacy Release Boundary" in release.model_card.markdown
    assert "SHIELD" in release.model_card.markdown
    assert "release-evidence/shield-report.json" in release.model_card.markdown
    assert "openmed.eval.datasets:load_clinical_phi_manifest" in (
        release.model_card.markdown
    )

    paths = release.write(tmp_path / "release")
    assert paths.gate_report.is_file()
    assert paths.model_card.is_file()
    assert paths.model_datasheet.is_file()
    assert paths.model_manifest_entry.is_file()
    assert paths.release_manifest.is_file()
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            paths.gate_report,
            paths.model_card,
            paths.model_datasheet,
            paths.model_manifest_entry,
            paths.release_manifest,
        )
    )
    assert RAW_SYNTHETIC_CANARY not in serialized
    assert "fixture-identifiers" not in serialized


def test_g1a_regression_is_signed_and_blocks_release() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    held_out = _held_out_report()
    metrics = dict(held_out.metrics)
    recall = dict(metrics["per_label_recall"])
    recall["ID_NUM"] = 0.989
    metrics["per_label_recall"] = recall
    held_out = replace(held_out, metrics=metrics)

    gate_report = _build_gate_report(
        held_out=held_out,
        checkpoint=checkpoint,
        provenance=provenance,
    )

    assert gate_report.decision == QUARANTINED
    assert gate_report.verify(SIGNING_KEY) is True
    assert _check(gate_report, "G1a").passed is False
    assert _check(gate_report, "G1a").details["violations"] == {"ID_NUM": 0.989}
    with pytest.raises(ClinicalPrivacyGateFailure) as exc_info:
        _build_release(
            held_out=held_out,
            checkpoint=checkpoint,
            provenance=provenance,
        )
    assert exc_info.value.report.verify(SIGNING_KEY) is True


def test_missing_g2_category_coverage_quarantines_candidate() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    held_out = _held_out_report()
    metrics = dict(held_out.metrics)
    name_labels = set(load_clinical_phi_manifest().label_groups["names"])
    metrics["per_label_recall"] = {
        label: value
        for label, value in metrics["per_label_recall"].items()
        if label not in name_labels
    }
    leakage = dict(metrics["leakage"])
    leakage["total_chars_by_label"] = {
        label: value
        for label, value in leakage["total_chars_by_label"].items()
        if label not in name_labels
    }
    metrics["leakage"] = leakage

    report = _build_gate_report(
        held_out=replace(held_out, metrics=metrics),
        checkpoint=checkpoint,
        provenance=provenance,
    )

    g2 = _check(report, "G2")
    assert report.decision == QUARANTINED
    assert g2.passed is False
    assert g2.details["missing_coverage_groups"] == ["names"]
    assert "held-out coverage missing" in g2.reason


def test_g3_requires_explicit_zero_leakage_count() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    held_out = _held_out_report()
    metrics = dict(held_out.metrics)
    metrics.pop("critical_leakage_count")

    report = _build_gate_report(
        held_out=replace(held_out, metrics=metrics),
        checkpoint=checkpoint,
        provenance=provenance,
    )

    g3 = _check(report, "G3")
    assert report.decision == QUARANTINED
    assert g3.passed is False
    assert g3.details["explicit_critical_leakage_count"] is False
    assert g3.reason == "critical leakage count is not explicitly reported"


def test_g3_requires_zero_residual_leakage() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    held_out = _held_out_report()
    metrics = dict(held_out.metrics)
    leakage = dict(metrics["leakage"])
    leakage["overall"] = 0.001
    metrics["leakage"] = leakage

    report = _build_gate_report(
        held_out=replace(held_out, metrics=metrics),
        checkpoint=checkpoint,
        provenance=provenance,
    )

    g3 = _check(report, "G3")
    assert report.decision == QUARANTINED
    assert g3.passed is False
    assert g3.details["residual_leakage_rate"] == 0.001
    assert g3.reason == "clinical PHI residual leakage must be exactly zero"


def test_rejects_shield_report_promoted_to_release_gate() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    shield = _shield_report(checkpoint)
    metrics = dict(shield.metrics)
    comparison = dict(metrics["shield_comparison"])
    comparison["high_recall_release_gate"] = True
    metrics["shield_comparison"] = comparison

    with pytest.raises(ClinicalPrivacyReleaseError, match="comparison-only"):
        _build_gate_report(
            shield=replace(shield, metrics=metrics),
            checkpoint=checkpoint,
            provenance=provenance,
        )


def test_rejects_training_provenance_from_another_dataset() -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    wrong = build_training_provenance(
        rng_seeds=provenance["rng_seeds"],
        data_manifest_hash="sha256:" + "f" * 64,
        recipe_config_hash=provenance["recipe_config_hash"],
        env_lock_digest=provenance["env_lock_digest"],
        base_model=provenance["base_model"],
        base_model_revision=provenance["base_model_revision"],
        git_sha=provenance["git_sha"],
        repo_id=CLINICAL_PRIVACY_MODEL_ID,
    )
    checkpoint = dict(checkpoint)
    checkpoint["reproducibility_hash"] = wrong["reproducibility_hash"]

    with pytest.raises(ClinicalPrivacyReleaseError, match="dataset manifest"):
        _build_gate_report(checkpoint=checkpoint, provenance=wrong)


def _build_release(
    *,
    checkpoint: dict[str, object],
    provenance: dict[str, object],
    held_out: BenchmarkReport | None = None,
):
    return build_clinical_privacy_release(
        held_out or _held_out_report(),
        _shield_report(checkpoint),
        checkpoint_manifest=checkpoint,
        training_provenance=provenance,
        checkpoint_manifest_ref="models.jsonl#clinical-privacy-tier0",
        held_out_report_ref="release-evidence/held-out-report.json",
        shield_report_ref="release-evidence/shield-report.json",
        release_date="2026-08-01",
        signing_key=SIGNING_KEY,
        key_id="synthetic-unit-key",
    )


def _build_gate_report(
    *,
    checkpoint: dict[str, object],
    provenance: dict[str, object],
    held_out: BenchmarkReport | None = None,
    shield: BenchmarkReport | None = None,
):
    return build_clinical_privacy_gate_report(
        held_out or _held_out_report(),
        shield or _shield_report(checkpoint),
        checkpoint_manifest=checkpoint,
        training_provenance=provenance,
        checkpoint_manifest_ref="models.jsonl#clinical-privacy-tier0",
        held_out_report_ref="release-evidence/held-out-report.json",
        shield_report_ref="release-evidence/shield-report.json",
        signing_key=SIGNING_KEY,
        key_id="synthetic-unit-key",
    )


def _checkpoint_and_provenance() -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_clinical_phi_manifest()
    provenance = build_training_provenance(
        rng_seeds={"python": 3803, "torch": 3803},
        data_manifest_hash=clinical_phi_manifest_hash(manifest),
        recipe_config_hash=compute_file_digest(CONFIG_DIR / "large_teacher.yaml"),
        env_lock_digest="sha256:" + "e" * 64,
        base_model="OpenMed/synthetic-clinical-teacher",
        base_model_revision="b" * 40,
        git_sha="a" * 40,
        repo_id=CLINICAL_PRIVACY_MODEL_ID,
        checkpoint_id="synthetic-tier0-checkpoint",
    )
    checkpoint: dict[str, object] = {
        "repo_id": CLINICAL_PRIVACY_MODEL_ID,
        "family": "ClinicalPrivacy",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Large",
        "param_count": 434_000_000,
        "architecture": "deberta-v2",
        "base_model": provenance["base_model"],
        "formats": ["pytorch"],
        "canonical_labels": list(manifest.required_labels()),
        "benchmark": {
            "dataset": "pending-certification",
            "micro_f1": None,
            "recall": None,
        },
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": provenance["reproducibility_hash"],
        "released": None,
    }
    return checkpoint, provenance


def _held_out_report() -> BenchmarkReport:
    labels = load_clinical_phi_manifest().required_labels()
    per_label_recall = {label: 0.991 for label in labels}
    per_label_precision = {label: 0.994 for label in labels}
    return BenchmarkReport(
        suite="clinical-phi-held-out",
        model_name=CLINICAL_PRIVACY_MODEL_ID,
        device="cpu",
        fixture_count=24,
        generated_at="2026-08-01T00:00:00Z",
        metrics={
            "per_label_recall": per_label_recall,
            "per_label_precision": per_label_precision,
            "exact_span_f1": {"f1": 0.992, "precision": 0.994, "recall": 0.991},
            "critical_leakage_count": 0,
            "leakage": {
                "overall": 0.0,
                "leaked_chars_by_label": {},
                "total_chars_by_label": {label: 20 for label in labels},
            },
            "latency": {"p50_ms": 40.0, "p95_ms": 90.0},
            "resources": {"peak_rss_mib": 512.0},
        },
        metadata={
            "eval_set_hash": "sha256:" + "1" * 64,
            "format": "pytorch",
            "leakage_fixture_hash": "sha256:" + "2" * 64,
            "synthetic_debug_value": RAW_SYNTHETIC_CANARY,
            "fixture_identifiers": ["fixture-identifiers"],
        },
    )


def _shield_report(checkpoint: dict[str, object]) -> BenchmarkReport:
    return BenchmarkReport(
        suite="shield",
        model_name=CLINICAL_PRIVACY_MODEL_ID,
        device="cpu",
        fixture_count=9,
        generated_at="2026-08-01T00:00:00Z",
        metrics={
            "shield_comparison": {
                "evidence_role": "comparison",
                "high_recall_release_gate": False,
                "aggregate": {
                    "exact_span_f1": 0.91,
                    "exact_span_precision": 0.92,
                    "exact_span_recall": 0.90,
                    "leakage": 0.10,
                    "recall": 0.90,
                },
                "by_label": {
                    label: {"leakage": 0.05, "recall": 0.95}
                    for label in set(SHIELD_LABEL_TO_CANONICAL.values())
                },
            }
        },
        metadata={
            "comparison_evidence_only": True,
            "gate_target": False,
            "checkpoint_manifest": {
                "model_id": CLINICAL_PRIVACY_MODEL_ID,
                "reproducibility_hash": checkpoint["reproducibility_hash"],
            },
            "dataset_manifest": {
                "manifest_hash": clinical_phi_manifest_hash(),
                "manifest_id": CLINICAL_PHI_MANIFEST_ID,
                "manifest_ref": CLINICAL_PHI_MANIFEST_REF,
            },
            "public_corpus_reference": {
                "dataset": "shield",
                "redistribution": "reference-only",
                "source_url": "https://example.invalid/shield-synthetic-reference",
            },
            "fixture_ids": ["sha256:" + "3" * 64],
        },
    )


def _check(report, gate: str):
    return next(check for check in report.gate_results if check.gate == gate)


def test_written_release_manifest_is_deterministic_json(tmp_path: Path) -> None:
    checkpoint, provenance = _checkpoint_and_provenance()
    release = _build_release(checkpoint=checkpoint, provenance=provenance)

    first = release.write(tmp_path / "first")
    second = release.write(tmp_path / "second")

    first_payload = json.loads(first.release_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second.release_manifest.read_text(encoding="utf-8"))
    assert first_payload == second_payload == release.release_manifest
