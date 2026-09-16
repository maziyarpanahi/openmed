"""Synthetic, offline tests for DirectID Tiny release certification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.audit import stable_hash
from openmed.core.repro_hash import build_training_provenance
from openmed.eval.directid import DIRECTID_EVIDENCE_SCHEMA_VERSION
from openmed.eval.directid_release import (
    DIRECTID_MODEL_ID,
    DIRECTID_REQUIRED_GATES,
    DirectIDGateFailure,
    DirectIDReleaseError,
    build_directid_gate_report,
    build_directid_release,
)
from openmed.eval.release_gates import QUARANTINED, RELEASABLE
from openmed.training.directid import (
    DIRECTID_CONTRACT_REF,
    DIRECTID_FAMILY,
    DIRECTID_TIER,
    DIRECTID_TINY_HEAD_CONTRACT,
)
from openmed.training.directid_dataset import (
    DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION,
    DIRECTID_DATASET_MANIFEST_ID,
    directid_dataset_manifest_hash,
)

SIGNING_KEY = "synthetic-directid-release-key"
SYNTHETIC_CANARY = "Synthetic Canary Identifier 0602"


def test_builds_signed_phi_free_release_package(tmp_path: Path) -> None:
    evidence = _release_evidence()

    release = _build_release(evidence)

    assert release.published is True
    assert release.gate_report.decision == RELEASABLE
    assert release.gate_report.verify(SIGNING_KEY) is True
    assert tuple(check.gate for check in release.gate_report.gate_results) == (
        DIRECTID_REQUIRED_GATES
    )
    assert all(check.passed for check in release.gate_report.gate_results)
    assert release.checkpoint_manifest is not None
    assert release.checkpoint_manifest["repo_id"] == DIRECTID_MODEL_ID
    assert release.checkpoint_manifest["formats"] == ["mlx-8bit"]
    assert release.model_card is not None
    assert "Dataset Provenance" in release.model_card.markdown
    assert "Safety-Sweep Evidence" in release.model_card.markdown
    assert "Quantization Evidence" in release.model_card.markdown
    assert release.release_manifest["publication"] == {
        "publish_target": DIRECTID_MODEL_ID,
        "status": "published",
    }

    paths = release.write(tmp_path / "release")
    assert paths.gate_report.is_file()
    assert paths.release_manifest.is_file()
    assert paths.checkpoint_manifest is not None
    assert paths.checkpoint_manifest.is_file()
    assert paths.model_card is not None
    assert paths.model_card.is_file()
    assert paths.model_datasheet is not None
    assert paths.model_datasheet.is_file()
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            paths.gate_report,
            paths.release_manifest,
            paths.checkpoint_manifest,
            paths.model_card,
            paths.model_datasheet,
        )
    )
    assert SYNTHETIC_CANARY not in serialized
    assert "raw_identifier" not in serialized


def test_g1b_regression_writes_quarantine_only(tmp_path: Path) -> None:
    evidence = _release_evidence()
    for label in DIRECTID_TINY_HEAD_CONTRACT.structured_id_labels:
        evidence["directid"]["combined"]["per_label_recall"][label] = 0.994
        evidence["directid"]["gate_evidence"]["G1b"]["per_label_recall"][label] = 0.994
    evidence["directid"]["combined"]["structured_id_recall"] = 0.994
    evidence["directid"]["gate_evidence"]["G1b"]["structured_id_recall"] = 0.994

    release = _build_release(evidence)

    assert release.published is False
    assert release.gate_report.decision == QUARANTINED
    assert release.gate_report.verify(SIGNING_KEY) is True
    assert _check(release, "G1b").passed is False
    assert _check(release, "G1b").reason == (
        "structured-id recall is below the G1b floor"
    )
    assert release.checkpoint_manifest is None
    assert release.model_card is None
    assert release.release_manifest["publication"] == {
        "publish_target": None,
        "status": "quarantined",
    }
    assert release.release_manifest["failed_gates"] == ["G1b"]

    paths = release.write(tmp_path / "quarantine")
    assert paths.gate_report.is_file()
    assert paths.release_manifest.is_file()
    assert paths.checkpoint_manifest is None
    assert paths.model_card is None
    assert paths.model_datasheet is None
    assert not (tmp_path / "quarantine" / "checkpoint-manifest.json").exists()
    assert not (tmp_path / "quarantine" / "README.md").exists()

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "README.md").write_text("stale publish artifact", encoding="utf-8")
    with pytest.raises(DirectIDReleaseError, match="beside publishable artifacts"):
        release.write(stale)

    with pytest.raises(DirectIDGateFailure) as error:
        _build_release(evidence, require_releasable=True)
    assert error.value.report.verify(SIGNING_KEY) is True


def test_failed_optional_int4_is_quarantined_without_blocking_int8() -> None:
    evidence = _release_evidence()
    failed_int4 = dict(evidence["quantization"]["artifacts"][0])
    failed_int4.update(
        {
            "format": "mlx-4bit",
            "bits": 4,
            "artifact_hash": _digest("4"),
            "quant_recall_delta": 0.011,
            "disposition": "quarantined",
            "g4": {"passed": False},
        }
    )
    evidence["quantization"]["artifacts"].append(failed_int4)
    evidence["quantization"]["quarantined_formats"] = ["mlx-4bit"]

    release = _build_release(evidence)

    assert release.gate_report.decision == RELEASABLE
    assert release.gate_report.blocked_formats == ("mlx-4bit",)
    assert release.release_manifest["published_formats"] == ["mlx-8bit"]
    assert release.release_manifest["quarantined_formats"] == ["mlx-4bit"]


@pytest.mark.parametrize(
    ("gate", "updates", "reason"),
    [
        (
            "G3",
            {"critical_leakage_count": 1, "residual_leakage_rate": 0.001},
            "critical leakage must be exactly zero",
        ),
        (
            "G4",
            {
                "quant_recall_delta": 0.006,
                "g4": {"passed": False},
                "disposition": "quarantined",
            },
            "selected format exceeds the G4 recall-delta limit",
        ),
        (
            "G5",
            {
                "p95_ms": 151.0,
                "tiny_tier_fit": {"passed": False},
                "disposition": "quarantined",
            },
            "selected format exceeds the Tiny-tier budget",
        ),
    ],
)
def test_each_release_gate_quarantines_its_selected_artifact(
    gate: str,
    updates: dict[str, object],
    reason: str,
) -> None:
    evidence = _release_evidence()
    if gate == "G3":
        evidence["directid"].update(updates)
        evidence["directid"]["gate_evidence"]["G3"].update(updates)
    else:
        evidence["quantization"]["artifacts"][0].update(updates)
        evidence["quantization"]["accepted_formats"] = []
        evidence["quantization"]["quarantined_formats"] = ["mlx-8bit"]

    report = _build_gate_report(evidence)

    assert report.decision == QUARANTINED
    check = next(item for item in report.gate_results if item.gate == gate)
    assert check.passed is False
    assert check.reason == reason
    assert report.verify(SIGNING_KEY) is True


def test_rejects_training_provenance_from_another_dataset() -> None:
    evidence = _release_evidence()
    wrong = build_training_provenance(
        rng_seeds={"numpy": 3803, "python": 3803, "torch": 3803},
        data_manifest_hash=_digest("f"),
        recipe_config_hash=_digest("2"),
        env_lock_digest=_digest("3"),
        base_model="OpenMed/synthetic-directid-student",
        base_model_revision="a" * 40,
        git_sha="b" * 40,
        repo_id=DIRECTID_MODEL_ID,
        checkpoint_id=evidence["candidate"]["checkpoint_ref"],
    )
    evidence["provenance"] = wrong
    evidence["candidate"]["reproducibility_hash"] = wrong["reproducibility_hash"]
    evidence["training_report"]["candidate_checkpoint"] = dict(evidence["candidate"])
    evidence["run_manifest"]["candidate_checkpoint"] = dict(evidence["candidate"])
    evidence["run_manifest"]["training_provenance"] = dict(wrong)
    evidence["quantization"]["candidate"]["reproducibility_hash"] = wrong[
        "reproducibility_hash"
    ]

    with pytest.raises(DirectIDReleaseError, match="dataset manifest"):
        _build_gate_report(evidence)


def _build_release(
    evidence: dict[str, dict[str, object]],
    *,
    require_releasable: bool = False,
):
    return build_directid_release(
        evidence["directid"],
        candidate_checkpoint=evidence["candidate"],
        training_report=evidence["training_report"],
        run_manifest=evidence["run_manifest"],
        training_provenance=evidence["provenance"],
        dataset_evidence=evidence["dataset"],
        quantization_evidence=evidence["quantization"],
        release_date="2026-08-02",
        signing_key=SIGNING_KEY,
        key_id="synthetic-unit-key",
        require_releasable=require_releasable,
    )


def _build_gate_report(evidence: dict[str, dict[str, object]]):
    return build_directid_gate_report(
        evidence["directid"],
        candidate_checkpoint=evidence["candidate"],
        training_report=evidence["training_report"],
        run_manifest=evidence["run_manifest"],
        training_provenance=evidence["provenance"],
        dataset_evidence=evidence["dataset"],
        quantization_evidence=evidence["quantization"],
        signing_key=SIGNING_KEY,
        key_id="synthetic-unit-key",
    )


def _release_evidence() -> dict[str, dict[str, object]]:
    dataset_hash = directid_dataset_manifest_hash()
    eval_hash = _digest("5")
    provenance = build_training_provenance(
        rng_seeds={"numpy": 3803, "python": 3803, "torch": 3803},
        data_manifest_hash=dataset_hash,
        recipe_config_hash=_digest("2"),
        env_lock_digest=_digest("3"),
        base_model="OpenMed/synthetic-directid-student",
        base_model_revision="a" * 40,
        git_sha="b" * 40,
        repo_id=DIRECTID_MODEL_ID,
        checkpoint_id="openmed://candidate/directid/synthetic-0602",
    )
    candidate: dict[str, object] = {
        "schema_version": "openmed.training.directid_candidate.v1",
        "artifact_hash": _digest("1"),
        "certified": False,
        "checkpoint_ref": provenance["checkpoint_id"],
        "family": DIRECTID_FAMILY,
        "format": "pytorch-fp32",
        "published": False,
        "ready_for_quantization": True,
        "repo_id": DIRECTID_MODEL_ID,
        "reproducibility_hash": provenance["reproducibility_hash"],
        "run_id": "directid-tiny-synthetic-0602",
        "tier": DIRECTID_TIER,
    }
    training_report: dict[str, object] = {
        "schema_version": "openmed.training.directid_training_report.v1",
        "candidate_checkpoint": dict(candidate),
        "dataset_manifest_hash": dataset_hash,
        "eval_set_hash": eval_hash,
        "family": DIRECTID_FAMILY,
        "raw_phi_persisted": False,
        "restricted_dataset_payloads_persisted": False,
        "run_id": candidate["run_id"],
        "tier": DIRECTID_TIER,
        "debug_canary": SYNTHETIC_CANARY,
    }
    run_manifest: dict[str, object] = {
        "schema_version": "openmed.training.directid_run_manifest.v1",
        "candidate_checkpoint": dict(candidate),
        "contract_ref": DIRECTID_CONTRACT_REF,
        "dataset": {
            "manifest_hash": dataset_hash,
            "manifest_id": DIRECTID_DATASET_MANIFEST_ID,
            "split_hashes": {
                "test": eval_hash,
                "train": _digest("6"),
                "validation": _digest("7"),
            },
        },
        "family": DIRECTID_FAMILY,
        "raw_phi_persisted": False,
        "ready_for_gate_evaluation": True,
        "restricted_dataset_payloads_persisted": False,
        "run_id": candidate["run_id"],
        "tier": DIRECTID_TIER,
        "training_provenance": dict(provenance),
        "training_report_hash": stable_hash(training_report),
    }
    dataset: dict[str, object] = {
        "schema_version": DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION,
        "contract_ref": DIRECTID_CONTRACT_REF,
        "family": DIRECTID_FAMILY,
        "manifest_hash": dataset_hash,
        "manifest_id": DIRECTID_DATASET_MANIFEST_ID,
        "raw_records_persisted": False,
        "source_provenance": [
            {
                "content_hash_required": True,
                "license_id": "CC-BY-4.0",
                "revision": "synthetic-revision-0602",
                "source_class": "synthetic",
                "source_id": "synthetic_directid_release_fixture",
                "source_manifest_hash": _digest("8"),
            }
        ],
        "synthetic_generation": {
            "settings": {"contains_real_phi": False, "seed": 3803},
            "settings_hash": _digest("9"),
        },
        "tier": DIRECTID_TIER,
    }
    labels = DIRECTID_TINY_HEAD_CONTRACT.labels
    recall = {label: 1.0 for label in labels}
    directid: dict[str, object] = {
        "schema_version": DIRECTID_EVIDENCE_SCHEMA_VERSION,
        "contract_ref": DIRECTID_CONTRACT_REF,
        "family": DIRECTID_FAMILY,
        "tier": DIRECTID_TIER,
        "eval_set_hash": eval_hash,
        "leakage_fixture_hash": _digest("c"),
        "patterns_version": "openmed-safety-sweep-v1",
        "per_label_denominators": {label: 100 for label in labels},
        "combined": {
            "per_label_recall": dict(recall),
            "per_label_precision": {label: 0.999 for label in labels},
            "structured_id_recall": 1.0,
        },
        "critical_leakage_count": 0,
        "residual_leakage_rate": 0.0,
        "safety_sweep": {
            "source": "safety_sweep",
            "patterns_version": "openmed-safety-sweep-v1",
            "spans_added": 2,
            "recovered_model_misses": 2,
            "structured_ids_recovered": 2,
        },
        "span_integrity": {"passed": True},
        "gate_evidence": {
            "G1b": {
                "eval_set_hash": eval_hash,
                "per_label_recall": dict(recall),
                "structured_id_recall": 1.0,
            },
            "G3": {
                "critical_leakage_count": 0,
                "leakage_fixture_hash": _digest("c"),
                "residual_leakage_rate": 0.0,
            },
        },
        "debug_canary": SYNTHETIC_CANARY,
    }
    artifact: dict[str, object] = {
        "format": "mlx-8bit",
        "bits": 8,
        "artifact_hash": _digest("d"),
        "artifact_size_bytes": 4_096_000,
        "eval_set_hash": eval_hash,
        "fp_parent_per_label_recall": dict(recall),
        "per_label_recall": dict(recall),
        "param_count": 24_000_000,
        "performance_fixture_hash": _digest("e"),
        "device": "synthetic-phone-profile",
        "sample_count": 800,
        "p50_ms": 20.0,
        "p95_ms": 50.0,
        "ram_mb": 120.0,
        "quant_recall_delta": 0.002,
        "g4": {"passed": True},
        "tiny_tier_fit": {"passed": True},
        "disposition": "accepted",
    }
    quantization: dict[str, object] = {
        "schema_version": "openmed.training.directid_quantization.v1",
        "contract_ref": DIRECTID_CONTRACT_REF,
        "family": DIRECTID_FAMILY,
        "tier": DIRECTID_TIER,
        "candidate": {
            key: candidate[key]
            for key in (
                "artifact_hash",
                "checkpoint_ref",
                "reproducibility_hash",
                "run_id",
            )
        },
        "eval_set_hash": eval_hash,
        "fp_parent_per_label_recall": dict(recall),
        "artifacts": [artifact],
        "accepted_formats": ["mlx-8bit"],
        "quarantined_formats": [],
        "raw_phi_persisted": False,
        "restricted_dataset_payloads_persisted": False,
    }
    return {
        "candidate": candidate,
        "dataset": dataset,
        "directid": directid,
        "provenance": provenance,
        "quantization": quantization,
        "run_manifest": run_manifest,
        "training_report": training_report,
    }


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _check(release, gate: str):
    return next(item for item in release.gate_report.gate_results if item.gate == gate)


def test_written_release_manifest_is_deterministic_json(tmp_path: Path) -> None:
    evidence = _release_evidence()
    release = _build_release(evidence)

    first = release.write(tmp_path / "first")
    second = release.write(tmp_path / "second")

    first_payload = json.loads(first.release_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second.release_manifest.read_text(encoding="utf-8"))
    assert first_payload == second_payload == release.release_manifest
