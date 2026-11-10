from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from pathlib import Path

import pytest

from openmed.core.repro_hash import (
    compute_canonical_payload_hash,
    verify_reproducibility,
)
from openmed.training import (
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDDistillationError,
    DirectIDHardNegativeOutcome,
    DirectIDLabelOutcome,
    DirectIDTrainingOutput,
    run_directid_tiny_distillation,
)
from openmed.training.hard_negatives import HARD_NEGATIVE_CATEGORIES

FIXED_GIT_SHA = "1" * 40
SYNTHETIC_MARKER = "synthetic.user@example.test"
DIRECTID_SPLITS = ("train", "validation", "test")


def _dataset_evidence() -> dict[str, object]:
    synthetic_settings = {
        "contains_real_phi": False,
        "hard_negative_categories": list(HARD_NEGATIVE_CATEGORIES),
        "records_per_id_subtype": 16,
        "records_per_label": 16,
        "schema_version": "openmed.training.directid_synthetic.v1",
        "seed": 3801,
    }
    manifest_descriptor = {
        "contract_ref": DIRECTID_TINY_HEAD_CONTRACT.contract_ref,
        "fixture": "synthetic-offline",
        "manifest_id": "openmed-directid-tiny-dataset-v1",
    }
    return {
        "contract_ref": DIRECTID_TINY_HEAD_CONTRACT.contract_ref,
        "family": DIRECTID_TINY_HEAD_CONTRACT.family,
        "manifest_hash": compute_canonical_payload_hash(manifest_descriptor),
        "manifest_id": manifest_descriptor["manifest_id"],
        "raw_records_persisted": False,
        "schema_version": "openmed.training.directid_dataset_evidence.v1",
        "splits": {
            split_name: {
                "dataset_hash": compute_canonical_payload_hash(
                    {"fixture": "synthetic-offline", "split": split_name}
                ),
                "hard_negative_category_counts": {
                    category: 4 for category in HARD_NEGATIVE_CATEGORIES
                },
                "id_subtype_counts": {
                    subtype: 16 for subtype in DIRECTID_TINY_HEAD_CONTRACT.id_subtypes
                },
                "label_counts": {
                    label: 16 for label in DIRECTID_TINY_HEAD_CONTRACT.labels
                },
            }
            for split_name in DIRECTID_SPLITS
        },
        "synthetic_generation": {
            "settings": synthetic_settings,
            "settings_hash": compute_canonical_payload_hash(synthetic_settings),
        },
        "tier": DIRECTID_TINY_HEAD_CONTRACT.tier,
    }


def _label_outcomes() -> tuple[DirectIDLabelOutcome, ...]:
    return tuple(
        DirectIDLabelOutcome(
            label=label,
            true_positive=999,
            false_negative=1,
            false_positive=2,
        )
        for label in DIRECTID_TINY_HEAD_CONTRACT.labels
    )


def _hard_negative_outcomes() -> tuple[DirectIDHardNegativeOutcome, ...]:
    return tuple(
        DirectIDHardNegativeOutcome(
            category=category,
            example_count=100,
            false_positive_count=int(index == 0),
        )
        for index, category in enumerate(HARD_NEGATIVE_CATEGORIES)
    )


def _trainer(checkpoint_path: Path):
    def train(context):
        assert context.recipe.mode == "A"
        assert context.recipe.preset_name == "tiny_distill"
        assert context.mode_a_pipeline_ref.endswith("ModeADistillationPipeline@v1")
        assert context.pipeline_type.__name__ == "ModeADistillationPipeline"
        assert context.rng_seeds == {"numpy": 3801, "python": 3801, "torch": 3801}
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "config.json").write_text(
            json.dumps(
                {
                    "fixture": SYNTHETIC_MARKER,
                    "labels": list(context.contract.labels),
                }
            ),
            encoding="utf-8",
        )
        (checkpoint_path / "model.safetensors").write_bytes(
            b"synthetic-offline-directid-checkpoint"
        )
        return DirectIDTrainingOutput(
            checkpoint_path=checkpoint_path,
            teacher_recall_by_label={label: 0.998 for label in context.contract.labels},
            label_outcomes=_label_outcomes(),
            hard_negative_outcomes=_hard_negative_outcomes(),
            training_steps=240,
            completed_epochs=3.0,
            final_loss=random.random(),
        )

    return train


def _run(tmp_path: Path, name: str):
    return run_directid_tiny_distillation(
        dataset_evidence=_dataset_evidence(),
        teacher_id="OpenMed/PII-Teacher-Synthetic",
        teacher_revision="teacher-v1",
        trainer=_trainer(tmp_path / f"{name}-checkpoint"),
        output_dir=tmp_path / f"{name}-evidence",
        git_sha=FIXED_GIT_SHA,
    )


def test_mode_a_run_emits_quantization_ready_hash_only_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _run(tmp_path, "first")

    assert artifacts.candidate.checkpoint_ref.startswith(
        "openmed://candidate/OpenMed-PII-DirectID-Tiny/"
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", artifacts.candidate.artifact_hash)
    assert artifacts.candidate.to_dict()["ready_for_quantization"] is True
    assert artifacts.candidate.to_dict()["certified"] is False
    assert artifacts.candidate.to_dict()["published"] is False
    assert verify_reproducibility(artifacts.training_provenance).startswith("sha256:")

    report = artifacts.training_report
    assert report["structured_id_recall"] == pytest.approx(0.999)
    assert report["per_label_recall"] == {
        label: pytest.approx(0.999) for label in DIRECTID_TINY_HEAD_CONTRACT.labels
    }
    assert report["critical_label_recall_min"] == pytest.approx(0.999)
    assert (
        report["eval_set_hash"]
        == artifacts.run_manifest["dataset"]["split_hashes"]["test"]
    )
    assert set(report["critical_label_recall"]) == set(
        DIRECTID_TINY_HEAD_CONTRACT.critical_labels
    )
    assert len(report["per_label_metrics"]) == len(DIRECTID_TINY_HEAD_CONTRACT.labels)
    assert report["hard_negative_metrics"]["false_positive_rate"] == pytest.approx(
        1 / 400
    )
    assert {
        row["category"] for row in report["hard_negative_metrics"]["per_category"]
    } == set(HARD_NEGATIVE_CATEGORIES)
    assert report["distillation"]["recall_gate_passed"] is True

    manifest = artifacts.run_manifest
    assert manifest["mode"] == "A"
    assert manifest["ready_for_quantization"] is True
    assert manifest["ready_for_gate_evaluation"] is True
    assert manifest["final_certification_performed"] is False
    assert manifest["publishing_performed"] is False
    assert manifest["candidate_checkpoint"]["artifact_hash"] == (
        artifacts.candidate.artifact_hash
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["recipe_config_hash"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["preset_config_hash"])
    assert set(manifest["dataset"]["split_hashes"]) == set(DIRECTID_SPLITS)

    evidence_text = "".join(
        path.read_text(encoding="utf-8")
        for path in (
            artifacts.candidate_path,
            artifacts.training_report_path,
            artifacts.run_manifest_path,
            artifacts.training_provenance_path,
        )
    )
    assert SYNTHETIC_MARKER not in evidence_text
    assert str(artifacts.checkpoint_path) not in evidence_text
    assert '"raw_phi_persisted": false' in evidence_text
    assert '"restricted_dataset_payloads_persisted": false' in evidence_text


def test_mode_a_run_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")

    assert first.candidate == second.candidate
    assert first.training_report == second.training_report
    assert first.run_manifest == second.run_manifest
    assert first.training_provenance == second.training_provenance
    assert (
        first.training_report_path.read_bytes()
        == second.training_report_path.read_bytes()
    )
    assert first.run_manifest_path.read_bytes() == second.run_manifest_path.read_bytes()


def test_run_fails_closed_for_incomplete_label_or_negative_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"

    def incomplete_labels(context):
        result = _trainer(checkpoint)(context)
        return DirectIDTrainingOutput(
            checkpoint_path=result.checkpoint_path,
            teacher_recall_by_label=result.teacher_recall_by_label,
            label_outcomes=tuple(result.label_outcomes[:-1]),
            hard_negative_outcomes=result.hard_negative_outcomes,
            training_steps=result.training_steps,
            completed_epochs=result.completed_epochs,
            final_loss=result.final_loss,
        )

    output = tmp_path / "evidence"
    with pytest.raises(DirectIDDistillationError, match="every DirectID"):
        run_directid_tiny_distillation(
            dataset_evidence=_dataset_evidence(),
            teacher_id="OpenMed/PII-Teacher-Synthetic",
            teacher_revision="teacher-v1",
            trainer=incomplete_labels,
            output_dir=output,
            git_sha=FIXED_GIT_SHA,
        )
    assert not output.exists()

    def incomplete_negatives(context):
        result = _trainer(tmp_path / "checkpoint-two")(context)
        return DirectIDTrainingOutput(
            checkpoint_path=result.checkpoint_path,
            teacher_recall_by_label=result.teacher_recall_by_label,
            label_outcomes=result.label_outcomes,
            hard_negative_outcomes=tuple(result.hard_negative_outcomes[:-1]),
            training_steps=result.training_steps,
            completed_epochs=result.completed_epochs,
            final_loss=result.final_loss,
        )

    with pytest.raises(DirectIDDistillationError, match="every required category"):
        run_directid_tiny_distillation(
            dataset_evidence=_dataset_evidence(),
            teacher_id="OpenMed/PII-Teacher-Synthetic",
            teacher_revision="teacher-v1",
            trainer=incomplete_negatives,
            output_dir=output,
            git_sha=FIXED_GIT_SHA,
        )
    assert not output.exists()


def test_run_rejects_raw_dataset_evidence_and_phi_shaped_identifiers(
    tmp_path: Path,
) -> None:
    unsafe_evidence = deepcopy(_dataset_evidence())
    unsafe_evidence["raw_records_persisted"] = True
    output = tmp_path / "evidence"

    with pytest.raises(DirectIDDistillationError, match="raw_records_persisted"):
        run_directid_tiny_distillation(
            dataset_evidence=unsafe_evidence,
            teacher_id="OpenMed/PII-Teacher-Synthetic",
            teacher_revision="teacher-v1",
            trainer=_trainer(tmp_path / "checkpoint"),
            output_dir=output,
            git_sha=FIXED_GIT_SHA,
        )
    assert not output.exists()

    with pytest.raises(DirectIDDistillationError, match="PHI-shaped"):
        run_directid_tiny_distillation(
            dataset_evidence=_dataset_evidence(),
            teacher_id="patient-123-45-6789",
            teacher_revision="teacher-v1",
            trainer=_trainer(tmp_path / "checkpoint-two"),
            output_dir=output,
            git_sha=FIXED_GIT_SHA,
        )
    assert not output.exists()
