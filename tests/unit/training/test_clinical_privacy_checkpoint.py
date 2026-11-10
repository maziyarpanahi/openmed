from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from openmed.core.manifest_schema import validate_manifest_row
from openmed.core.repro_hash import verify_reproducibility
from openmed.eval.datasets import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PRIVACY_MODEL_ID,
    load_clinical_phi_manifest,
)
from openmed.training import (
    CLINICAL_PRIVACY_CHECKPOINT_NAME,
    CLINICAL_PRIVACY_CONTRACT_REF,
    CLINICAL_PRIVACY_OBJECTIVE,
    CLINICAL_PRIVACY_TRAINING_SOURCE_IDS,
    ClinicalPrivacyTrainingError,
    build_clinical_privacy_checkpoint_manifest,
    load_preset,
    resolve_clinical_privacy_training_plan,
    validate_clinical_privacy_checkpoint_manifest,
    write_clinical_privacy_checkpoint_manifest,
)

ARTIFACT_HASH = "sha256:" + "d" * 64
LOCK_HASH = "sha256:" + "e" * 64
GIT_SHA = "a" * 40


def _source_revisions() -> dict[str, str]:
    return {
        source.source_id: f"{source.source_id}-revision-v1"
        for source in load_clinical_phi_manifest().sources
    }


def test_mode_c_resolves_all_sources_and_recall_first_label_weights() -> None:
    dataset_manifest = load_clinical_phi_manifest()
    plan = resolve_clinical_privacy_training_plan(_source_revisions())

    assert plan.model_id == CLINICAL_PRIVACY_MODEL_ID
    assert plan.manifest_id == CLINICAL_PHI_MANIFEST_ID
    assert plan.recipe.mode == "C"
    assert plan.recipe.preset_name == "large_teacher"
    assert plan.recipe.head_contract == CLINICAL_PRIVACY_CONTRACT_REF
    assert tuple(source.source_id for source in plan.sources) == tuple(
        source.source_id for source in dataset_manifest.sources
    )
    assert tuple(source.source_id for source in plan.training_sources()) == (
        CLINICAL_PRIVACY_TRAINING_SOURCE_IDS
    )
    assert all(source.content_included is False for source in plan.sources)
    assert all(
        not source.selected_for_training
        for source in plan.sources
        if source.eval_only or source.requires_credentials
    )

    gate_labels = {
        label
        for requirement in dataset_manifest.gate_families
        for label in requirement.labels
    }
    assert gate_labels <= set(plan.label_weights)
    assert set(dataset_manifest.required_labels()) <= set(plan.label_weights)
    assert set(plan.label_weights) == set(plan.recipe.loss.critical_labels)
    assert set(plan.label_weights.values()) == {4.0}

    thresholds = {threshold.gate: threshold for threshold in plan.thresholds}
    assert set(thresholds) == {"G1a", "G2", "G3"}
    assert thresholds["G1a"].value == pytest.approx(0.990)
    assert thresholds["G2"].value == pytest.approx(0.980)
    assert thresholds["G3"].value == pytest.approx(0.0)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", plan.recipe_config_hash)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", plan.source_revisions_hash)


def test_mode_c_requires_a_revision_for_every_manifest_source() -> None:
    revisions = _source_revisions()
    revisions.pop("n2c2_eval_only")

    with pytest.raises(ClinicalPrivacyTrainingError, match="n2c2_eval_only"):
        resolve_clinical_privacy_training_plan(revisions)


def test_mode_c_rejects_incomplete_clinical_label_weights() -> None:
    recipe = load_preset("C")
    recipe = replace(
        recipe,
        loss=replace(
            recipe.loss,
            critical_labels=recipe.loss.critical_labels[:-1],
        ),
    )

    with pytest.raises(ClinicalPrivacyTrainingError, match="PASSWORD"):
        resolve_clinical_privacy_training_plan(
            _source_revisions(),
            recipe=recipe,
        )


def test_checkpoint_manifest_records_mode_c_thresholds_and_provenance() -> None:
    checkpoint = build_clinical_privacy_checkpoint_manifest(
        _source_revisions(),
        checkpoint_artifact_hash=ARTIFACT_HASH,
        git_sha=GIT_SHA,
        env_lock_digest=LOCK_HASH,
        param_count=434_000_000,
    )
    payload = checkpoint.to_dict()
    row = payload["checkpoints"][0]

    assert payload["checkpoint_name"] == CLINICAL_PRIVACY_CHECKPOINT_NAME
    assert payload["model_id"] == CLINICAL_PRIVACY_MODEL_ID
    assert payload["dataset_manifest"]["id"] == CLINICAL_PHI_MANIFEST_ID
    assert payload["training_recipe"]["mode"] == "C"
    assert payload["training_recipe"]["objective"] == CLINICAL_PRIVACY_OBJECTIVE
    assert payload["training_recipe"]["preset_name"] == "large_teacher"
    assert checkpoint.training_plan.to_dict()["objective"] == (
        CLINICAL_PRIVACY_OBJECTIVE
    )
    assert {item["gate"] for item in payload["thresholds"]} == {
        "G1a",
        "G2",
        "G3",
    }
    assert payload["checkpoint_artifact_hash"] == ARTIFACT_HASH
    assert payload["source_revisions"] == _source_revisions()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", payload["checkpoint_manifest_hash"])

    assert row["repo_id"] == CLINICAL_PRIVACY_MODEL_ID
    assert row["family"] == "ClinicalPrivacy"
    assert row["task"] == "token-classification"
    assert row["released"] is None
    assert validate_manifest_row(row, line_number=1) == []
    assert (
        verify_reproducibility(checkpoint.training_provenance)
        == (row["reproducibility_hash"])
    )
    assert checkpoint.training_provenance["data_manifest_hash"] == (
        checkpoint.training_plan.manifest_hash
    )
    assert checkpoint.training_provenance["recipe_config_hash"] == (
        checkpoint.training_plan.recipe_config_hash
    )


def test_checkpoint_manifest_contains_no_gated_content_and_writes_json(
    tmp_path,
) -> None:
    checkpoint = build_clinical_privacy_checkpoint_manifest(
        _source_revisions(),
        checkpoint_artifact_hash=ARTIFACT_HASH,
        git_sha=GIT_SHA,
        env_lock_digest=LOCK_HASH,
    )

    validate_clinical_privacy_checkpoint_manifest(checkpoint)
    payload = checkpoint.to_dict()
    assert payload["privacy"] == {
        "contains_gated_evaluation_content": False,
        "contains_raw_phi": False,
        "metadata_only_resolution": True,
    }
    gated = {
        source.source_id: source
        for source in checkpoint.training_plan.sources
        if source.requires_credentials
    }
    assert set(gated) == {"i2b2_eval_only", "n2c2_eval_only"}
    assert all(source.content_included is False for source in gated.values())
    assert all(source.selected_for_training is False for source in gated.values())

    output_path = write_clinical_privacy_checkpoint_manifest(
        tmp_path / "checkpoint" / "manifest.json",
        checkpoint,
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_source_revision_changes_checkpoint_manifest_hash() -> None:
    first_revisions = _source_revisions()
    second_revisions = dict(first_revisions)
    second_revisions["shield_public_sample"] = "shield-public-sample-revision-v2"

    first = build_clinical_privacy_checkpoint_manifest(
        first_revisions,
        checkpoint_artifact_hash=ARTIFACT_HASH,
        git_sha=GIT_SHA,
        env_lock_digest=LOCK_HASH,
    )
    second = build_clinical_privacy_checkpoint_manifest(
        second_revisions,
        checkpoint_artifact_hash=ARTIFACT_HASH,
        git_sha=GIT_SHA,
        env_lock_digest=LOCK_HASH,
    )

    assert first.training_plan.source_revisions_hash != (
        second.training_plan.source_revisions_hash
    )
    assert first.checkpoint_manifest_hash != second.checkpoint_manifest_hash
