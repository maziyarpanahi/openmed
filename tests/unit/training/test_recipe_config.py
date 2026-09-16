from __future__ import annotations

import copy
import re
from importlib import import_module

import pytest

from openmed.core.audit import stable_hash
from openmed.eval.datasets import validate_clinical_family_dataset_evidence
from openmed.eval.release_gates import RELEASABLE, GateCheck, GateReport
from openmed.training import (
    CLINICAL_MODEL_FAMILY_SPECS,
    CONFIG_SCHEMA_VERSION,
    MAX_LORA_TRAINABLE_RATIO,
    PRESET_BY_MODE,
    RecipeConfigError,
    TrainingRecipeConfig,
    build_clinical_family_release,
    clinical_family_recipe_hash,
    clinical_model_family_spec,
    config_hash,
    load_preset,
    run_recipe,
    runtime_dependencies,
)


def test_all_committed_presets_validate_and_dry_run_emits_hash():
    hashes = set()

    for mode, preset_name in PRESET_BY_MODE.items():
        config = load_preset(mode)
        assert config.schema_version == CONFIG_SCHEMA_VERSION
        assert config.mode == mode
        assert config.preset_name == preset_name
        assert config.hard_negatives_required is True
        assert config.lora.target_trainable_ratio < MAX_LORA_TRAINABLE_RATIO
        assert config.loss.name == "focal_class_weighted"
        assert config.loss.class_weighted is True
        assert config.loss.critical_label_weight > 1

        result = run_recipe(mode)
        assert result.mode == mode
        assert result.preset_name == preset_name
        assert result.seed == config.seed
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", result.config_hash)
        hashes.add(result.config_hash)

    assert len(hashes) == 3


def test_recipe_entrypoint_accepts_preset_names_too():
    result = run_recipe("laptop_lora")

    assert result.mode == "B"
    assert result.output_tier == "laptop"
    assert result.quant_default == "int8"


def test_config_hash_is_deterministic_for_valid_config():
    config = load_preset("A")

    assert config_hash(config) == config_hash(config)


def test_missing_hard_negative_flag_is_rejected():
    raw = load_preset("A").to_dict()
    del raw["hard_negatives_required"]

    with pytest.raises(RecipeConfigError, match="hard_negatives_required"):
        TrainingRecipeConfig.from_mapping(raw)


def test_lora_ratio_over_schema_limit_is_rejected():
    raw = load_preset("B").to_dict()
    raw["lora"]["target_trainable_ratio"] = 0.02

    with pytest.raises(RecipeConfigError, match="target_trainable_ratio"):
        TrainingRecipeConfig.from_mapping(raw)


def test_lora_ratio_equal_to_limit_is_rejected():
    raw = load_preset("B").to_dict()
    raw["lora"]["target_trainable_ratio"] = MAX_LORA_TRAINABLE_RATIO

    with pytest.raises(RecipeConfigError, match="target_trainable_ratio"):
        TrainingRecipeConfig.from_mapping(raw)


def test_schema_rejects_non_focal_class_weighted_loss():
    raw = copy.deepcopy(load_preset("C").to_dict())
    raw["loss"]["name"] = "cross_entropy"

    with pytest.raises(RecipeConfigError, match="focal_class_weighted"):
        TrainingRecipeConfig.from_mapping(raw)


def test_recipe_reuses_existing_anonymizer_merger_and_decoding_imports():
    dependencies = runtime_dependencies()
    modules = dependencies.module_names()

    assert modules["anonymizer"] == "openmed.core.anonymizer.engine"
    assert modules["merger"] == "openmed.core.pii_entity_merger"
    assert modules["decoding"] == "openmed.core.decoding.viterbi"
    assert callable(dependencies.merger)
    assert callable(dependencies.decoder)


def test_clinical_family_targets_cover_every_family_and_respect_recipe_modes():
    laptop = load_preset("B")
    teacher = load_preset("C")

    assert set(laptop.clinical_family_targets) == set(CLINICAL_MODEL_FAMILY_SPECS)
    assert teacher.clinical_family_targets == ("relex_med", "relex_ade", "link")
    assert clinical_model_family_spec("doctype").allowed_modes == ("A", "B")
    assert clinical_model_family_spec("relex_med").allowed_modes == ("B", "C")
    assert clinical_model_family_spec("link").runtime_ref == (
        "openmed.clinical.grounding:ground"
    )
    for spec in CLINICAL_MODEL_FAMILY_SPECS.values():
        module_name, attribute = spec.runtime_ref.split(":", maxsplit=1)
        assert hasattr(import_module(module_name), attribute)
        label_module, versioned_attribute = spec.label_set_ref.split(":", maxsplit=1)
        label_attribute = versioned_attribute.partition("@")[0]
        assert hasattr(import_module(label_module), label_attribute)


@pytest.mark.parametrize(
    ("family", "manifest_family", "repo_name", "task", "benchmark"),
    [
        (
            "doctype",
            "DocType",
            "OpenMed/OpenMed-DocType-Base",
            "text-classification",
            "synthetic section/doctype",
        ),
        (
            "section",
            "Section",
            "OpenMed/OpenMed-Section-Base",
            "token-classification",
            "synthetic section/doctype",
        ),
        (
            "relex_med",
            "RelEx-Med",
            "OpenMed/OpenMed-RelEx-Med-Base",
            "relation-extraction",
            "DrugProt",
        ),
        (
            "relex_ade",
            "RelEx-ADE",
            "OpenMed/OpenMed-RelEx-ADE-Base",
            "relation-extraction",
            "DrugProt",
        ),
        (
            "link",
            "Link",
            "OpenMed/OpenMed-Link-HPO-Base",
            "feature-extraction",
            "MedMentions st21pv",
        ),
    ],
)
def test_clinical_family_release_requires_signed_g5_g6_and_writes_artifacts(
    tmp_path,
    family,
    manifest_family,
    repo_name,
    task,
    benchmark,
):
    signing_key = "synthetic-clinical-family-release-key"
    evidence = _clinical_dataset_evidence(family)
    manifest = _clinical_manifest(
        family=manifest_family,
        repo_id=repo_name,
        task=task,
        benchmark=benchmark,
        evidence=evidence,
    )
    report = _clinical_gate_report(
        family=manifest_family,
        repo_id=repo_name,
        signing_key=signing_key,
    )

    release = build_clinical_family_release(
        family,
        recipe_mode="B",
        manifest_row=manifest,
        gate_report=report,
        dataset_evidence=evidence,
        signing_key=signing_key,
    )
    paths = release.write(tmp_path / family)

    assert release.to_dict() == release.to_dict()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", release.to_dict()["release_hash"])
    assert "| G5 tier fit | passed" in release.model_card
    assert "| G6 latency | passed" in release.model_card
    assert "Corpus rows and restricted-vocabulary content are not bundled" in (
        release.model_card
    )
    if family == "link":
        assert "top1_accuracy=0.71" in release.model_card
    assert all(path.is_file() for path in paths.values())


def test_clinical_family_release_rejects_unsigned_or_failed_gate_report():
    family = "relex_med"
    evidence = _clinical_dataset_evidence(family)
    manifest = _clinical_manifest(
        family="RelEx-Med",
        repo_id="OpenMed/OpenMed-RelEx-Med-Base",
        task="relation-extraction",
        benchmark="DrugProt",
        evidence=evidence,
    )
    report = _clinical_gate_report(
        family="RelEx-Med",
        repo_id="OpenMed/OpenMed-RelEx-Med-Base",
        signing_key="different-key",
    )

    with pytest.raises(RecipeConfigError, match="signature"):
        build_clinical_family_release(
            family,
            recipe_mode="B",
            manifest_row=manifest,
            gate_report=report,
            dataset_evidence=evidence,
            signing_key="expected-key",
        )

    failed_report = _clinical_gate_report(
        family="RelEx-Med",
        repo_id="OpenMed/OpenMed-RelEx-Med-Base",
        signing_key="expected-key",
        g5_passed=False,
    )
    with pytest.raises(RecipeConfigError, match="failed G5"):
        build_clinical_family_release(
            family,
            recipe_mode="B",
            manifest_row=manifest,
            gate_report=failed_report,
            dataset_evidence=evidence,
            signing_key="expected-key",
        )


def _clinical_dataset_evidence(family):
    digest = "sha256:" + "2" * 64
    common = {
        "manifest_hash": digest,
        "corpus_bundled": False,
        "restricted_vocabulary_bundled": False,
    }
    if family in {"doctype", "section"}:
        return {
            "synthetic_section_doctype": {
                **common,
                "uses": ["train", "eval"],
                "metrics": {"micro_f1": 0.95},
            }
        }
    if family in {"relex_med", "relex_ade"}:
        return {
            "drugprot": {
                **common,
                "uses": ["train", "eval"],
                "metrics": {"micro_f1": 0.91},
            }
        }
    return {
        "redistributable_vocabulary": {
            **common,
            "uses": ["train"],
            "metrics": {},
            "redistributable": True,
            "vocab": "HPO",
        },
        "medmentions": {
            **common,
            "uses": ["eval"],
            "metrics": {"top1_accuracy": 0.71},
        },
    }


def _clinical_manifest(*, family, repo_id, task, benchmark, evidence):
    recipe = load_preset("B")
    validated_evidence = validate_clinical_family_dataset_evidence(
        {
            "DocType": "doctype",
            "Section": "section",
            "RelEx-Med": "relex_med",
            "RelEx-ADE": "relex_ade",
            "Link": "link",
        }[family],
        evidence,
    )
    reproducibility_hash = "sha256:" + "3" * 64
    return {
        "repo_id": repo_id,
        "family": family,
        "task": task,
        "languages": ["en"],
        "tier": "Base",
        "param_count": 180_000_000,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/base-clinical-encoder",
        "formats": ["int8"],
        "canonical_labels": ["CLINICAL_LABEL"],
        "benchmark": {"dataset": benchmark, "micro_f1": 0.91, "recall": 0.92},
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": reproducibility_hash,
        "released": "2026-08-04",
        "latency_ms": {"p50": 75.0, "p95": 180.0},
        "peak_ram_mb": {"measured": 512.0},
        "training_provenance": {
            "base_model": "OpenMed/base-clinical-encoder",
            "base_model_revision": "synthetic-revision",
            "data_manifest_hash": stable_hash(validated_evidence),
            "env_lock_digest": "sha256:" + "4" * 64,
            "git_sha": "5" * 40,
            "recipe_config_hash": clinical_family_recipe_hash(
                {
                    "DocType": "doctype",
                    "Section": "section",
                    "RelEx-Med": "relex_med",
                    "RelEx-ADE": "relex_ade",
                    "Link": "link",
                }[family],
                recipe.mode,
            ),
            "reproducibility_hash": reproducibility_hash,
            "rng_seeds": {"python": recipe.seed},
        },
    }


def _clinical_gate_report(*, family, repo_id, signing_key, g5_passed=True):
    return GateReport(
        repo_id=repo_id,
        family=family,
        tier="Base",
        param_count=180_000_000,
        format="int8",
        per_label_recall={"CLINICAL_LABEL": 0.92},
        per_label_precision={"CLINICAL_LABEL": 0.90},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=75.0,
        p95_ms=180.0,
        ram_mb=512.0,
        eval_set_hash="sha256:" + "6" * 64,
        leakage_fixture_hash="sha256:" + "7" * 64,
        decision=RELEASABLE,
        gate_results=(
            GateCheck("G5", g5_passed, reason="ok" if g5_passed else "over budget"),
            GateCheck("G6", True, reason="ok"),
        ),
    ).sign(signing_key)
