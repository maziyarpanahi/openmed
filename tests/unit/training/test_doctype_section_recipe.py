"""Tests for the reproducible DocType/Section recipe dry-run harness."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from openmed.eval.tiers import TIERS
from openmed.training.recipe import (
    CONFIG_SCHEMA_VERSION,
    DOCTYPE_SECTION_CONFIG_PATH,
    DOCTYPE_SECTION_DRY_RUN_SCHEMA_VERSION,
    DOCTYPE_SECTION_HEAD_SCHEMA_VERSION,
    DOCTYPE_SECTION_LABEL_SET_REF,
    DOCTYPE_SECTION_PRESET,
    DOCTYPE_SECTION_REQUIRED_GATES,
    PRESET_BY_MODE,
    RecipeConfigError,
    TrainingRecipeConfig,
    load_config_file,
    load_preset,
    main,
    resolve_doctype_section_head_contract,
    run_doctype_section_dry_run,
)
from openmed.training.synthetic.section_labels import (
    CANONICAL_DOCUMENT_TYPES,
    CANONICAL_SECTION_LABELS,
    load_section_dataset,
    load_section_manifest,
)


def _raw_config() -> dict[str, object]:
    return load_config_file(DOCTYPE_SECTION_CONFIG_PATH)


def test_doctype_section_recipe_v1_resolves_dual_head_and_tier_contract() -> None:
    config = load_preset(DOCTYPE_SECTION_PRESET)
    resolved = resolve_doctype_section_head_contract(config)

    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.mode == "B"
    assert config.label_set_ref == DOCTYPE_SECTION_LABEL_SET_REF
    assert config.clinical_family_targets == ("doctype", "section")
    assert config.head_contract is not None
    assert config.head_contract.schema_version == DOCTYPE_SECTION_HEAD_SCHEMA_VERSION
    assert config.required_gates == DOCTYPE_SECTION_REQUIRED_GATES == ("G5", "G6")
    assert set(resolved.section_labels) == CANONICAL_SECTION_LABELS
    assert set(resolved.document_types) == CANONICAL_DOCUMENT_TYPES
    assert resolved.section_task == "token_classification"
    assert resolved.section_encoding == "BIO"
    assert resolved.doctype_task == "sequence_classification"
    assert resolved.doctype_pooling == "first_token_window"

    tier = TIERS[config.output_tier]
    assert config.output_tier == "Base"
    assert config.quantization.default.upper() in str(tier["default_format"]).upper()
    assert config.quantization.allow_fp32_fallback is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.pop("head_contract"), "requires a head_contract mapping"),
        (
            lambda raw: raw["head_contract"]["section_bio"].update(
                {"encoding": "IOB2"}
            ),
            "head_contract.section_bio.encoding must be 'BIO'",
        ),
        (
            lambda raw: raw["head_contract"]["doctype"].pop("pooling"),
            "head_contract.doctype missing required field.*pooling",
        ),
        (
            lambda raw: raw.update({"label_set_ref": "missing:labels@v1"}),
            "doctype_section_lora label_set_ref must be",
        ),
    ],
)
def test_malformed_doctype_section_contract_fails_fast(mutation, message: str) -> None:
    raw = copy.deepcopy(_raw_config())
    mutation(raw)

    with pytest.raises(RecipeConfigError, match=message):
        TrainingRecipeConfig.from_mapping(raw)


def test_missing_label_set_ref_is_actionable() -> None:
    raw = copy.deepcopy(_raw_config())
    del raw["label_set_ref"]

    with pytest.raises(
        RecipeConfigError, match="missing required field.*label_set_ref"
    ):
        TrainingRecipeConfig.from_mapping(raw)


def test_doctype_section_tier_and_quantization_must_match_eval_tiers() -> None:
    raw = copy.deepcopy(_raw_config())
    raw["quantization"]["default"] = "bf16"

    with pytest.raises(RecipeConfigError, match="incompatible with Base tier"):
        TrainingRecipeConfig.from_mapping(raw)

    raw = copy.deepcopy(_raw_config())
    raw["required_gates"] = ["G5"]
    with pytest.raises(RecipeConfigError, match="required_gates must be.*G5.*G6"):
        TrainingRecipeConfig.from_mapping(raw)


def test_existing_pii_recipe_presets_still_validate() -> None:
    for mode, preset_name in PRESET_BY_MODE.items():
        config = load_preset(mode)

        assert config.preset_name == preset_name
        assert config.mode == mode
        assert isinstance(config.head_contract, (str, type(None)))
        assert config.required_gates == ()


def test_seed_pinned_dry_run_is_byte_stable_and_never_trains(tmp_path: Path) -> None:
    first = run_doctype_section_dry_run(tmp_path / "first")
    second = run_doctype_section_dry_run(tmp_path / "second")

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.reproducibility_hash)
    assert first.reproducibility_hash == second.reproducibility_hash
    assert first.to_dict() == second.to_dict()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.dataset_path.read_bytes() == second.dataset_path.read_bytes()
    assert (
        first.dataset_manifest_path.read_bytes()
        == second.dataset_manifest_path.read_bytes()
    )

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = load_section_manifest(first.dataset_manifest_path)
    dataset = load_section_dataset(first.dataset_path)
    assert manifest["schema_version"] == DOCTYPE_SECTION_DRY_RUN_SCHEMA_VERSION
    assert manifest["dry_run"] is True
    assert manifest["training_launched"] is False
    assert manifest["network_required"] is False
    assert manifest["dataset"]["record_count"] == len(CANONICAL_DOCUMENT_TYPES)
    assert manifest["dataset"]["dataset_hash"] == dataset_manifest["dataset_hash"]
    assert len(dataset) == len(CANONICAL_DOCUMENT_TYPES)
    assert {row["doc_type"] for row in dataset} == CANONICAL_DOCUMENT_TYPES
    assert "text" not in json.dumps(dataset_manifest).casefold()


def test_dry_run_rejects_empty_dataset_without_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run"

    with pytest.raises(ValueError, match="positive integer"):
        run_doctype_section_dry_run(output_dir, dataset_record_count=0)

    assert not output_dir.exists()


def test_cli_entrypoint_emits_manifest_without_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            DOCTYPE_SECTION_PRESET,
            "--output-dir",
            str(tmp_path / "cli"),
            "--dataset-record-count",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["training_launched"] is False
    assert payload["dataset"]["record_count"] == 2
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", payload["reproducibility_hash"])
