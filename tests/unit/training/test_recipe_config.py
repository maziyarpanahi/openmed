from __future__ import annotations

import copy
import re

import pytest

from openmed.training import (
    CONFIG_SCHEMA_VERSION,
    MAX_LORA_TRAINABLE_RATIO,
    PRESET_BY_MODE,
    RecipeConfigError,
    TrainingRecipeConfig,
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
