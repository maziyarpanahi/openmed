from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.training import (
    DOCTYPE_SECTION_DRY_RUN_SCHEMA_VERSION,
    DOCTYPE_SECTION_LABEL_SET_REF,
    DOCTYPE_SECTION_PRESET,
    PRESET_BY_MODE,
    RecipeConfigError,
    TrainingRecipeConfig,
    dry_run_recipe,
    load_preset,
    resolve_doctype_section_head_contract,
    run_recipe,
)


def test_doctype_section_config_validates_and_resolves_dual_head_contract():
    config = load_preset(DOCTYPE_SECTION_PRESET)

    assert config.preset_name == DOCTYPE_SECTION_PRESET
    assert config.mode == "B"
    assert config.label_set_ref == DOCTYPE_SECTION_LABEL_SET_REF
    contract = resolve_doctype_section_head_contract(config)
    assert contract["section_head"]["scheme"] == "BIO"
    assert contract["section_head"]["labels"][0] == "O"
    assert contract["doctype_head"]["type"] == "sequence_classification"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda raw: raw.pop("head_contract"), "head_contract"),
        (
            lambda raw: raw["head_contract"].pop("doctype_head"),
            "doctype_head",
        ),
        (
            lambda raw: raw["head_contract"]["section_head"].__setitem__(
                "scheme",
                "IOBES",
            ),
            "section_head.scheme",
        ),
    ],
)
def test_doctype_section_rejects_missing_or_malformed_head_contract_with_actionable_error(
    mutator,
    message,
):
    raw = load_preset(DOCTYPE_SECTION_PRESET).to_dict()
    mutator(raw)

    with pytest.raises(RecipeConfigError, match=message):
        TrainingRecipeConfig.from_mapping(raw)


def test_doctype_section_dry_run_reproducibility_hash_is_stable_for_same_seed():
    first = run_recipe(DOCTYPE_SECTION_PRESET)
    second = run_recipe(DOCTYPE_SECTION_PRESET)

    assert first.reproducibility_hash == second.reproducibility_hash
    assert first.manifest is not None
    assert first.manifest["schema_version"] == DOCTYPE_SECTION_DRY_RUN_SCHEMA_VERSION
    assert first.manifest["case_count"] == 5

    changed_seed = replace(load_preset(DOCTYPE_SECTION_PRESET), seed=first.seed + 1)
    changed = dry_run_recipe(changed_seed)
    assert changed.reproducibility_hash != first.reproducibility_hash


def test_doctype_section_declares_tier_and_quantization_fit_against_eval_tiers():
    config = load_preset(DOCTYPE_SECTION_PRESET)
    result = run_recipe(DOCTYPE_SECTION_PRESET)

    assert config.output_tier in {"G5", "G6"}
    assert config.quantization.default == "int8"
    assert config.quantization.allow_fp32_fallback is True
    assert result.manifest is not None
    assert result.manifest["tier_budget"]["ram_mb_max"] == 900


def test_existing_pii_presets_still_validate_after_doctype_section_extension():
    for mode, preset_name in PRESET_BY_MODE.items():
        config = load_preset(mode)
        result = run_recipe(preset_name)

        assert config.preset_name == preset_name
        assert result.preset_name == preset_name
        assert result.reproducibility_hash is None
