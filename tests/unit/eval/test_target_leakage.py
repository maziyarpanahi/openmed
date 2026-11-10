"""Tests for the per-family Model-Steward target-leakage config and reader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openmed.eval.target_leakage import (
    DEFAULT_OVER_REDACTION_CEILING,
    DEFAULT_RESIDUAL_LEAKAGE_CEILING,
    DEFAULT_TARGET_LEAKAGE_RATE,
    TARGET_LEAKAGE_PATH,
    TARGET_LEAKAGE_SCHEMA_VERSION,
    LeakageTargets,
    TargetLeakageConfigError,
    TargetLeakageMiss,
    get_targets,
    load_target_leakage_config,
    target_leakage_key,
    validate_target_leakage_config,
)

SEEDED_FAMILIES = ("DirectID", "ClinicalPrivacy", "multilingual-pii")


def _synthetic_config(
    *,
    rate: float = 0.0,
    residual: float = 0.004,
    over_redaction: float = 0.03,
) -> dict:
    """Build a small algorithmic config exercising every schema field."""

    entries = {}
    for index, family in enumerate(SEEDED_FAMILIES):
        # Vary the numbers algorithmically so lookups can be told apart.
        key = target_leakage_key(family, "small", "pytorch")
        entries[key] = {
            "family": family,
            "tier": "small",
            "format": "pytorch",
            "target_leakage_rate": rate,
            "residual_leakage_ceiling": round(residual + index * 0.001, 6),
            "over_redaction_ceiling": round(over_redaction + index * 0.005, 6),
            "steward_owner": f"steward-{index}",
        }
    # One entry that omits ceilings to exercise default inheritance.
    inherit_key = target_leakage_key("DirectID", "base", "onnx")
    entries[inherit_key] = {
        "family": "DirectID",
        "tier": "base",
        "format": "onnx",
    }
    return {
        "schema_version": TARGET_LEAKAGE_SCHEMA_VERSION,
        "steward_owner": "synthetic-council",
        "defaults": {
            "target_leakage_rate": 0.005,
            "residual_leakage_ceiling": 0.005,
            "over_redaction_ceiling": 0.02,
        },
        "entries": entries,
    }


def test_key_is_normalised_across_case_and_separators():
    assert target_leakage_key("DirectID", "Small", "PyTorch") == (
        "directid::small::pytorch"
    )
    assert target_leakage_key("multilingual_pii", None, "onnx") == (
        "multilingual-pii::none::onnx"
    )


def test_loader_returns_configured_targets_per_family_tier_format():
    config = _synthetic_config()

    for index, family in enumerate(SEEDED_FAMILIES):
        targets = get_targets(family, "small", "pytorch", config=config)
        assert isinstance(targets, LeakageTargets)
        assert targets.family == family
        assert targets.target_leakage_rate == 0.0
        assert targets.residual_leakage_ceiling == round(0.004 + index * 0.001, 6)
        assert targets.over_redaction_ceiling == round(0.03 + index * 0.005, 6)
        assert targets.steward_owner == f"steward-{index}"


def test_omitted_ceilings_inherit_config_defaults():
    config = _synthetic_config()

    targets = get_targets("DirectID", "base", "onnx", config=config)

    assert targets.target_leakage_rate == 0.005
    assert targets.residual_leakage_ceiling == 0.005
    assert targets.over_redaction_ceiling == 0.02
    # steward_owner falls back to the config-level owner when unset.
    assert targets.steward_owner == "synthetic-council"


def test_missing_family_lookup_raises_clear_error():
    config = _synthetic_config()

    with pytest.raises(TargetLeakageMiss) as excinfo:
        get_targets("UnknownFamily", "small", "pytorch", config=config)

    message = str(excinfo.value)
    assert "unknownfamily::small::pytorch" in message
    assert "gates/target_leakage.yaml" in message


def test_schema_rejects_out_of_range_leakage_rate():
    config = _synthetic_config()
    key = target_leakage_key("DirectID", "small", "pytorch")
    config["entries"][key]["target_leakage_rate"] = 1.5

    with pytest.raises(TargetLeakageConfigError) as excinfo:
        validate_target_leakage_config(config)
    assert "target_leakage_rate" in str(excinfo.value)


def test_schema_rejects_mismatched_key():
    config = _synthetic_config()
    entry = {
        "family": "DirectID",
        "tier": "small",
        "format": "pytorch",
    }
    config["entries"]["not-the-right-key"] = entry

    with pytest.raises(TargetLeakageConfigError) as excinfo:
        validate_target_leakage_config(config)
    assert "does not match" in str(excinfo.value)


def test_schema_requires_steward_owner():
    config = _synthetic_config()
    config.pop("steward_owner")

    with pytest.raises(TargetLeakageConfigError):
        validate_target_leakage_config(config)


def test_load_and_write_roundtrip_from_disk(tmp_path: Path):
    config = _synthetic_config()
    path = tmp_path / "target_leakage.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    loaded = load_target_leakage_config(path)
    targets = get_targets("ClinicalPrivacy", "small", "pytorch", path=path)

    assert loaded["schema_version"] == TARGET_LEAKAGE_SCHEMA_VERSION
    assert targets.family == "ClinicalPrivacy"


def test_committed_config_is_present_and_seeds_p0_p1_families():
    assert TARGET_LEAKAGE_PATH.exists()
    config = load_target_leakage_config()

    seeded = {entry["family"] for entry in config["entries"].values()}
    assert {"DirectID", "ClinicalPrivacy", "multilingual-pii"} <= seeded

    # Section 6.4: direct-identifier families carry a critical-zero floor.
    directid = get_targets("DirectID", "small", "pytorch")
    assert directid.target_leakage_rate == 0.0
    assert directid.residual_leakage_ceiling <= 0.005


def test_module_defaults_match_section_six_four_numbers():
    assert DEFAULT_RESIDUAL_LEAKAGE_CEILING == 0.005
    assert DEFAULT_TARGET_LEAKAGE_RATE == 0.005
    assert 0.0 <= DEFAULT_OVER_REDACTION_CEILING <= 1.0


def test_leakage_targets_to_dict_is_serialisable():
    config = _synthetic_config()
    targets = get_targets("DirectID", "small", "pytorch", config=config)

    payload = targets.to_dict()
    assert payload["family"] == "DirectID"
    assert set(payload) == {
        "family",
        "tier",
        "format",
        "target_leakage_rate",
        "residual_leakage_ceiling",
        "over_redaction_ceiling",
        "steward_owner",
    }
