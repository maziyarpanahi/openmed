from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import pytest

from openmed.training.adapters import (
    ADAPTER_METADATA_SCHEMA_VERSION,
    DEFAULT_ADAPTER_TRAINING_DISCLAIMER,
    AdapterParameterAccounting,
    AdapterTrainingRecipeError,
    DonorToTargetAdapterRecipe,
    LocalTrainingAsset,
    build_donor_to_target_adapter_recipe,
    dry_run_donor_to_target_adapter_recipe,
)


def _local_asset(tmp_path: Path, name: str) -> LocalTrainingAsset:
    asset_path = tmp_path / name
    asset_path.mkdir()
    (asset_path / "manifest.json").write_text(
        json.dumps({"fixture": f"synthetic-{name}"}),
        encoding="utf-8",
    )
    return LocalTrainingAsset(
        asset_id=f"synthetic/{name}",
        path=asset_path,
    )


def _recipe(tmp_path: Path) -> DonorToTargetAdapterRecipe:
    synthetic_gold_path = tmp_path / "synthetic_hi_to_te_gold.jsonl"
    synthetic_gold_path.write_text(
        json.dumps(
            {
                "fixture_id": "synthetic-hi-te-001",
                "label_counts": {"ID_NUM": 1, "PERSON": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return build_donor_to_target_adapter_recipe(
        recipe_id="family-transfer/hi-to-te/v1",
        donor_language="hi-IN",
        target_language="te_IN",
        output_adapter_id="family-transfer/indic-hi-to-te",
        backbone=_local_asset(tmp_path, "clinical-backbone"),
        donor_adapter=_local_asset(tmp_path, "hi-donor-adapter"),
        synthetic_gold=LocalTrainingAsset(
            asset_id="synthetic-gold/indic-hi-to-te/v1",
            path=synthetic_gold_path,
        ),
        parameter_accounting=AdapterParameterAccounting(
            shared_backbone_parameter_count=110_000_000,
            adapter_trainable_parameter_count=524_288,
            task_head_trainable_parameter_count=65_536,
            full_language_model_trainable_parameter_count=110_065_536,
        ),
        provenance="Synthetic Indic family-transfer recipe fixture",
    )


def test_synthetic_donor_to_target_recipe_dry_runs_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("adapter dry-run attempted network access")

    monkeypatch.setattr(socket, "create_connection", unexpected_network)
    recipe = _recipe(tmp_path)
    result = dry_run_donor_to_target_adapter_recipe(recipe)
    payload = result.to_dict()

    assert result.metadata.donor_language == "hi"
    assert result.metadata.target_language == "te"
    assert payload["initialization_source"] == "synthetic/hi-donor-adapter"
    assert payload["network_accessed"] is False
    assert payload["training_started"] is False
    assert payload["verified_local_assets"] == [
        "synthetic/clinical-backbone",
        "synthetic/hi-donor-adapter",
        "synthetic-gold/indic-hi-to-te/v1",
    ]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", result.recipe_hash)


def test_adapter_metadata_is_serializable_path_free_and_complete(
    tmp_path: Path,
) -> None:
    recipe = _recipe(tmp_path)
    result = dry_run_donor_to_target_adapter_recipe(recipe)
    metadata = result.metadata.to_dict()
    encoded = json.dumps(result.to_dict(), sort_keys=True)

    assert metadata["schema_version"] == ADAPTER_METADATA_SCHEMA_VERSION
    assert metadata["adapter_id"] == "family-transfer/indic-hi-to-te"
    assert metadata["donor_language"] == "hi"
    assert metadata["target_language"] == "te"
    assert metadata["backbone"] == "synthetic/clinical-backbone"
    assert metadata["donor_adapter"] == "synthetic/hi-donor-adapter"
    assert metadata["initialization_source"] == "synthetic/hi-donor-adapter"
    assert metadata["shared_backbone_frozen"] is True
    assert metadata["synthetic_gold_source"] == "synthetic-gold/indic-hi-to-te/v1"
    assert metadata["license"] == "apache-2.0"
    assert metadata["disclaimer"] == DEFAULT_ADAPTER_TRAINING_DISCLAIMER
    assert metadata["provenance"] == {
        "initialization": "donor_adapter",
        "recipe": "Synthetic Indic family-transfer recipe fixture",
        "shared_backbone": "synthetic/clinical-backbone",
        "synthetic_gold_source": "synthetic-gold/indic-hi-to-te/v1",
    }
    assert metadata["offline_runnable"] is True
    assert metadata["local_files_only"] is True
    assert str(tmp_path) not in encoded
    assert {"text", "start", "end", "offsets"}.isdisjoint(_walk_keys(result.to_dict()))


def test_parameter_accounting_is_lower_than_full_model_baseline(
    tmp_path: Path,
) -> None:
    accounting = _recipe(tmp_path).parameter_accounting
    payload = accounting.to_dict()

    assert accounting.trainable_parameter_count == 589_824
    assert accounting.frozen_parameter_count == 110_000_000
    assert (
        accounting.trainable_parameter_count
        < accounting.full_language_model_trainable_parameter_count
    )
    assert payload["parameter_reduction_count"] == 109_475_712
    assert payload["trainable_fraction_of_full_model"] < 0.01


def test_parameter_accounting_rejects_full_model_sized_adapter() -> None:
    with pytest.raises(AdapterTrainingRecipeError, match="must be lower"):
        AdapterParameterAccounting(
            shared_backbone_parameter_count=100,
            adapter_trainable_parameter_count=100,
            task_head_trainable_parameter_count=1,
            full_language_model_trainable_parameter_count=101,
        )


def test_recipe_rejects_remote_or_missing_training_assets(tmp_path: Path) -> None:
    with pytest.raises(AdapterTrainingRecipeError, match="local-only"):
        LocalTrainingAsset(
            asset_id="synthetic/remote",
            path="https://example.test/adapter",
        )

    recipe = _recipe(tmp_path)
    donor_path = Path(recipe.donor_adapter.path)
    for child in donor_path.iterdir():
        child.unlink()
    donor_path.rmdir()

    with pytest.raises(AdapterTrainingRecipeError, match="not available locally"):
        dry_run_donor_to_target_adapter_recipe(recipe)


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys
