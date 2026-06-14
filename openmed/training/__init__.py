"""Standardized OpenMed training recipes."""

from importlib import import_module
from typing import Any

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "MAX_LORA_TRAINABLE_RATIO",
    "PRESET_BY_MODE",
    "HARD_NEGATIVE_CATEGORIES",
    "DryRunResult",
    "HardNegativeExample",
    "HardNegativeGenerator",
    "HardNegativeSampler",
    "RecipeConfigError",
    "TrainingRecipeConfig",
    "config_hash",
    "count_hard_negatives",
    "dry_run_recipe",
    "load_preset",
    "requires_hard_negative_sampler",
    "run_recipe",
    "runtime_dependencies",
    "sample_hard_negatives",
    "sampler_for_recipe",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CONFIG_SCHEMA_VERSION",
        "MAX_LORA_TRAINABLE_RATIO",
        "PRESET_BY_MODE",
        "DryRunResult",
        "RecipeConfigError",
        "TrainingRecipeConfig",
        "config_hash",
        "dry_run_recipe",
        "load_preset",
        "run_recipe",
        "runtime_dependencies",
    }:
        recipe = import_module(".recipe", __name__)
        return getattr(recipe, name)
    if name in {
        "HARD_NEGATIVE_CATEGORIES",
        "HardNegativeExample",
        "HardNegativeGenerator",
        "HardNegativeSampler",
        "count_hard_negatives",
        "requires_hard_negative_sampler",
        "sample_hard_negatives",
        "sampler_for_recipe",
    }:
        hard_negatives = import_module(".hard_negatives", __name__)
        return getattr(hard_negatives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
