"""Standardized OpenMed training recipes."""

from importlib import import_module
from typing import Any

__all__ = [
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
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        recipe = import_module(".recipe", __name__)
        return getattr(recipe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
