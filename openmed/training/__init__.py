"""Standardized OpenMed training recipes."""

from importlib import import_module
from typing import Any

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "MAX_LORA_TRAINABLE_RATIO",
    "PRESET_BY_MODE",
    "AdjudicationCandidate",
    "AdjudicationItem",
    "AdjudicationQueue",
    "DryRunResult",
    "RecipeConfigError",
    "TrainingRecipeConfig",
    "WeakLabelDecision",
    "WeakLabelSpan",
    "config_hash",
    "dry_run_recipe",
    "load_preset",
    "make_adjudication_item",
    "normalize_weak_span",
    "run_recipe",
    "runtime_dependencies",
    "weak_label_document",
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
        "AdjudicationCandidate",
        "AdjudicationItem",
        "AdjudicationQueue",
        "make_adjudication_item",
    }:
        adjudication = import_module(".adjudication", __name__)
        return getattr(adjudication, name)
    if name in {
        "WeakLabelDecision",
        "WeakLabelSpan",
        "normalize_weak_span",
        "weak_label_document",
    }:
        weak_labeling = import_module(".weak_labeling", __name__)
        return getattr(weak_labeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
