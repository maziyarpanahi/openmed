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
    "DIRECTID_CONTRACT_REF",
    "DIRECTID_FAMILY",
    "DIRECTID_GATE_CODES",
    "DIRECTID_REQUIRED_ID_SUBTYPES",
    "DIRECTID_TINY_HEAD_CONTRACT",
    "DirectIDContractError",
    "DirectIDHeadContract",
    "DirectIDPresetValidation",
    "DistillationReport",
    "DistillationTargets",
    "HARD_NEGATIVE_CATEGORIES",
    "HardNegativeExample",
    "HardNegativeGenerator",
    "HardNegativeSampler",
    "KDLossBreakdown",
    "LabelRecallDelta",
    "ModeADistillationPipeline",
    "RecipeConfigError",
    "RepairedSpan",
    "TrainingRecipeConfig",
    "WeakLabelDecision",
    "WeakLabelSpan",
    "build_distillation_report",
    "config_hash",
    "count_hard_negatives",
    "compute_kd_loss",
    "decode_repaired_spans",
    "dry_run_recipe",
    "gate_requirements_by_code",
    "load_preset",
    "make_adjudication_item",
    "normalize_weak_span",
    "requires_hard_negative_sampler",
    "run_recipe",
    "runtime_dependencies",
    "sample_hard_negatives",
    "sampler_for_recipe",
    "soft_label_distributions",
    "span_logits_from_repaired_spans",
    "student_backbone_from_tiny_distill_preset",
    "validate_directid_contract",
    "validate_directid_preset",
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
        "DIRECTID_CONTRACT_REF",
        "DIRECTID_FAMILY",
        "DIRECTID_GATE_CODES",
        "DIRECTID_REQUIRED_ID_SUBTYPES",
        "DIRECTID_TINY_HEAD_CONTRACT",
        "DirectIDContractError",
        "DirectIDHeadContract",
        "DirectIDPresetValidation",
        "gate_requirements_by_code",
        "validate_directid_contract",
        "validate_directid_preset",
    }:
        directid = import_module(".directid", __name__)
        return getattr(directid, name)
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
    if name in {
        "AdjudicationCandidate",
        "AdjudicationItem",
        "AdjudicationQueue",
        "make_adjudication_item",
    }:
        adjudication = import_module(".adjudication", __name__)
        return getattr(adjudication, name)
    if name in {
        "DistillationReport",
        "DistillationTargets",
        "KDLossBreakdown",
        "LabelRecallDelta",
        "ModeADistillationPipeline",
        "RepairedSpan",
        "build_distillation_report",
        "compute_kd_loss",
        "decode_repaired_spans",
        "soft_label_distributions",
        "span_logits_from_repaired_spans",
        "student_backbone_from_tiny_distill_preset",
    }:
        distill = import_module(".distill", __name__)
        return getattr(distill, name)
    if name in {
        "WeakLabelDecision",
        "WeakLabelSpan",
        "normalize_weak_span",
        "weak_label_document",
    }:
        weak_labeling = import_module(".weak_labeling", __name__)
        return getattr(weak_labeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
