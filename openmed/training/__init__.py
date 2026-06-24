"""Standardized OpenMed training recipes."""

from importlib import import_module
from typing import Any

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DAPT_CORPUS_MANIFEST_PATH",
    "DAPT_CORPUS_SCHEMA_VERSION",
    "MAX_LORA_TRAINABLE_RATIO",
    "PRESET_BY_MODE",
    "AdjudicationCandidate",
    "AdjudicationItem",
    "AdjudicationQueue",
    "CorpusManifestError",
    "DaptCorpusAssemblyResult",
    "DryRunResult",
    "DIRECTID_CONTRACT_REF",
    "DIRECTID_FAMILY",
    "DIRECTID_GATE_CODES",
    "DIRECTID_REQUIRED_ID_SUBTYPES",
    "DIRECTID_TINY_HEAD_CONTRACT",
    "DirectIDContractError",
    "DirectIDHeadContract",
    "DirectIDPresetValidation",
    "GatedCorpusAccessError",
    "HARD_NEGATIVE_CATEGORIES",
    "HardNegativeExample",
    "HardNegativeGenerator",
    "HardNegativeSampler",
    "JsonlPassageSource",
    "MimicIIIDuaSource",
    "Passage",
    "PassageSource",
    "PUBLIC_DAPT_SOURCES",
    "RecordPassageSource",
    "RecipeConfigError",
    "TrainingRecipeConfig",
    "WeakLabelDecision",
    "WeakLabelSpan",
    "arxiv_qbio_source",
    "assemble_dapt_corpus",
    "assert_manifest_has_no_raw_text",
    "config_hash",
    "corpus_manifest_hash",
    "count_hard_negatives",
    "dry_run_recipe",
    "gate_requirements_by_code",
    "load_corpus_manifest",
    "load_preset",
    "make_adjudication_item",
    "manifest_row_for_passage",
    "normalize_passage_text",
    "normalize_weak_span",
    "pmc_abstract_source",
    "pubmed_abstract_source",
    "requires_hard_negative_sampler",
    "run_recipe",
    "runtime_dependencies",
    "sample_hard_negatives",
    "sampler_for_recipe",
    "token_count",
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
        "DAPT_CORPUS_MANIFEST_PATH",
        "DAPT_CORPUS_SCHEMA_VERSION",
        "CorpusManifestError",
        "DaptCorpusAssemblyResult",
        "GatedCorpusAccessError",
        "JsonlPassageSource",
        "MimicIIIDuaSource",
        "Passage",
        "PassageSource",
        "PUBLIC_DAPT_SOURCES",
        "RecordPassageSource",
        "arxiv_qbio_source",
        "assemble_dapt_corpus",
        "assert_manifest_has_no_raw_text",
        "corpus_manifest_hash",
        "load_corpus_manifest",
        "manifest_row_for_passage",
        "normalize_passage_text",
        "pmc_abstract_source",
        "pubmed_abstract_source",
        "token_count",
    }:
        corpus = import_module(".corpus", __name__)
        return getattr(corpus, name)
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
        "WeakLabelDecision",
        "WeakLabelSpan",
        "normalize_weak_span",
        "weak_label_document",
    }:
        weak_labeling = import_module(".weak_labeling", __name__)
        return getattr(weak_labeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
