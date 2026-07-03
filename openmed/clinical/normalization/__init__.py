"""Clinical concept normalization with pluggable terminology backends."""

from __future__ import annotations

from .backend import (
    SYNTHETIC_CODE_SYSTEMS,
    SYNTHETIC_CONCEPTS,
    BackendIdentity,
    CodeSystemMetadata,
    SyntheticTerminologyBackend,
    TerminologyBackend,
    TerminologyConcept,
    normalize_surface,
    validate_backend_identity,
)
from .cache import (
    ConceptNormalizationCache,
    NormalizationCacheStats,
    make_normalization_cache_key,
)
from .ranker import (
    SYNTHETIC_GOLD_SET,
    CandidateProvenance,
    ConceptNormalizer,
    NormalizationEvaluationResult,
    NormalizationGoldCase,
    RankedConcept,
    evaluate_normalization_gold,
    generate_query_variants,
)

__all__ = [
    "BackendIdentity",
    "CandidateProvenance",
    "CodeSystemMetadata",
    "ConceptNormalizationCache",
    "ConceptNormalizer",
    "NormalizationCacheStats",
    "NormalizationEvaluationResult",
    "NormalizationGoldCase",
    "RankedConcept",
    "SYNTHETIC_CODE_SYSTEMS",
    "SYNTHETIC_CONCEPTS",
    "SYNTHETIC_GOLD_SET",
    "SyntheticTerminologyBackend",
    "TerminologyBackend",
    "TerminologyConcept",
    "evaluate_normalization_gold",
    "generate_query_variants",
    "make_normalization_cache_key",
    "normalize_surface",
    "validate_backend_identity",
]
