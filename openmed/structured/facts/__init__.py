"""Evidence-bound normalization of heterogeneous clinical component outputs."""

from .normalization import (
    CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION,
    FACT_FIELD_STATES,
    FACT_FRAGMENT_KINDS,
    FACT_PROFILE_SPECS,
    NORMALIZER_VERSION,
    ClinicalFactNormalizer,
    ComponentOutputEnvelope,
    FactFragment,
    FactNormalizationError,
    FactNormalizationRequest,
    FactProfileSpec,
    MappingFactAdapter,
    NormalizedClinicalFact,
    RelationParticipant,
    load_clinical_fact_normalization_schema,
)

__all__ = [
    "CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION",
    "FACT_FIELD_STATES",
    "FACT_FRAGMENT_KINDS",
    "FACT_PROFILE_SPECS",
    "NORMALIZER_VERSION",
    "ClinicalFactNormalizer",
    "ComponentOutputEnvelope",
    "FactFragment",
    "FactNormalizationError",
    "FactNormalizationRequest",
    "FactProfileSpec",
    "MappingFactAdapter",
    "NormalizedClinicalFact",
    "RelationParticipant",
    "load_clinical_fact_normalization_schema",
]
