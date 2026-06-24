"""Clinical document understanding package for section 4.2.

Intended contents include sections.py, context.py, grounding.py, relations.py,
sdoh.py, and FHIR/OMOP exporters.
"""

from .context import (
    AFFIRMED,
    CERTAIN,
    CERTAINTY_VALUES,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    NEGATION_VALUES,
    RECENT,
    TEMPORALITY_VALUES,
    UNCERTAIN,
    Certainty,
    ClinicalContextResult,
    Negation,
    resolve_negation,
    resolve_span_context,
    resolve_temporality,
    resolve_uncertainty,
)
from .lab_values import (
    LAB_FLAG_ADVISORY,
    AbnormalFlag,
    ReferenceRange,
    derive_abnormal_flag,
    parse_reference_range,
)

__all__ = [
    "AFFIRMED",
    "NEGATED",
    "NEGATION_VALUES",
    "Negation",
    "ClinicalContextResult",
    "resolve_negation",
    "resolve_span_context",
    "RECENT",
    "HISTORICAL",
    "HYPOTHETICAL",
    "TEMPORALITY_VALUES",
    "resolve_temporality",
    "Certainty",
    "CERTAIN",
    "UNCERTAIN",
    "CERTAINTY_VALUES",
    "resolve_uncertainty",
    "AbnormalFlag",
    "ReferenceRange",
    "LAB_FLAG_ADVISORY",
    "parse_reference_range",
    "derive_abnormal_flag",
]
