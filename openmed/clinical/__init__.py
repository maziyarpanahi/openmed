"""Clinical document understanding package for section 4.2.

Intended contents include sections.py, context.py, grounding.py, relations.py,
sdoh.py, and FHIR/OMOP exporters.
"""

from .context import (
    CERTAIN,
    CERTAINTY_VALUES,
    HISTORICAL,
    HYPOTHETICAL,
    RECENT,
    TEMPORALITY_VALUES,
    UNCERTAIN,
    Certainty,
    resolve_temporality,
    resolve_uncertainty,
)

__all__ = [
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
]
