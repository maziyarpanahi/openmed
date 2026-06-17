"""Clinical document understanding package for section 4.2.

Intended contents include sections.py, context.py, grounding.py, relations.py,
sdoh.py, and FHIR/OMOP exporters.
"""

from .context import (
    HISTORICAL,
    HYPOTHETICAL,
    RECENT,
    TEMPORALITY_VALUES,
    resolve_temporality,
)

__all__ = [
    "RECENT",
    "HISTORICAL",
    "HYPOTHETICAL",
    "TEMPORALITY_VALUES",
    "resolve_temporality",
]
