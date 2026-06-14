"""Clinical document understanding package for section 4.2.

Intended contents include sections.py, context.py, grounding.py, relations.py,
sdoh.py, and FHIR/OMOP exporters.
"""

from .context import Certainty, resolve_uncertainty

__all__ = [
    "Certainty",
    "resolve_uncertainty",
]
