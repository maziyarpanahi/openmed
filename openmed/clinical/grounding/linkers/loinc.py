"""LOINC approximate-match linker for laboratory-test spans.

LOINC is free with registration. Registration is a user step, not bundled
with OpenMed; callers provide a local LOINC snapshot through ``VocabLoader``.
The linker uses the shared exact and approximate alias matching implemented by
:class:`~openmed.clinical.grounding.linkers.base.VocabLinker`.
"""

from __future__ import annotations

from openmed.core.labels import LAB_TEST

from ..registry import register_linker
from .base import VocabLinker


class LoincLinker(VocabLinker):
    """Map laboratory-test text to ranked LOINC ``Candidate`` codes."""

    system = "LOINC"
    key = "loinc"
    required_label = LAB_TEST


register_linker("loinc", LoincLinker)
