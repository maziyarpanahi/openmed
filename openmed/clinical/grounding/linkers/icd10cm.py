"""ICD-10-CM diagnosis linker for condition spans.

Follows the RxNorm linker pattern via the shared :mod:`.base` matching, adding
ICD-10-CM specifics: dotted code formatting (``E119`` -> ``E11.9``) and a
preference for billable leaf codes over their 3-character category.
"""

from __future__ import annotations

from openmed.core.labels import CONDITION

from ..registry import register_linker
from ..types import Candidate
from .base import VocabLinker


class Icd10cmLinker(VocabLinker):
    """Map diagnosis/condition text to ranked ICD-10-CM ``Candidate`` codes."""

    system = "ICD10CM"
    key = "icd10cm"
    required_label = CONDITION

    def _format_code(self, code: str) -> str:
        code = str(code).strip().upper()
        if "." in code or len(code) <= 3:
            return code
        # ICD-10-CM: 3-character category, then a dot, then the subcategory.
        return f"{code[:3]}.{code[3:]}"

    def _rank_key(self, candidate: Candidate) -> tuple[object, ...]:
        # Score desc, then prefer billable leaf codes (longer/more specific),
        # then code asc for a deterministic tie-break.
        specificity = len(candidate.code.replace(".", ""))
        return (-candidate.score, -specificity, candidate.code)


register_linker("icd10cm", Icd10cmLinker)
