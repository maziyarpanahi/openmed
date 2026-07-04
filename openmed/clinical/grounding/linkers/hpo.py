"""HPO phenotype linker for clinical-finding spans.

Rounds out the free-vocabulary linker set (RxNorm/ICD-10-CM/LOINC/HPO) via the
shared :mod:`.base` matching, normalizing codes to the ``HP:nnnnnnn`` format.
"""

from __future__ import annotations

import re

from openmed.core.labels import CONDITION

from ..registry import register_linker
from ..types import Candidate
from .base import VocabLinker


class HpoLinker(VocabLinker):
    """Map phenotype/finding text to ranked HPO ``Candidate`` codes."""

    system = "HPO"
    key = "hpo"
    required_label = CONDITION

    def _format_code(self, code: str) -> str:
        code = str(code).strip()
        digits = re.sub(r"\D", "", code)
        # HPO codes are seven zero-padded digits behind the ``HP:`` prefix.
        return f"HP:{digits.zfill(7)}" if digits else code


register_linker("hpo", HpoLinker)
