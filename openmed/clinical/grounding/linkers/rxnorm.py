"""RxNorm approximate-match linker for medication spans.

Reference implementation for the free-vocabulary linkers; the shared
candidate-generation lives in :mod:`.base`.
"""

from __future__ import annotations

from openmed.core.labels import MEDICATION

from ..registry import register_linker
from .base import VocabLinker


class RxNormLinker(VocabLinker):
    """Map medication text to ranked RxNorm ``Candidate`` codes."""

    system = "RXNORM"
    key = "rxnorm"
    required_label = MEDICATION


register_linker("rxnorm", RxNormLinker)
