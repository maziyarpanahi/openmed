"""Clinical concept grounding: map clinical spans to coded vocabulary concepts.

Provides the shared, reusable surface for the free-vocabulary linkers
(RxNorm/ICD-10-CM/LOINC/HPO): the :class:`Candidate` type, a lightweight
:func:`load_vocab` index loader, and a linker registry. Importing this package
registers the bundled linkers by system key.
"""

from __future__ import annotations

# Importing the linkers package registers them in the registry (e.g. "rxnorm").
from . import linkers as _linkers  # noqa: F401  (import for registration side effect)
from .registry import available_linkers, get_linker, register_linker
from .types import Candidate
from .vocab import VocabEntry, VocabIndex, load_vocab, normalize_term

__all__ = [
    "Candidate",
    "VocabEntry",
    "VocabIndex",
    "load_vocab",
    "normalize_term",
    "register_linker",
    "get_linker",
    "available_linkers",
]
