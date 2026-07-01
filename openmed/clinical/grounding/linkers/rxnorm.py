"""RxNorm approximate-match linker for medication spans.

Reference implementation for the free-vocabulary linkers (ICD-10-CM, LOINC,
HPO follow the same pattern). Candidate generation is two-stage: exact alias
match first, then approximate match (rapidfuzz) over synonyms. Exact matching
and importing this module work without the optional ``grounding`` extra; the
approximate stage is skipped when rapidfuzz is not installed.
"""

from __future__ import annotations

from openmed.core.labels import MEDICATION, normalize_label

from ..registry import register_linker
from ..types import Candidate
from ..vocab import VocabEntry, VocabIndex, normalize_term

SYSTEM = "RXNORM"

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.6


class RxNormLinker:
    """Map medication text to ranked RxNorm ``Candidate`` codes."""

    system = "rxnorm"

    def __init__(self, vocab: VocabIndex) -> None:
        self._vocab = vocab

    def link(
        self,
        text: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        min_score: float = _DEFAULT_MIN_SCORE,
        canonical_label: str | None = None,
        fuzzy: bool = True,
    ) -> list[Candidate]:
        """Return ranked RxNorm candidates for ``text``.

        Only operates on medication spans: if ``canonical_label`` is given and
        does not normalize to ``MEDICATION``, an empty list is returned.
        """
        if (
            canonical_label is not None
            and normalize_label(canonical_label) != MEDICATION
        ):
            return []

        query = normalize_term(text)
        if not query:
            return []

        best: dict[str, Candidate] = {}
        for entry in self._vocab.exact(query):
            best[entry.code] = Candidate(SYSTEM, entry.code, entry.display, 1.0)

        if fuzzy:
            for entry, score in self._approximate(query):
                if score < min_score:
                    continue
                existing = best.get(entry.code)
                if existing is None or score > existing.score:
                    best[entry.code] = Candidate(
                        SYSTEM, entry.code, entry.display, score
                    )

        # Deterministic ranking: score desc, then code asc for ties.
        ranked = sorted(
            best.values(), key=lambda candidate: (-candidate.score, candidate.code)
        )
        return ranked[:top_k]

    def _approximate(self, query: str) -> list[tuple[VocabEntry, float]]:
        try:
            from rapidfuzz import fuzz
        except ImportError:  # pragma: no cover - exercised without the extra
            return []

        best_score: dict[str, float] = {}
        best_entry: dict[str, VocabEntry] = {}
        for alias, entry in self._vocab.iter_alias_entries():
            score = fuzz.ratio(query, alias) / 100.0
            if score > best_score.get(entry.code, 0.0):
                best_score[entry.code] = score
                best_entry[entry.code] = entry
        return [(best_entry[code], score) for code, score in best_score.items()]


register_linker("rxnorm", RxNormLinker)
