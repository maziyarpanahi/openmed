"""Shared candidate-generation base for free-vocabulary linkers.

Both the RxNorm and ICD-10-CM linkers share this two-stage matching (exact
alias, then rapidfuzz approximate) so matching logic is implemented once.
Subclasses set ``system``/``required_label`` and may override code formatting
and ranking (e.g. ICD-10-CM dotted codes and leaf-over-category preference).
Importing this module works without the optional ``grounding`` extra; the
approximate stage is skipped when rapidfuzz is absent.
"""

from __future__ import annotations

from typing import Any

from openmed.core.labels import normalize_label

from ..types import Candidate
from ..vocab import VocabEntry, VocabIndex, normalize_term

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.6


class VocabLinker:
    """Map span text to ranked ``Candidate`` codes over a vocabulary index."""

    #: Vocabulary system emitted on candidates, e.g. ``"RXNORM"``.
    system: str = ""
    #: Registry key the linker registers under, e.g. ``"rxnorm"``.
    key: str = ""
    #: Canonical label the span must carry, e.g. ``MEDICATION`` (None = any).
    required_label: str | None = None

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
        """Return ranked candidates for ``text``.

        If ``canonical_label`` is given and does not normalize to
        :attr:`required_label`, an empty list is returned.
        """
        if (
            canonical_label is not None
            and self.required_label is not None
            and normalize_label(canonical_label) != self.required_label
        ):
            return []

        query = normalize_term(text)
        if not query:
            return []

        best: dict[str, Candidate] = {}
        for entry in self._vocab.exact(query):
            self._offer(best, entry, 1.0)
        if fuzzy:
            for entry, score in approximate_match(self._vocab, query):
                if score >= min_score:
                    self._offer(best, entry, score)

        return sorted(best.values(), key=self._rank_key)[:top_k]

    def _offer(
        self, best: dict[str, Candidate], entry: VocabEntry, score: float
    ) -> None:
        candidate = self._candidate(entry, score)
        existing = best.get(candidate.code)
        if existing is None or score > existing.score:
            best[candidate.code] = candidate

    def _candidate(self, entry: VocabEntry, score: float) -> Candidate:
        return Candidate(
            self.system, self._format_code(entry.code), entry.display, float(score)
        )

    def _format_code(self, code: str) -> str:
        """Hook for vocabulary-specific code formatting (default: unchanged)."""
        return code

    def _rank_key(self, candidate: Candidate) -> tuple[Any, ...]:
        """Deterministic ranking: score desc, then code asc for ties."""
        return (-candidate.score, candidate.code)


def approximate_match(vocab: VocabIndex, query: str) -> list[tuple[VocabEntry, float]]:
    """Best rapidfuzz similarity per entry; empty when rapidfuzz is absent."""
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - exercised without the extra
        return []

    best_score: dict[str, float] = {}
    best_entry: dict[str, VocabEntry] = {}
    for alias, entry in vocab.iter_alias_entries():
        score = fuzz.ratio(query, alias) / 100.0
        if score > best_score.get(entry.code, 0.0):
            best_score[entry.code] = score
            best_entry[entry.code] = entry
    return [(best_entry[code], score) for code, score in best_score.items()]
