"""Shared types for clinical concept grounding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A ranked grounding candidate: a coded concept for a clinical span.

    ``system`` is the vocabulary (e.g. ``"RXNORM"``), ``code`` its identifier
    (e.g. an RxCUI), ``display`` a human-readable name, and ``score`` a
    ``0.0``-``1.0`` match confidence (``1.0`` for an exact alias match).

    The optional provenance fields let downstream rankers and exporters audit
    where a candidate came from without re-running retrieval.  ``source`` names
    the generator that emitted it (e.g. ``"sparse"``), ``matched_alias`` is the
    normalized vocabulary alias that matched (never the caller's raw span text),
    ``match_kind`` records how it matched (``"exact"`` or ``"fuzzy"``), and
    ``vocab_version`` is a content hash of the vocabulary snapshot the candidate
    was drawn from.
    """

    system: str
    code: str
    display: str
    score: float
    source_language: str = "en"
    source: str = ""
    matched_alias: str | None = None
    match_kind: str | None = None
    vocab_version: str | None = None

    @property
    def concept_id(self) -> str:
        """Alias for :attr:`code`, matching the free-vocabulary schema."""

        return self.code

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready candidate record including source language."""

        return {
            "system": self.system,
            "code": self.code,
            "display": self.display,
            "score": self.score,
            "source_language": self.source_language,
            "source": self.source,
            "matched_alias": self.matched_alias,
            "match_kind": self.match_kind,
            "vocab_version": self.vocab_version,
        }
