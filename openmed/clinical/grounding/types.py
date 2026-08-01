"""Shared types for clinical concept grounding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openmed.clinical.context import ClinicalAssertion


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


@dataclass(frozen=True)
class GroundedSpan:
    """A source span and its selected standard-concept candidates.

    ``candidates`` is ordered by the grounding ranker. The convenience
    :attr:`codes`, :attr:`cui`, and :attr:`score` properties expose the compact
    public contract from the grounding epic without discarding the richer
    per-candidate provenance required by downstream exporters.

    Args:
        text: Source surface for the span.
        start: Inclusive character offset in the source document.
        end: Exclusive character offset in the source document.
        candidates: At most one selected candidate per requested system.
        canonical_label: Optional canonical clinical label such as
            ``"CONDITION"`` or ``"MEDICATION"``.
        assertion: Optional composed clinical assertion consumed by FHIR and
            OMOP exporters.
        source_language: Normalized language used for grounding.
        metadata: Optional caller-owned structured export context.

    Note:
        Grounding output is advisory and requires human verification. It is not
        an autonomous clinical coding, diagnosis, treatment, or billing
        decision.
    """

    text: str
    start: int
    end: int
    candidates: tuple[Candidate, ...] = ()
    canonical_label: str | None = None
    assertion: ClinicalAssertion | None = None
    source_language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("grounded span text must be a string")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("grounded span start must be a non-negative integer")
        if type(self.end) is not int or self.end < self.start:
            raise ValueError("grounded span end must be an integer at or after start")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("grounded span candidates must be Candidate objects")
        if self.canonical_label is not None and not self.canonical_label.strip():
            raise ValueError("canonical_label must be non-empty when provided")
        if not isinstance(self.source_language, str) or not self.source_language:
            raise ValueError("source_language must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("grounded span metadata must be a mapping")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def codes(self) -> dict[str, str]:
        """Return selected codes keyed by normalized vocabulary system."""

        codes: dict[str, str] = {}
        for candidate in self.candidates:
            codes.setdefault(candidate.system.casefold(), candidate.code)
        return codes

    @property
    def cui(self) -> str | None:
        """Return the selected UMLS CUI, if a gated UMLS matcher was used."""

        for candidate in self.candidates:
            if candidate.system.casefold() == "umls":
                return candidate.code
        return None

    @property
    def score(self) -> float:
        """Return the highest selected candidate score, or ``0.0`` on abstention."""

        if not self.candidates:
            return 0.0
        score = float(self.candidates[0].score)
        if not math.isfinite(score):
            raise ValueError("grounded span score must be finite")
        return score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready grounding record with offsets and provenance."""

        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "cui": self.cui,
            "codes": self.codes,
            "score": self.score,
            "canonical_label": self.canonical_label,
            "assertion": self.assertion.to_dict() if self.assertion else None,
            "source_language": self.source_language,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }
