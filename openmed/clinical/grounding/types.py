"""Shared types for clinical concept grounding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openmed.clinical.context import ClinicalAssertion, ClinicalContextResult


GROUNDING_CONFIDENCE_BANDS = frozenset(("accept", "uncertain"))


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
        calibrated_score: Optional post-calibration probability.
        calibrated_confidence: Optional post-calibration linking confidence.
        confidence_band: Optional ``"accept"`` or ``"uncertain"`` band.
        abstained: Whether calibrated grounding withheld the selected codes.
        provenance: Optional grounding-calibration provenance.
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
    calibrated_score: float | None = None
    abstained: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    canonical_label: str | None = None
    assertion: ClinicalAssertion | ClinicalContextResult | None = None
    source_language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    calibrated_confidence: float | None = None
    confidence_band: str | None = None

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
        if not isinstance(self.provenance, Mapping):
            raise TypeError("grounded span provenance must be a mapping")
        calibrated_score = _optional_probability(
            self.calibrated_score,
            "calibrated_score",
        )
        calibrated_confidence = _optional_probability(
            self.calibrated_confidence,
            "calibrated_confidence",
        )
        if (
            calibrated_score is not None
            and calibrated_confidence is not None
            and calibrated_score != calibrated_confidence
        ):
            raise ValueError(
                "calibrated_score and calibrated_confidence must agree when both "
                "are provided"
            )
        if calibrated_score is None:
            calibrated_score = calibrated_confidence
        if calibrated_confidence is None:
            calibrated_confidence = calibrated_score
        confidence_band = self.confidence_band
        if confidence_band is not None:
            confidence_band = str(confidence_band).strip().lower()
            if confidence_band not in GROUNDING_CONFIDENCE_BANDS:
                raise ValueError("confidence_band must be 'accept' or 'uncertain'")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "calibrated_score", calibrated_score)
        object.__setattr__(self, "calibrated_confidence", calibrated_confidence)
        object.__setattr__(self, "confidence_band", confidence_band)

    @property
    def band(self) -> str | None:
        """Alias for the explicit calibrated linking-confidence band."""

        return self.confidence_band

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
            "calibrated_score": self.calibrated_score,
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_band": self.confidence_band,
            "abstained": self.abstained,
            "provenance": dict(self.provenance),
            "canonical_label": self.canonical_label,
            "assertion": self.assertion.to_dict() if self.assertion else None,
            "source_language": self.source_language,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }


def _optional_probability(value: Any, name: str) -> float | None:
    """Validate an optional probability without importing calibration helpers."""

    if value is None:
        return None
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability
