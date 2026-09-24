"""Typed, serializable results for the public grounding facade.

The existing :class:`~openmed.clinical.grounding.types.GroundedSpan` contract is
kept as the sequence view of :class:`GroundingResult` for interoperability with
the FHIR and OMOP exporters.  ``GroundingResult.concepts`` provides the
facade-oriented one-system-per-concept view requested by new callers.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .systems import canonical_system, system_uri
from .types import Candidate, GroundedSpan
from .vocab import GroundingConfigError

__all__ = [
    "ConceptSpan",
    "GroundedConcept",
    "GroundingCandidate",
    "GroundingConfigError",
    "GroundingResult",
    "GroundingSpan",
]


@dataclass(frozen=True)
class ConceptSpan:
    """Inclusive/exclusive character offsets for a grounded surface."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("concept span start must be a non-negative integer")
        if type(self.end) is not int or self.end < self.start:
            raise ValueError("concept span end must be at or after start")

    def to_dict(self) -> dict[str, int]:
        """Return JSON-ready offset fields."""

        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConceptSpan":
        """Build offsets from a serialized mapping."""

        return cls(start=int(value["start"]), end=int(value["end"]))


# A short alias is convenient for callers that want to annotate a span without
# importing the longer facade-oriented name.
GroundingSpan = ConceptSpan


@dataclass(frozen=True)
class GroundingCandidate:
    """One ranked candidate attached to a :class:`GroundedConcept`."""

    system: str
    code: str
    display: str
    confidence: float
    source: str = ""
    match_kind: str | None = None
    vocabulary_snapshot_version: str | None = None

    def __post_init__(self) -> None:
        normalized_system = canonical_system(self.system)
        if not normalized_system:
            raise ValueError("candidate system must not be empty")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("candidate code must be a non-empty string")
        if not isinstance(self.display, str) or not self.display.strip():
            raise ValueError("candidate display must be a non-empty string")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "system", normalized_system)
        object.__setattr__(self, "confidence", confidence)

    @property
    def score(self) -> float:
        """Backward-compatible score alias."""

        return self.confidence

    @property
    def system_uri(self) -> str | None:
        """Return the canonical FHIR system URI."""

        return system_uri(self.system)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready candidate record."""

        return {
            "system": self.system,
            "system_uri": self.system_uri,
            "code": self.code,
            "display": self.display,
            "confidence": self.confidence,
            "score": self.confidence,
            "source": self.source,
            "match_kind": self.match_kind,
            "vocabulary_snapshot_version": self.vocabulary_snapshot_version,
        }

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> "GroundingCandidate":
        """Convert the established candidate type to the facade type."""

        return cls(
            system=candidate.system,
            code=candidate.code,
            display=candidate.display,
            confidence=candidate.score,
            source=candidate.source,
            match_kind=candidate.match_kind,
            vocabulary_snapshot_version=candidate.vocab_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundingCandidate":
        """Build a candidate from a JSON-compatible mapping."""

        return cls(
            system=str(value.get("system") or value.get("system_uri") or ""),
            code=str(value.get("code") or ""),
            display=str(value.get("display") or ""),
            confidence=float(value.get("confidence", value.get("score", 0.0))),
            source=str(value.get("source") or ""),
            match_kind=(
                str(value["match_kind"])
                if value.get("match_kind") is not None
                else None
            ),
            vocabulary_snapshot_version=(
                str(value["vocabulary_snapshot_version"])
                if value.get("vocabulary_snapshot_version") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class GroundedConcept:
    """A typed concept suggestion for one source span and one terminology system.

    Grounding is assistive and requires qualified human review.  A missing code
    represents a deterministic abstention, not a diagnosis or coding decision.
    """

    span: ConceptSpan
    surface_text: str
    system: str
    code: str | None
    display: str | None
    confidence: float
    candidates: tuple[GroundingCandidate, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    section_context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span, ConceptSpan):
            raise TypeError("grounded concept span must be a ConceptSpan")
        if not isinstance(self.surface_text, str):
            raise TypeError("grounded concept surface_text must be a string")
        normalized_system = canonical_system(self.system) if self.system else ""
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("grounded concept confidence must be between 0.0 and 1.0")
        candidates = tuple(self.candidates)
        if any(
            not isinstance(candidate, GroundingCandidate) for candidate in candidates
        ):
            raise TypeError(
                "grounded concept candidates must be GroundingCandidate objects"
            )
        if self.code is not None and not isinstance(self.code, str):
            raise TypeError("grounded concept code must be text or None")
        if self.display is not None and not isinstance(self.display, str):
            raise TypeError("grounded concept display must be text or None")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("grounded concept provenance must be a mapping")
        object.__setattr__(self, "system", normalized_system)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def start(self) -> int:
        """Return the inclusive source offset."""

        return self.span.start

    @property
    def end(self) -> int:
        """Return the exclusive source offset."""

        return self.span.end

    @property
    def surface(self) -> str:
        """Return the source surface alias."""

        return self.surface_text

    @property
    def text(self) -> str:
        """Return the source surface using the legacy field name."""

        return self.surface_text

    @property
    def score(self) -> float:
        """Return the confidence using the legacy score name."""

        return self.confidence

    @property
    def top_k(self) -> tuple[GroundingCandidate, ...]:
        """Return the ranked candidates retained for this concept."""

        return self.candidates

    @property
    def system_uri(self) -> str | None:
        """Return the canonical FHIR system URI."""

        return system_uri(self.system) if self.system else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready concept record with offsets and provenance."""

        return {
            "span": self.span.to_dict(),
            "start": self.start,
            "end": self.end,
            "surface": self.surface_text,
            "surface_text": self.surface_text,
            "text": self.surface_text,
            "system": self.system,
            "system_uri": self.system_uri,
            "code": self.code,
            "display": self.display,
            "confidence": self.confidence,
            "score": self.confidence,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "top_k": [candidate.to_dict() for candidate in self.candidates],
            "provenance": dict(self.provenance),
            "section_context": self.section_context,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundedConcept":
        """Build a typed concept from a JSON-compatible mapping."""

        raw_span = value.get("span")
        if isinstance(raw_span, Mapping):
            span = ConceptSpan.from_dict(raw_span)
        else:
            span = ConceptSpan(start=int(value["start"]), end=int(value["end"]))
        raw_candidates = value.get("candidates", value.get("top_k", ()))
        candidates = tuple(
            GroundingCandidate.from_dict(candidate)
            for candidate in raw_candidates
            if isinstance(candidate, Mapping)
        )
        code = value.get("code")
        display = value.get("display")
        system = str(value.get("system") or "")
        if code is not None and not candidates and system:
            candidates = (
                GroundingCandidate(
                    system=system,
                    code=str(code),
                    display=str(display or ""),
                    confidence=float(value.get("confidence", 0.0)),
                ),
            )
        return cls(
            span=span,
            surface_text=str(
                value.get("surface_text", value.get("surface", value.get("text", "")))
            ),
            system=system,
            code=str(code) if code is not None else None,
            display=str(display) if display is not None else None,
            confidence=float(value.get("confidence", value.get("score", 0.0))),
            candidates=candidates,
            provenance=(
                value.get("provenance", {})
                if isinstance(value.get("provenance", {}), Mapping)
                else {}
            ),
            section_context=(
                str(value["section_context"])
                if value.get("section_context") is not None
                else None
            ),
        )

    @classmethod
    def from_grounded_span(
        cls,
        span: GroundedSpan,
        *,
        systems: Sequence[str],
        top_k: int,
    ) -> tuple["GroundedConcept", ...]:
        """Project one legacy span into one concept per matched system."""

        ranked = (*span.candidates, *span.alternatives)
        by_system: dict[str, list[Candidate]] = {}
        for candidate in ranked:
            by_system.setdefault(canonical_system(candidate.system), []).append(
                candidate
            )

        requested = tuple(dict.fromkeys(canonical_system(system) for system in systems))
        ordered_systems = requested or tuple(by_system)
        concepts: list[GroundedConcept] = []
        for system in ordered_systems:
            candidates = tuple(by_system.get(system, ()))[:top_k]
            selected = candidates[0] if candidates else None
            snapshot_details = span.provenance.get("snapshot_provenance", {})
            details = (
                snapshot_details.get(system, {})
                if isinstance(snapshot_details, Mapping)
                else {}
            )
            if not isinstance(details, Mapping):
                details = {}
            snapshot_version = (
                details.get("version")
                or details.get("release_version")
                or (
                    selected.vocab_version
                    if selected is not None and selected.vocab_version
                    else None
                )
            )
            linker = (
                selected.source
                if selected is not None and selected.source
                else ("none" if selected is None else "lexical")
            )
            provenance = dict(span.provenance)
            provenance.update(
                {
                    "vocabulary_snapshot_version": snapshot_version,
                    "snapshot_version": snapshot_version,
                    "linker": linker,
                    "linker_name": linker,
                    "section_context": span.section,
                    "source_language": span.source_language,
                    "abstained": selected is None,
                }
            )
            concepts.append(
                cls(
                    span=ConceptSpan(span.start, span.end),
                    surface_text=span.text,
                    system=system,
                    code=selected.code if selected is not None else None,
                    display=selected.display if selected is not None else None,
                    confidence=selected.score if selected is not None else 0.0,
                    candidates=tuple(
                        GroundingCandidate.from_candidate(candidate)
                        for candidate in candidates
                    ),
                    provenance=provenance,
                    section_context=span.section,
                )
            )
        return tuple(concepts)


@dataclass(frozen=True)
class GroundingResult(Sequence[GroundedSpan]):
    """The typed result returned by :func:`openmed.ground`.

    Iteration and integer indexing intentionally expose the established
    ``GroundedSpan`` values so existing exporters remain source-compatible.
    New callers should use :attr:`concepts` for one-system-per-concept records.
    """

    spans: tuple[GroundedSpan, ...] = ()
    concepts: tuple[GroundedConcept, ...] = ()
    systems: tuple[str, ...] = ()
    language: str = "en"
    top_k: int = 1
    offline: bool = True

    def __post_init__(self) -> None:
        spans = tuple(self.spans)
        concepts = tuple(self.concepts)
        if any(not isinstance(span, GroundedSpan) for span in spans):
            raise TypeError("grounding result spans must be GroundedSpan objects")
        if any(not isinstance(concept, GroundedConcept) for concept in concepts):
            raise TypeError("grounding result concepts must be GroundedConcept objects")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("grounding result top_k must be a positive integer")
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "concepts", concepts)
        object.__setattr__(
            self,
            "systems",
            tuple(dict.fromkeys(canonical_system(system) for system in self.systems)),
        )

    @classmethod
    def from_spans(
        cls,
        spans: Sequence[GroundedSpan],
        *,
        systems: Sequence[str],
        language: str,
        top_k: int,
        offline: bool,
    ) -> "GroundingResult":
        """Build the facade result from the established span pipeline."""

        legacy_spans = tuple(spans)
        concepts = tuple(
            concept
            for span in legacy_spans
            for concept in GroundedConcept.from_grounded_span(
                span,
                systems=systems,
                top_k=top_k,
            )
        )
        return cls(
            spans=legacy_spans,
            concepts=concepts,
            systems=tuple(systems),
            language=language,
            top_k=top_k,
            offline=offline,
        )

    @property
    def grounded_concepts(self) -> tuple[GroundedConcept, ...]:
        """Alias for :attr:`concepts` used by service-oriented callers."""

        return self.concepts

    @property
    def entries(self) -> tuple[GroundedConcept, ...]:
        """Return typed concept entries for callers using an entry-oriented API."""

        return self.concepts

    @property
    def lang(self) -> str:
        """Return the normalized source language using the facade argument name."""

        return self.language

    @property
    def results(self) -> tuple[GroundedSpan, ...]:
        """Return the compatibility span sequence."""

        return self.spans

    def __getitem__(
        self, index: int | slice
    ) -> GroundedSpan | tuple[GroundedSpan, ...]:
        return self.spans[index]

    def __iter__(self) -> Iterator[GroundedSpan]:
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result with spans, codes, and provenance."""

        return {
            "schema_version": "openmed.grounding.v1",
            "systems": list(self.systems),
            "language": self.language,
            "lang": self.language,
            "top_k": self.top_k,
            "offline": bool(self.offline),
            "spans": [span.to_dict() for span in self.spans],
            "concepts": [concept.to_dict() for concept in self.concepts],
            "grounded_concepts": [concept.to_dict() for concept in self.concepts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundingResult":
        """Reconstruct a result from :meth:`to_dict` output."""

        raw_concepts = value.get("concepts", value.get("grounded_concepts", ()))
        concepts = tuple(
            GroundedConcept.from_dict(item)
            for item in raw_concepts
            if isinstance(item, Mapping)
        )
        raw_spans = value.get("spans", ())
        spans = tuple(
            _grounded_span_from_dict(item)
            for item in raw_spans
            if isinstance(item, Mapping)
        )
        if not spans and concepts:
            spans = _spans_from_concepts(concepts)
        language = str(value.get("language", value.get("lang", "en")))
        return cls(
            spans=spans,
            concepts=concepts,
            systems=tuple(value.get("systems", ())),
            language=language,
            top_k=int(value.get("top_k", 1)),
            offline=bool(value.get("offline", True)),
        )


def _grounded_span_from_dict(value: Mapping[str, Any]) -> GroundedSpan:
    """Build a compatibility span from serialized result data."""

    raw_candidates = value.get("candidates", ())
    candidates = tuple(
        _candidate_from_dict(candidate)
        for candidate in raw_candidates
        if isinstance(candidate, Mapping)
    )
    raw_alternatives = value.get("alternatives", value.get("ranked_alternatives", ()))
    alternatives = tuple(
        _candidate_from_dict(candidate)
        for candidate in raw_alternatives
        if isinstance(candidate, Mapping)
    )
    return GroundedSpan(
        text=str(value.get("text", value.get("surface", ""))),
        start=int(value.get("start", 0)),
        end=int(value.get("end", 0)),
        candidates=candidates,
        alternatives=alternatives,
        calibrated_score=value.get("calibrated_score"),
        abstained=bool(value.get("abstained", False)),
        provenance=(
            value.get("provenance", {})
            if isinstance(value.get("provenance", {}), Mapping)
            else {}
        ),
        canonical_label=value.get("canonical_label"),
        source_language=str(value.get("source_language", "en")),
        metadata=(
            value.get("metadata", {})
            if isinstance(value.get("metadata", {}), Mapping)
            else {}
        ),
        section=value.get("section"),
    )


def _candidate_from_dict(value: Mapping[str, Any]) -> Candidate:
    """Build an established candidate from a serialized candidate mapping."""

    return Candidate(
        system=str(value.get("system") or value.get("system_uri") or ""),
        code=str(value.get("code") or ""),
        display=str(value.get("display") or ""),
        score=float(value.get("score", value.get("confidence", 0.0))),
        source_language=str(value.get("source_language", "en")),
        source=str(value.get("source", "")),
        matched_alias=value.get("matched_alias"),
        match_kind=value.get("match_kind"),
        vocab_version=value.get(
            "vocab_version", value.get("vocabulary_snapshot_version")
        ),
    )


def _spans_from_concepts(
    concepts: Sequence[GroundedConcept],
) -> tuple[GroundedSpan, ...]:
    """Group concept records back into compatibility spans."""

    grouped: dict[tuple[int, int, str], list[GroundedConcept]] = {}
    for concept in concepts:
        grouped.setdefault(
            (concept.start, concept.end, concept.surface_text), []
        ).append(concept)
    spans: list[GroundedSpan] = []
    for (start, end, surface), entries in grouped.items():
        candidates = tuple(
            Candidate(
                system=entry.system,
                code=entry.code,
                display=entry.display or "",
                score=entry.confidence,
                source=str(entry.provenance.get("linker", "")),
                vocab_version=entry.provenance.get("vocabulary_snapshot_version"),
            )
            for entry in entries
            if entry.code is not None and entry.display is not None
        )
        spans.append(
            GroundedSpan(
                text=surface,
                start=start,
                end=end,
                candidates=candidates,
                provenance=dict(entries[0].provenance),
                section=entries[0].section_context,
            )
        )
    return tuple(spans)
