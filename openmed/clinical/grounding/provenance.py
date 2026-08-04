"""Durable, reproducible provenance for clinical concept grounding decisions.

When grounding emits a code, a regulated deployment must be able to reproduce
and defend that decision: which vocabulary edition produced it, which retrieval
or rerank method chose it, what the calibrated score was, and which runner-up
concepts were considered. :class:`GroundingProvenance` captures that decision
chain as a compact, serializable record that feeds straight into the signed
de-identification audit report alongside detector provenance.

Every record is PHI-safe by construction. The surrounding note text is never
stored: only the input span offsets and a content hash of the span surface are
kept, together with terminology metadata (system, code, display, version) drawn
from the vocabulary rather than the note. The alternative concepts deliberately
omit the note-derived matched alias for the same reason.

Codes captured here are machine-suggested retrieval results for human review,
never an autonomous clinical coding, treatment, or billing decision; see
:data:`GROUNDING_ASSIST_ONLY_ADVISORY`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openmed.core.audit import hash_text

from .types import Candidate

__all__ = [
    "GROUNDING_METHODS",
    "GROUNDING_ASSIST_ONLY_ADVISORY",
    "GroundingAlternative",
    "GroundingProvenance",
    "grounding_provenance",
    "provenance_version_pins",
    "scan_provenance_for_raw_text",
]

#: Retrieval/rerank methods a grounding stage may record.
GROUNDING_METHODS: frozenset[str] = frozenset(
    {"sparse", "dense", "rerank", "post-coordinated", "composite"}
)

#: Medical-device advisory stamped on every emitted grounding record.
GROUNDING_ASSIST_ONLY_ADVISORY = (
    "Machine-suggested terminology grounding for human verification; not an "
    "autonomous clinical coding, treatment, or billing decision."
)


@runtime_checkable
class _GroundedSpanLike(Protocol):
    """Structural view of a grounded span consumed by :func:`grounding_provenance`."""

    text: str
    start: int
    end: int
    candidates: Sequence[Candidate]


def _validate_offsets(start: int, end: int) -> None:
    if type(start) is not int or start < 0:
        raise ValueError("grounding span start must be a non-negative integer")
    if type(end) is not int or end < start:
        raise ValueError("grounding span end must be an integer at or after start")


def _validate_method(method: str) -> str:
    if method not in GROUNDING_METHODS:
        allowed = ", ".join(sorted(GROUNDING_METHODS))
        raise ValueError(
            f"unknown grounding method {method!r}; expected one of: {allowed}"
        )
    return method


def _safe_score(score: float) -> float:
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("grounding score must be a finite number")
    return value


@dataclass(frozen=True)
class GroundingAlternative:
    """A runner-up concept considered but not chosen for a grounded span.

    Carries only vocabulary-derived metadata (``system``, ``code``, ``display``)
    plus the ``score``, the generating ``source``, and the ``match_kind``. The
    note-derived matched alias is intentionally excluded so the record stays free
    of raw note text.
    """

    system: str
    code: str
    display: str
    score: float
    source: str = ""
    match_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, JSON-serializable mapping for this alternative."""
        return {
            "system": self.system,
            "code": self.code,
            "display": self.display,
            "score": _safe_score(self.score),
            "source": self.source,
            "match_kind": self.match_kind,
        }

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> "GroundingAlternative":
        """Build an alternative from a grounding :class:`Candidate`."""
        return cls(
            system=candidate.system,
            code=candidate.code,
            display=candidate.display,
            score=float(candidate.score),
            source=candidate.source or "",
            match_kind=candidate.match_kind or "",
        )


@dataclass(frozen=True)
class GroundingProvenance:
    """Reproducible provenance for one grounding decision on a clinical span.

    The record binds the input span (offsets plus a content hash of its surface,
    never the surface itself) to the chosen concept, the alternatives considered,
    the calibrated ``score``, the retrieval/rerank ``method``, the vocabulary
    ``vocab_version`` (terminology edition), and the ``encoder_index_key`` so a
    reviewer can re-derive exactly which terminology version and method produced
    each emitted code. ``abstained`` marks decisions where no code was emitted.
    """

    start: int
    end: int
    text_hash: str
    system: str
    code: str
    display: str
    score: float
    method: str
    vocab_version: str = ""
    encoder_index_key: str = ""
    abstained: bool = False
    alternatives: tuple[GroundingAlternative, ...] = ()

    def __post_init__(self) -> None:
        _validate_offsets(self.start, self.end)
        _validate_method(self.method)
        _safe_score(self.score)

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, PHI-safe, JSON-serializable provenance mapping."""
        return {
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "system": self.system,
            "code": self.code,
            "display": self.display,
            "score": _safe_score(self.score),
            "method": self.method,
            "vocab_version": self.vocab_version,
            "encoder_index_key": self.encoder_index_key,
            "abstained": bool(self.abstained),
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "advisory": GROUNDING_ASSIST_ONLY_ADVISORY,
        }

    @classmethod
    def from_candidates(
        cls,
        *,
        start: int,
        end: int,
        candidates: Sequence[Candidate],
        method: str,
        span_text: str | None = None,
        text_hash: str | None = None,
        calibrated_score: float | None = None,
        encoder_index_key: str = "",
        abstained: bool | None = None,
    ) -> "GroundingProvenance":
        """Build a record from ranked candidates for one grounded span.

        The highest-ranked candidate (``candidates[0]``) is the chosen concept and
        the remainder are recorded as alternatives. ``span_text`` is hashed and
        discarded so no raw note text is retained; pass ``text_hash`` directly to
        avoid handing the surface to this function at all. An empty candidate list
        yields an abstention record. An explicit abstention emits no chosen code and
        preserves every candidate as a review alternative. ``method`` must be one
        of :data:`GROUNDING_METHODS`.
        """
        _validate_method(method)
        if text_hash is None:
            if span_text is None:
                raise ValueError("either span_text or text_hash must be provided")
            text_hash = hash_text(span_text)

        ranked = list(candidates)
        should_abstain = not ranked or bool(abstained)
        if should_abstain:
            return cls(
                start=start,
                end=end,
                text_hash=text_hash,
                system="",
                code="",
                display="",
                score=0.0 if calibrated_score is None else float(calibrated_score),
                method=method,
                vocab_version="",
                encoder_index_key=encoder_index_key,
                abstained=True,
                alternatives=tuple(
                    GroundingAlternative.from_candidate(candidate)
                    for candidate in ranked
                ),
            )

        chosen = ranked[0]
        score = (
            float(chosen.score) if calibrated_score is None else float(calibrated_score)
        )
        return cls(
            start=start,
            end=end,
            text_hash=text_hash,
            system=chosen.system,
            code=chosen.code,
            display=chosen.display,
            score=score,
            method=method,
            vocab_version=chosen.vocab_version or "",
            encoder_index_key=encoder_index_key,
            abstained=False if abstained is None else bool(abstained),
            alternatives=tuple(
                GroundingAlternative.from_candidate(candidate)
                for candidate in ranked[1:]
            ),
        )


def grounding_provenance(
    grounded: Iterable[_GroundedSpanLike],
    *,
    method: str = "composite",
    encoder_index_key: str = "",
    calibrated_scores: Sequence[float] | None = None,
) -> list[GroundingProvenance]:
    """Return a complete provenance chain for a sequence of grounded spans.

    Each grounded span (any object exposing ``text``, ``start``, ``end``, and a
    ``candidates`` sequence — for example
    :class:`openmed.clinical.exporters.codeable_concept.GroundedSpan`) yields one
    :class:`GroundingProvenance` record. Spans with candidates produce a record
    for the emitted code; spans without candidates produce an abstention record,
    so the returned chain is complete. ``method`` must be one of
    :data:`GROUNDING_METHODS`; calibrated scores and abstention flags carried by
    the spans are preserved. When ``calibrated_scores`` is given it must align
    positionally with ``grounded`` and overrides each span's calibrated score.

    The span surface is hashed and never stored, so the returned chain is
    PHI-safe and, for identical input, byte-for-byte reproducible.
    """
    _validate_method(method)
    spans = list(grounded)
    if calibrated_scores is not None and len(calibrated_scores) != len(spans):
        raise ValueError("calibrated_scores must align with grounded spans")

    records: list[GroundingProvenance] = []
    for index, span in enumerate(spans):
        calibrated = (
            getattr(span, "calibrated_score", None)
            if calibrated_scores is None
            else calibrated_scores[index]
        )
        records.append(
            GroundingProvenance.from_candidates(
                start=span.start,
                end=span.end,
                span_text=span.text,
                candidates=tuple(span.candidates),
                method=method,
                calibrated_score=calibrated,
                encoder_index_key=encoder_index_key,
                abstained=bool(getattr(span, "abstained", False)),
            )
        )
    return records


def provenance_version_pins(
    records: Iterable[GroundingProvenance],
) -> dict[str, str]:
    """Map canonical HL7 system URIs to the vocabulary version each record used.

    The result is shaped for
    :func:`openmed.clinical.exporters.stamp_coding_provenance` so every emitted
    FHIR coding can be stamped with the same terminology edition that produced
    it. Abstentions, records without a version, and unmappable systems are
    skipped; when several records share a system the last version wins.
    """
    from openmed.clinical.exporters.codeable_concept import SYSTEM_URI

    pins: dict[str, str] = {}
    for record in records:
        if record.abstained or not record.vocab_version or not record.system:
            continue
        uri = SYSTEM_URI.get(record.system.upper())
        if uri is None:
            continue
        pins[uri] = record.vocab_version
    return pins


def scan_provenance_for_raw_text(
    records: Iterable[GroundingProvenance],
    forbidden_texts: Iterable[str],
) -> tuple[str, ...]:
    """Return any forbidden raw-text fragments that leaked into the provenance.

    Serializes the chain and reports which of ``forbidden_texts`` (for example
    the raw note or the raw span surfaces) appear verbatim. An empty result means
    the chain carries only offsets, hashes, and code metadata — never raw note
    text — satisfying the no-PHI invariant.
    """
    payload = json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=False,  # so non-ASCII PHI (e.g. accented names) is matchable
        sort_keys=True,
    )
    leaked = {
        fragment for fragment in forbidden_texts if fragment and fragment in payload
    }
    return tuple(sorted(leaked))
