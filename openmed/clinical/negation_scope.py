"""Negation-scope boundary detection for clinical spans (roadmap v1.9).

The shipped ConText negation layer (:mod:`openmed.clinical.context`) decides
*whether* a negation cue is present, scoping it to the sentence. That is too
coarse: in ``"no chest pain but has fever"`` a sentence-level scope wrongly
negates ``fever``. This module computes the exact token region each negation
cue governs, terminating the scope at conjunctions, clause breaks, and
scope-terminating cue words, so an entity is negated only when it falls inside a
cue's computed influence span.

Forward cues (``no``, ``denies``, ``no evidence of`` ...) govern the text to
their right; backward cues (``ruled out``, ``absent`` ...) govern the text to
their left. Pseudo-negation phrases (``cannot be excluded``, ``not ruled out``)
are masked first so they never open a scope. The detector is deterministic and
offline, and reuses the authoritative cue lexicons from
:mod:`openmed.clinical.context` so the two layers cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .context import NEGATION_CUES, PSEUDO_NEGATION_CUES

NEGATION_SCOPE_ADVISORY = (
    "Negation scopes are computed deterministically from surface cues and "
    "clause boundaries. They sharpen assertion accuracy but are not a "
    "substitute for clinician review; verify before clinical use."
)

Direction = str  # "forward" | "backward"

FORWARD: Direction = "forward"
BACKWARD: Direction = "backward"

# Post-coordinated cues that govern the text to their LEFT rather than their
# right (kept in sync with the backward cues used by the ConText layer).
_BACKWARD_CUES: frozenset[str] = frozenset(
    {
        "absent",
        "not present",
        "ruled out",
        "resolved",
    }
)

# A negation scope terminates at sentence punctuation, coordinating
# conjunctions, and classic NegEx clause-break terminators. The punctuation and
# ``and``/``or``/``but``/``however`` set mirror the shipped ConText
# ``_SCOPE_TERMINATOR_RE``; the remaining NegEx terminators sharpen the boundary
# beyond sentence-level scoping. Comma is intentionally not a terminator so
# coordinated negation lists ("no fever, chills, or cough") stay in scope.
_TERMINATOR_RE = re.compile(
    r"(?:[.!?;]|(?<!\w)(?:"
    r"and|but|however|or|although|though|nevertheless|yet|still|"
    r"except|aside from|apart from|secondary to|which|who|because"
    r")(?!\w))",
    re.IGNORECASE,
)


def _cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(cues, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_NEGATION_RE = _cue_pattern(NEGATION_CUES)
_PSEUDO_NEGATION_RE = _cue_pattern(PSEUDO_NEGATION_CUES)


@dataclass(frozen=True)
class NegationScope:
    """The region of text governed by one negation cue.

    ``scope_start`` / ``scope_end`` are half-open character offsets into the
    source text bounding the governed region (excluding the cue itself).
    ``governed`` holds the ``(start, end)`` offsets of the input entity spans
    that fall entirely inside that region.
    """

    cue: str
    cue_start: int
    cue_end: int
    direction: Direction
    scope_start: int
    scope_end: int
    governed: tuple[tuple[int, int], ...]


def _mask_pseudo_negation(text: str) -> str:
    """Blank out pseudo-negation phrases, preserving every character offset."""

    return _PSEUDO_NEGATION_RE.sub(lambda match: " " * len(match.group(0)), text)


def _normalize_cue(surface: str) -> str:
    return " ".join(surface.casefold().split())


def _forward_bound(text: str, start: int) -> int:
    """First terminator at or after ``start``; end of text if none."""

    match = _TERMINATOR_RE.search(text, start)
    return match.start() if match else len(text)


def _backward_bound(text: str, end: int) -> int:
    """Offset just after the last terminator before ``end``; 0 if none."""

    last = 0
    for match in _TERMINATOR_RE.finditer(text, 0, end):
        last = match.end()
    return last


def _coerce_span(span: Mapping[str, object]) -> tuple[int, int]:
    try:
        start = int(span["start"])  # type: ignore[arg-type]
        end = int(span["end"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("entity spans require integer 'start' and 'end'") from exc
    if start > end:
        raise ValueError("entity span 'start' must not exceed 'end'")
    return start, end


def detect_negation_scopes(
    text: str,
    spans: Iterable[Mapping[str, object]] = (),
) -> list[NegationScope]:
    """Compute the governed scope of every negation cue in ``text``.

    ``spans`` are optional entity mentions (mappings with integer ``start`` /
    ``end``); each is attached to a scope's ``governed`` tuple when it lies
    fully inside that scope. Pseudo-negation phrases are masked first and never
    open a scope. Scopes are returned ordered by cue offset.
    """

    entities = [_coerce_span(span) for span in spans]
    masked = _mask_pseudo_negation(text)

    scopes: list[NegationScope] = []
    for match in _NEGATION_RE.finditer(masked):
        cue_start, cue_end = match.start(), match.end()
        surface = text[cue_start:cue_end]
        direction = BACKWARD if _normalize_cue(surface) in _BACKWARD_CUES else FORWARD
        if direction == FORWARD:
            scope_start, scope_end = cue_end, _forward_bound(text, cue_end)
        else:
            scope_start, scope_end = _backward_bound(text, cue_start), cue_start

        governed = tuple(
            (start, end)
            for start, end in entities
            if start >= scope_start and end <= scope_end
        )
        scopes.append(
            NegationScope(
                cue=surface,
                cue_start=cue_start,
                cue_end=cue_end,
                direction=direction,
                scope_start=scope_start,
                scope_end=scope_end,
                governed=governed,
            )
        )
    return scopes


def negated_spans(
    text: str,
    spans: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int], ...]:
    """Return the ``(start, end)`` offsets of spans that fall inside a scope.

    A span is negated only when it lies within some negation cue's computed
    scope, not merely the same sentence. Offsets are returned in first-seen
    order without duplicates.
    """

    scopes = detect_negation_scopes(text, spans)
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for scope in scopes:
        for offset in scope.governed:
            if offset not in seen:
                seen.add(offset)
                ordered.append(offset)
    return tuple(ordered)


__all__ = [
    "NEGATION_SCOPE_ADVISORY",
    "FORWARD",
    "BACKWARD",
    "Direction",
    "NegationScope",
    "detect_negation_scopes",
    "negated_spans",
]
