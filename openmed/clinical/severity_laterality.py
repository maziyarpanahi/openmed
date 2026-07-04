"""Deterministic severity / laterality modifier extraction (roadmap v1.9).

Findings frequently carry two orthogonal modifier families that a bare concept
mention does not encode:

* **Severity** -- how bad the finding is, expressed on one of three scales:
  descriptive (``mild`` / ``moderate`` / ``severe``), graded (``grade III``),
  or a numeric pain score (``8/10``). All three normalize onto a single ordinal
  bucket so downstream grounding and relation heads can compare them.
* **Laterality / position** -- which side (``left`` / ``right`` / ``bilateral``)
  or anatomical position (``proximal`` / ``distal`` ...) the finding sits on.
  Laterality normalizes to a controlled four-value set; position is kept on a
  separate axis because it is not a side.

This module only extracts the modifiers already present as literal cues in the
text and attaches each to the nearest governing finding span within a bounded
character window. It performs no clinical interpretation, no mapping to a coded
severity terminology, and never echoes the governing finding's surface text --
attachments reference their target by offset only, so the output carries no raw
PHI beyond offsets and normalized controlled values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

SEVERITY_LATERALITY_ADVISORY = (
    "Severity/laterality modifiers are extracted deterministically from literal "
    "cues and attached to the nearest finding by character distance. They are "
    "not a substitute for coded severity terminologies or model-based relation "
    "extraction; review before clinical use."
)

# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------

ModifierKind = Literal["severity", "laterality", "position"]
SeverityScale = Literal["descriptive", "graded", "numeric_pain"]

SEVERITY_NONE = "none"
SEVERITY_MILD = "mild"
SEVERITY_MODERATE = "moderate"
SEVERITY_SEVERE = "severe"

#: Severity buckets ranked from absent to most severe.
SEVERITY_ORDINAL: dict[str, int] = {
    SEVERITY_NONE: 0,
    SEVERITY_MILD: 1,
    SEVERITY_MODERATE: 2,
    SEVERITY_SEVERE: 3,
}

LATERALITY_LEFT = "left"
LATERALITY_RIGHT = "right"
LATERALITY_BILATERAL = "bilateral"
LATERALITY_UNSPECIFIED = "unspecified"

LATERALITY_VALUES = (
    LATERALITY_LEFT,
    LATERALITY_RIGHT,
    LATERALITY_BILATERAL,
    LATERALITY_UNSPECIFIED,
)

# Descriptive severity lexicon -> bucket.
_DESCRIPTIVE_SEVERITY: dict[str, str] = {
    "minimal": SEVERITY_MILD,
    "slight": SEVERITY_MILD,
    "minor": SEVERITY_MILD,
    "mild": SEVERITY_MILD,
    "moderate": SEVERITY_MODERATE,
    "severe": SEVERITY_SEVERE,
    "marked": SEVERITY_SEVERE,
    "significant": SEVERITY_SEVERE,
    "extensive": SEVERITY_SEVERE,
}

# Laterality lexicon -> controlled value.
_LATERALITY_LEXICON: dict[str, str] = {
    "left-sided": LATERALITY_LEFT,
    "left sided": LATERALITY_LEFT,
    "left": LATERALITY_LEFT,
    "lt": LATERALITY_LEFT,
    "right-sided": LATERALITY_RIGHT,
    "right sided": LATERALITY_RIGHT,
    "right": LATERALITY_RIGHT,
    "rt": LATERALITY_RIGHT,
    "bilateral": LATERALITY_BILATERAL,
    "bilaterally": LATERALITY_BILATERAL,
    "b/l": LATERALITY_BILATERAL,
    "both": LATERALITY_BILATERAL,
    "unilateral": LATERALITY_UNSPECIFIED,
}

# Anatomical position modifiers (a separate axis from laterality).
_POSITION_TERMS = (
    "proximal",
    "distal",
    "medial",
    "lateral",
    "anterior",
    "posterior",
    "superior",
    "inferior",
    "dorsal",
    "ventral",
    "upper",
    "lower",
)

# Roman/arabic grade -> bucket. Grade I-II are mild/moderate; III and above severe.
_GRADE_BUCKET: dict[int, str] = {1: SEVERITY_MILD, 2: SEVERITY_MODERATE}
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}


class ModifierAttachment(TypedDict):
    """A modifier attached to its governing finding span.

    The attachment references its target finding by offset only; the finding's
    surface text is never copied into the output.
    """

    kind: ModifierKind
    text: str
    offset: tuple[int, int]
    normalized: str
    ordinal: int | None
    scale: str | None
    target_offset: tuple[int, int]
    target_label: str


# --------------------------------------------------------------------------
# Cue matchers
# --------------------------------------------------------------------------

_DESCRIPTIVE_RE = re.compile(
    r"\b(" + "|".join(sorted(_DESCRIPTIVE_SEVERITY, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_GRADE_RE = re.compile(
    r"\bgrade\s+(iv|iii|ii|i|v|[0-9]+)\b",
    re.IGNORECASE,
)
_PAIN_RE = re.compile(r"\b(10|[0-9])\s*/\s*10\b")
_LATERALITY_RE = re.compile(
    r"(?<![\w/])("
    + "|".join(
        re.escape(term) for term in sorted(_LATERALITY_LEXICON, key=len, reverse=True)
    )
    + r")(?![\w])",
    re.IGNORECASE,
)
_POSITION_RE = re.compile(r"\b(" + "|".join(_POSITION_TERMS) + r")\b", re.IGNORECASE)


def _grade_bucket(token: str) -> str:
    token = token.lower()
    value = _ROMAN.get(token, None)
    if value is None:
        value = int(token)
    return _GRADE_BUCKET.get(value, SEVERITY_SEVERE)


def _pain_bucket(score: int) -> str:
    if score <= 0:
        return SEVERITY_NONE
    if score <= 3:
        return SEVERITY_MILD
    if score <= 6:
        return SEVERITY_MODERATE
    return SEVERITY_SEVERE


def _coerce_finding(span: Mapping[str, object]) -> tuple[int, int, str]:
    try:
        start = int(span["start"])  # type: ignore[arg-type]
        end = int(span["end"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("finding spans require integer 'start' and 'end'") from exc
    if start > end:
        raise ValueError("finding span 'start' must not exceed 'end'")
    label = str(span.get("label", "") or "")
    return start, end, label


def _span_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character gap between two spans; 0 when they touch or overlap."""

    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0


def _attach(
    kind: ModifierKind,
    match_start: int,
    match_end: int,
    surface: str,
    normalized: str,
    ordinal: int | None,
    scale: str | None,
    findings: Sequence[tuple[int, int, str]],
    max_distance: int,
) -> ModifierAttachment | None:
    best: tuple[int, int] | None = None  # (gap, tie-break) index carrier
    best_index = -1
    for index, (f_start, f_end, _label) in enumerate(findings):
        gap = _span_gap(match_start, match_end, f_start, f_end)
        if gap > max_distance:
            continue
        # Prefer the smaller gap; on ties prefer the finding the modifier
        # precedes (pre-modifier reading, e.g. "severe headache").
        precedes = 0 if f_start >= match_end else 1
        key = (gap, precedes)
        if best is None or key < best:
            best = key
            best_index = index
    if best_index < 0:
        return None
    f_start, f_end, f_label = findings[best_index]
    return ModifierAttachment(
        kind=kind,
        text=surface,
        offset=(match_start, match_end),
        normalized=normalized,
        ordinal=ordinal,
        scale=scale,
        target_offset=(f_start, f_end),
        target_label=f_label,
    )


def extract_severity_laterality(
    spans: Iterable[Mapping[str, object]],
    text: str,
    *,
    max_distance: int = 60,
) -> list[ModifierAttachment]:
    """Extract severity / laterality / position modifiers and attach them.

    ``spans`` are the governing finding mentions (mappings with integer
    ``start`` / ``end`` and an optional ``label``). ``text`` is the source
    document. Each literal severity, laterality, or position cue found in
    ``text`` is attached to the nearest finding whose character gap does not
    exceed ``max_distance``; cues with no finding in range are dropped rather
    than misattached.

    Returns a list of :class:`ModifierAttachment` mappings ordered by cue
    offset. The output never echoes a finding's surface text.
    """

    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    findings = [_coerce_finding(span) for span in spans]
    if not findings:
        return []

    attachments: list[ModifierAttachment] = []
    claimed: set[tuple[int, int]] = set()

    def emit(hit: ModifierAttachment | None, key: tuple[int, int]) -> None:
        if hit is not None and key not in claimed:
            claimed.add(key)
            attachments.append(hit)

    # Graded severity first so "grade" is consumed before the descriptive pass.
    for match in _GRADE_RE.finditer(text):
        bucket = _grade_bucket(match.group(1))
        emit(
            _attach(
                "severity",
                match.start(),
                match.end(),
                match.group(0),
                bucket,
                SEVERITY_ORDINAL[bucket],
                "graded",
                findings,
                max_distance,
            ),
            (match.start(), match.end()),
        )

    for match in _PAIN_RE.finditer(text):
        bucket = _pain_bucket(int(match.group(1)))
        emit(
            _attach(
                "severity",
                match.start(),
                match.end(),
                match.group(0),
                bucket,
                SEVERITY_ORDINAL[bucket],
                "numeric_pain",
                findings,
                max_distance,
            ),
            (match.start(), match.end()),
        )

    for match in _DESCRIPTIVE_RE.finditer(text):
        bucket = _DESCRIPTIVE_SEVERITY[match.group(1).lower()]
        emit(
            _attach(
                "severity",
                match.start(),
                match.end(),
                match.group(0),
                bucket,
                SEVERITY_ORDINAL[bucket],
                "descriptive",
                findings,
                max_distance,
            ),
            (match.start(), match.end()),
        )

    for match in _LATERALITY_RE.finditer(text):
        value = _LATERALITY_LEXICON[match.group(1).lower()]
        emit(
            _attach(
                "laterality",
                match.start(),
                match.end(),
                match.group(0),
                value,
                None,
                None,
                findings,
                max_distance,
            ),
            (match.start(), match.end()),
        )

    for match in _POSITION_RE.finditer(text):
        # "lateral" is a position term but also reads as a side qualifier; keep
        # it on the position axis and skip if already claimed as laterality.
        key = (match.start(), match.end())
        if key in claimed:
            continue
        emit(
            _attach(
                "position",
                match.start(),
                match.end(),
                match.group(0),
                match.group(1).lower(),
                None,
                None,
                findings,
                max_distance,
            ),
            key,
        )

    attachments.sort(key=lambda a: a["offset"])
    return attachments


__all__ = [
    "SEVERITY_LATERALITY_ADVISORY",
    "SEVERITY_NONE",
    "SEVERITY_MILD",
    "SEVERITY_MODERATE",
    "SEVERITY_SEVERE",
    "SEVERITY_ORDINAL",
    "LATERALITY_LEFT",
    "LATERALITY_RIGHT",
    "LATERALITY_BILATERAL",
    "LATERALITY_UNSPECIFIED",
    "LATERALITY_VALUES",
    "ModifierKind",
    "SeverityScale",
    "ModifierAttachment",
    "extract_severity_laterality",
]
