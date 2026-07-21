"""Deterministic radiology report section segmentation and stated RADS capture.

The parser segments a free-text radiology report into a
findings / impression / recommendation template and captures a *stated*
BI-RADS or Lung-RADS assessment category when it is explicitly written. It
makes no assessment decision of its own: the category is read verbatim from the
report text, never computed or inferred from the findings.

Segmentation uses a documented heading lexicon matched at line starts, with
sentence-boundary inline cues (e.g. ``IMPRESSION:``) for flattened reports.
Character spans for each captured section are preserved for provenance.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, TypedDict

SpanOffset = tuple[int, int]

RADIOLOGY_REPORT_ADVISORY = (
    "A captured BI-RADS or Lung-RADS category is read verbatim from the report "
    "text, never computed or inferred from the findings. This parser performs "
    "section segmentation and stated-category capture only, and makes no "
    "assessment decision of its own."
)

# Canonical template sections.
FINDINGS = "findings"
IMPRESSION = "impression"
RECOMMENDATION = "recommendation"
SECTION_KEYS: tuple[str, str, str] = (FINDINGS, IMPRESSION, RECOMMENDATION)

AssessmentSystem = Literal["BI-RADS", "Lung-RADS"]

#: Documented heading lexicon: canonical section -> recognised heading phrases.
#: Matched case-insensitively at the start of a line, optionally followed by a
#: colon. Longer phrases are tried before shorter ones so ``FOLLOW-UP`` wins
#: over a bare ``PLAN`` alias.
SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    FINDINGS: (
        "findings",
        "finding",
        "observations",
    ),
    IMPRESSION: (
        "impression",
        "impressions",
        "conclusion",
        "conclusions",
        "interpretation",
    ),
    RECOMMENDATION: (
        "recommendations",
        "recommendation",
        "recommended follow-up",
        "follow-up recommendations",
        "follow-up",
        "follow up",
    ),
}

#: Inline cue phrases that introduce a section in flattened report text. Cues
#: must occur at the start of the report or after sentence/line punctuation, so
#: ordinary prose such as ``not an impression:`` is not treated as a label.
SECTION_INLINE_CUES: dict[str, tuple[str, ...]] = {
    FINDINGS: ("findings:", "observations:"),
    IMPRESSION: (
        "clinical impression:",
        "overall impression:",
        "impression:",
        "conclusion:",
        "interpretation:",
    ),
    RECOMMENDATION: (
        "recommendation:",
        "recommendations:",
        "recommended follow-up:",
        "follow-up:",
    ),
}


class RadiologyReportTemplate(TypedDict):
    """Segmented radiology report with a stated (never inferred) RADS category."""

    findings_text: str
    impression_text: str
    recommendation_text: str
    assessment_system: Optional[str]
    assessment_category: Optional[str]
    section_spans: dict[str, Optional[SpanOffset]]


# --- section marker detection -------------------------------------------------


def _heading_pattern() -> re.Pattern[str]:
    """Build the line-start heading pattern (section captured by named group)."""
    parts = []
    for section, phrases in SECTION_HEADINGS.items():
        # Longest-first so multi-word aliases win over their prefixes.
        alternation = "|".join(
            re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True)
        )
        parts.append(f"(?P<{section}>{alternation})")
    body = "|".join(parts)
    # A heading occupies the start of a line and is either followed by a colon
    # (content may continue on the same line, e.g. ``FINDINGS: ...``) or stands
    # alone on its own line (bare/underlined heading). Content begins at the end
    # of the match, so the heading label never leaks into the section text.
    return re.compile(
        rf"(?im)^[ \t]*(?:{body})[ \t]*(?::[ \t]*|(?=\r?\n|$))",
    )


_HEADING_RE = _heading_pattern()


class _Marker:
    __slots__ = ("section", "content_start")

    def __init__(self, section: str, content_start: int):
        self.section = section
        self.content_start = content_start


def _matched_section(match: re.Match[str]) -> str:
    for section in SECTION_KEYS:
        if match.groupdict().get(section):
            return section
    raise AssertionError("marker matched no section group")  # pragma: no cover


def _heading_markers(text: str) -> list[tuple[int, _Marker]]:
    markers: list[tuple[int, _Marker]] = []
    for match in _HEADING_RE.finditer(text):
        markers.append((match.start(), _Marker(_matched_section(match), match.end())))
    return markers


def _inline_cue_markers(text: str) -> list[tuple[int, _Marker]]:
    markers: list[tuple[int, _Marker]] = []
    for section, cues in SECTION_INLINE_CUES.items():
        alternatives = "|".join(
            re.escape(cue) for cue in sorted(cues, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?i)(?:^|(?<=[.!?;\n]))[ \t]*(?P<cue>{alternatives})[ \t]*"
        )
        for match in pattern.finditer(text):
            markers.append((match.start("cue"), _Marker(section, match.end())))
    return markers


def _dedupe_markers(markers: list[tuple[int, _Marker]]) -> list[tuple[int, _Marker]]:
    """Keep the first marker per section, ordered by position."""
    ordered = sorted(markers, key=lambda item: item[0])
    seen: set[str] = set()
    unique: list[tuple[int, _Marker]] = []
    for start, marker in ordered:
        if marker.section in seen:
            continue
        seen.add(marker.section)
        unique.append((start, marker))
    return unique


def _segment(text: str) -> dict[str, Optional[SpanOffset]]:
    """Return canonical section -> (start, end) char span in ``text`` (or None)."""
    spans: dict[str, Optional[SpanOffset]] = {key: None for key in SECTION_KEYS}
    if not text.strip():
        return spans

    markers = _dedupe_markers([*_heading_markers(text), *_inline_cue_markers(text)])

    if not markers:
        # No headings or cues: the whole report is findings prose.
        spans[FINDINGS] = _trim_span(text, 0, len(text))
        return spans

    # Prose before the first marker is findings context when findings has no
    # explicit heading of its own.
    boundaries = [start for start, _ in markers]
    first_marker_start = boundaries[0]
    if spans_needs_leading_findings(markers) and first_marker_start > 0:
        leading = _trim_span(text, 0, first_marker_start)
        if leading is not None:
            spans[FINDINGS] = leading

    for index, (_, marker) in enumerate(markers):
        content_start = marker.content_start
        content_end = (
            boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
        )
        trimmed = _trim_span(text, content_start, content_end)
        if trimmed is not None and spans.get(marker.section) is None:
            spans[marker.section] = trimmed

    return spans


def spans_needs_leading_findings(markers: list[tuple[int, _Marker]]) -> bool:
    """True when no findings marker exists, so leading prose seeds findings."""
    return all(marker.section != FINDINGS for _, marker in markers)


def _trim_span(text: str, start: int, end: int) -> Optional[SpanOffset]:
    """Trim surrounding whitespace, returning offsets into ``text`` or None."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return (start, end)


def _span_text(text: str, span: Optional[SpanOffset]) -> str:
    if span is None:
        return ""
    return text[span[0] : span[1]]


# --- stated RADS category capture --------------------------------------------

# A stated category is captured only when it is unambiguously an assessment:
# either a qualifier word (``category``/``assessment``/``score``) introduces the
# number, or the number is "terminal" — immediately followed by end-of-clause
# punctuation, a line break, or end of text. This rejects counts and scale
# legends adjacent to the token (e.g. ``BI-RADS 4 lesions``, ``BI-RADS 0-6
# scale``), which are not stated assessments and must never be inferred.
_TERMINAL = r"(?=\s*(?:[.,;)\]]|\r?\n|$))"
_QUALIFIER = r"(?:(?:final\s+)?assessment(?:\s+category)?|category|score)"
_ASSIGNMENT = r"\s*(?:is\s*)?[:=]?\s*"
_BIRADS_RE = re.compile(
    r"(?i)\bBI[- ]?RADS\b[\s:]*"
    r"(?:"
    rf"{_QUALIFIER}{_ASSIGNMENT}(?P<q>[0-6])\b"
    rf"|(?P<t>[0-6]){_TERMINAL}"
    r")"
)
_LUNGRADS_RE = re.compile(
    r"(?i)\bLung[- ]?RADS\b[\s:]*"
    r"(?:"
    rf"{_QUALIFIER}{_ASSIGNMENT}(?P<q>4A|4B|4X|[0-3])\b"
    rf"|(?P<t>4A|4B|4X|[0-3]){_TERMINAL}"
    r")"
)
_REFERENCE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:atlas|categories|guideline|legend|lexicon|scale)\b"
)


def _is_reference_context(text: str, match_start: int) -> bool:
    """Return whether a RADS token is introduced as reference material."""
    clause_start = max(
        text.rfind("\n", 0, match_start),
        text.rfind(".", 0, match_start),
        text.rfind(";", 0, match_start),
    )
    return (
        _REFERENCE_CONTEXT_RE.search(text[clause_start + 1 : match_start]) is not None
    )


def _capture_assessment(text: str) -> tuple[Optional[str], Optional[str]]:
    """Capture a stated BI-RADS/Lung-RADS category; never infer one.

    Returns the earliest explicitly-stated ``(system, category)`` pair, or
    ``(None, None)`` when no category is written in the report.
    """
    candidates: list[tuple[int, str, str]] = []
    birads = next(
        (
            match
            for match in _BIRADS_RE.finditer(text)
            if not _is_reference_context(text, match.start())
        ),
        None,
    )
    if birads is not None:
        candidates.append(
            (birads.start(), "BI-RADS", birads.group("q") or birads.group("t"))
        )
    lungrads = next(
        (
            match
            for match in _LUNGRADS_RE.finditer(text)
            if not _is_reference_context(text, match.start())
        ),
        None,
    )
    if lungrads is not None:
        category = (lungrads.group("q") or lungrads.group("t")).upper()
        candidates.append((lungrads.start(), "Lung-RADS", category))
    if not candidates:
        return (None, None)
    candidates.sort(key=lambda item: item[0])
    _, system, category = candidates[0]
    return (system, category)


def parse_radiology_report(text: str) -> RadiologyReportTemplate:
    """Segment a radiology report and capture a stated RADS category.

    Args:
        text: Raw radiology report text.

    Returns:
        A :class:`RadiologyReportTemplate`. ``section_spans`` maps each of
        ``findings``/``impression``/``recommendation`` to its character span in
        ``text`` (or ``None`` when absent). ``assessment_system`` and
        ``assessment_category`` are populated only when a BI-RADS or Lung-RADS
        category is explicitly written; otherwise both are ``None``.
    """
    source = text or ""
    spans = _segment(source)
    system, category = _capture_assessment(source)
    return RadiologyReportTemplate(
        findings_text=_span_text(source, spans[FINDINGS]),
        impression_text=_span_text(source, spans[IMPRESSION]),
        recommendation_text=_span_text(source, spans[RECOMMENDATION]),
        assessment_system=system,
        assessment_category=category,
        section_spans=spans,
    )


__all__ = [
    "FINDINGS",
    "IMPRESSION",
    "RADIOLOGY_REPORT_ADVISORY",
    "RECOMMENDATION",
    "SECTION_HEADINGS",
    "SECTION_INLINE_CUES",
    "SECTION_KEYS",
    "AssessmentSystem",
    "RadiologyReportTemplate",
    "SpanOffset",
    "parse_radiology_report",
]
