"""Deterministic extraction of explicitly reported pathology result fields.

Pathology reports commonly place specimen, diagnosis, grade, and biomarker
results in labelled sections.  This module keeps those fields separate and
links every returned value to a half-open character span in the caller's
source text.  It uses only a small, transparent section and label grammar: it
does not infer a diagnosis, interpret a biomarker, derive a grade, or query a
terminology service.

The returned records intentionally do not contain the source document or any
logging payload.  Callers that need to display evidence can use the offsets
against their own protected source buffer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, TypedDict

PATHOLOGY_RESULT_ADVISORY = (
    "Pathology result extraction is deterministic assistive tooling for "
    "explicitly reported specimen, diagnosis, grade, and biomarker fields. "
    "It preserves evidence spans, performs no clinical inference or "
    "terminology lookup, and is not a diagnostic decision engine or a "
    "substitute for pathologist judgment."
)

PathologyFieldName = Literal["specimen", "diagnosis", "grade"]


class PathologyEvidenceSpan(TypedDict):
    """A half-open character range into the caller-owned source text."""

    start: int
    end: int


class PathologyField(TypedDict):
    """One explicitly reported pathology field and its source evidence."""

    value: str
    span: PathologyEvidenceSpan


class PathologyBiomarker(TypedDict):
    """One reported biomarker name/result pair with source evidence."""

    name: str
    result: str
    span: PathologyEvidenceSpan
    name_span: PathologyEvidenceSpan
    result_span: PathologyEvidenceSpan


class PathologyResult(TypedDict):
    """Structured pathology fields in deterministic source order.

    Each item in ``specimen``, ``diagnosis``, and ``grade`` contains the
    explicitly written ``value`` plus its ``span``.  Biomarkers retain the
    written marker ``name`` and ``result`` separately, along with spans for
    both parts and their combined line.  Empty lists mean that no supported,
    explicitly labelled field was found; they never represent an inferred
    negative result.
    """

    specimen: list[PathologyField]
    diagnosis: list[PathologyField]
    grade: list[PathologyField]
    biomarkers: list[PathologyBiomarker]
    advisory: str


PATHOLOGY_FIELD_NAMES: tuple[str, ...] = (
    "specimen",
    "diagnosis",
    "grade",
    "biomarkers",
)

_SECTION_ALIASES: dict[str, str | None] = {
    "specimen": "specimen",
    "specimens": "specimen",
    "material": "specimen",
    "materials": "specimen",
    "diagnosis": "diagnosis",
    "diagnoses": "diagnosis",
    "final diagnosis": "diagnosis",
    "pathologic diagnosis": "diagnosis",
    "pathological diagnosis": "diagnosis",
    "microscopic diagnosis": "diagnosis",
    "interpretation": "diagnosis",
    "grade": "grade",
    "grading": "grade",
    "histologic grade": "grade",
    "histological grade": "grade",
    "gleason score": "grade",
    "biomarker": "biomarkers",
    "biomarkers": "biomarkers",
    "biomarker results": "biomarkers",
    "immunohistochemistry": "biomarkers",
    "immunohistochemical stains": "biomarkers",
    "molecular": "biomarkers",
    "molecular testing": "biomarkers",
    "molecular results": "biomarkers",
    "ancillary": "biomarkers",
    "ancillary studies": "biomarkers",
    "comment": None,
    "comments": None,
    "clinical history": None,
    "gross description": None,
    "microscopic description": None,
    "synoptic": None,
    "staging": None,
}

_FIELD_ALIASES: dict[str, str] = {
    label: field for label, field in _SECTION_ALIASES.items() if field is not None
}
_FIELD_LABELS = "|".join(
    re.escape(label) for label in sorted(_FIELD_ALIASES, key=len, reverse=True)
)
_SECTION_LABELS = "|".join(
    re.escape(label) for label in sorted(_SECTION_ALIASES, key=len, reverse=True)
)

# A field label is accepted only at the start of a line and only with an
# explicit separator.  This avoids treating ordinary prose such as
# ``the diagnosis was discussed`` as a pathology result field.
_FIELD_LABEL_RE = re.compile(
    rf"^\s*(?:[-*•]\s*|(?:\(?[A-Za-z0-9]+[.)])\s+)?"
    rf"(?P<label>{_FIELD_LABELS})\s*(?::|[-–])\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    rf"^\s*(?:[-*•]\s*)?(?P<label>{_SECTION_LABELS})"
    rf"(?:\s*:\s*(?P<value>.*?))?\s*$",
    re.IGNORECASE,
)

# Unknown all-caps headings close a supported section, but only when they are
# clearly heading-shaped.  This keeps fields such as patient and accession
# metadata out of the result without storing or reporting those values.
_UNKNOWN_HEADING_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?[A-Z][A-Z0-9 /&_()\-]{2,}(?::)?\s*$"
)
_METADATA_LINE_RE = re.compile(
    r"^\s*(?:"
    r"patient(?:\s+(?:name|id|identifier))?|"
    r"accession(?:\s+(?:number|id))?|"
    r"medical\s+record(?:\s+(?:number|id))?|"
    r"mrn|dob|date\s+of\s+birth|case(?:\s+(?:number|id))?|"
    r"ordering\s+(?:clinician|provider)|physician"
    r")\s*(?::|=|[-–])\s*",
    re.IGNORECASE,
)
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]\s+|(?:\(?[A-Za-z0-9]+[.)])\s+)")

_BIOMARKER_SEPARATOR_RE = re.compile(r"\s*(?::|=|\b(?:is|was)\b)\s*", re.I)
_BIOMARKER_RESULT_RE = re.compile(
    r"(?i)\b(?:"
    r"not\s+detected|not\s+amplified|not\s+expressed|"
    r"positive|negative|detected|amplified|expressed|"
    r"stable|unstable|wild[- ]type|mutated|mutation|"
    r"loss|retained|equivocal|indeterminate|pending|"
    r"high|low|intact|abnormal|normal"
    r")\b"
)
_NON_BIOMARKER_NAMES = frozenset(
    {
        "accession",
        "case",
        "comment",
        "control",
        "date",
        "date of birth",
        "dob",
        "identifier",
        "internal control",
        "patient",
        "patient id",
        "physician",
        "provider",
        "quality control",
        "specimen",
    }
)

_GRADE_LABEL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?:(?:histologic|histological)\s+)?grade\s*(?::|=)?\s*"
    r"(?P<value>(?:[1-4](?:\s*(?:of|/)\s*[1-4])?|[Xx]|"
    r"unknown|pending|not\s+(?:reported|identified|applicable)))"
    r"(?![A-Za-z0-9])"
)
_GLEASON_INLINE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])gleason\s+score\s*(?::|=)?\s*"
    r"(?P<value>\d+\s*\+\s*\d+(?:\s*=\s*\d+)?)"
    r"(?![A-Za-z0-9])"
)
_G_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<value>G(?:[1-4X]))(?![A-Za-z0-9])")


def _empty_result() -> PathologyResult:
    return PathologyResult(
        specimen=[],
        diagnosis=[],
        grade=[],
        biomarkers=[],
        advisory=PATHOLOGY_RESULT_ADVISORY,
    )


def _trimmed_range(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Return a non-whitespace range without exposing source text in errors."""
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _content_range(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim a line and its optional list marker for field evidence."""
    trimmed = _trimmed_range(source, start, end)
    if trimmed is None:
        return None
    start, end = trimmed
    marker = _LIST_PREFIX_RE.match(source[start:end])
    if marker is not None:
        start += marker.end()
    return _trimmed_range(source, start, end)


def _span(start: int, end: int) -> PathologyEvidenceSpan:
    return {"start": start, "end": end}


def _field(source: str, start: int, end: int) -> PathologyField | None:
    value_range = _content_range(source, start, end)
    if value_range is None:
        return None
    value_start, value_end = value_range
    return {
        "value": source[value_start:value_end],
        "span": _span(value_start, value_end),
    }


def _add_field(
    fields: dict[str, list[PathologyField]],
    field_name: PathologyFieldName,
    source: str,
    start: int,
    end: int,
) -> None:
    item = _field(source, start, end)
    if item is None:
        return
    if item not in fields[field_name]:
        fields[field_name].append(item)


def _biomarker_name_is_safe(name: str) -> bool:
    normalized = " ".join(name.split()).casefold().strip(" -")
    if not normalized or normalized in _NON_BIOMARKER_NAMES:
        return False
    if len(normalized) > 80 or not re.search(r"[A-Za-z]", normalized):
        return False
    return True


def _biomarker(
    source: str,
    start: int,
    end: int,
) -> PathologyBiomarker | None:
    """Parse one biomarker line without interpreting its reported result."""
    content_range = _content_range(source, start, end)
    if content_range is None:
        return None
    content_start, content_end = content_range
    content = source[content_start:content_end]

    separator = _BIOMARKER_SEPARATOR_RE.search(content)
    if separator is not None:
        name_start = content_start
        name_end = content_start + separator.start()
        result_start = content_start + separator.end()
    else:
        result_match = _BIOMARKER_RESULT_RE.search(content)
        if result_match is None:
            return None
        name_start = content_start
        name_end = content_start + result_match.start()
        result_start = content_start + result_match.start()

    name_range = _trimmed_range(source, name_start, name_end)
    result_range = _trimmed_range(source, result_start, content_end)
    if name_range is None or result_range is None:
        return None
    name_start, name_end = name_range
    result_start, result_end = result_range
    name = source[name_start:name_end]
    if not _biomarker_name_is_safe(name):
        return None
    return {
        "name": name,
        "result": source[result_start:result_end],
        "span": _span(name_start, result_end),
        "name_span": _span(name_start, name_end),
        "result_span": _span(result_start, result_end),
    }


def _add_biomarker(
    biomarkers: list[PathologyBiomarker],
    source: str,
    start: int,
    end: int,
) -> None:
    item = _biomarker(source, start, end)
    if item is not None and item not in biomarkers:
        biomarkers.append(item)


def _add_inline_grades(
    fields: dict[str, list[PathologyField]],
    source: str,
    start: int,
    end: int,
) -> None:
    """Keep only grade values that have an explicit grade marker."""
    for pattern in (_GRADE_LABEL_RE, _GLEASON_INLINE_RE, _G_TOKEN_RE):
        for match in pattern.finditer(source, start, end):
            value_start, value_end = match.span("value")
            _add_field(fields, "grade", source, value_start, value_end)


def _section_from_label(label: str) -> str | None:
    normalized = " ".join(label.split()).casefold()
    return _SECTION_ALIASES.get(normalized)


def _field_from_label(label: str) -> str | None:
    normalized = " ".join(label.split()).casefold()
    return _FIELD_ALIASES.get(normalized)


def _line_items(source: str) -> Iterable[tuple[int, int, str]]:
    """Yield source offsets for lines while retaining code-point positions."""
    offset = 0
    for raw_line in source.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        yield offset, offset + len(line), line
        offset += len(raw_line)
    if offset < len(source):
        yield offset, len(source), source[offset:]


def extract_pathology_result(text: str) -> PathologyResult:
    """Extract explicitly reported pathology fields from ``text``.

    The parser accepts labelled fields such as ``SPECIMEN: ...`` and
    ``FINAL DIAGNOSIS: ...`` plus their supported section forms.  Multiple
    values are retained in source order.  Biomarkers are split only when a
    line contains an explicit ``:``, ``=``, ``is``, or ``was`` separator, or a
    transparent reported-result cue such as ``positive`` or ``not detected``.
    Values are copied only into the structured field required by this profile;
    the source document itself is never returned.

    Args:
        text: Caller-owned pathology report text. Character offsets in the
            result refer to this exact string.

    Returns:
        A deterministic :class:`PathologyResult` with empty lists for fields
        that are absent or not explicitly reported.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("pathology result text must be a string")
    if not text.strip():
        return _empty_result()

    fields: dict[str, list[PathologyField]] = {
        "specimen": [],
        "diagnosis": [],
        "grade": [],
    }
    biomarkers: list[PathologyBiomarker] = []
    current_section: str | None = None

    for line_start, line_end, line in _line_items(text):
        if not line.strip():
            continue
        if _METADATA_LINE_RE.match(line) is not None:
            current_section = None
            continue

        field_match = _FIELD_LABEL_RE.match(line)
        if field_match is not None:
            field_name = _field_from_label(field_match.group("label"))
            if field_name is None:
                current_section = None
                continue
            current_section = field_name
            value_start = line_start + field_match.start("value")
            value_end = line_start + field_match.end("value")
            if field_name == "biomarkers":
                _add_biomarker(biomarkers, text, value_start, value_end)
            elif field_name == "grade":
                _add_field(fields, "grade", text, value_start, value_end)
            else:
                _add_field(fields, field_name, text, value_start, value_end)
                if field_name == "diagnosis":
                    _add_inline_grades(fields, text, value_start, value_end)
            continue

        section_match = _SECTION_RE.match(line)
        if section_match is not None:
            current_section = _section_from_label(section_match.group("label"))
            value = section_match.group("value")
            if current_section is not None and value and value.strip():
                value_start = line_start + section_match.start("value")
                value_end = line_start + section_match.end("value")
                if current_section == "biomarkers":
                    _add_biomarker(biomarkers, text, value_start, value_end)
                elif current_section == "grade":
                    _add_field(fields, "grade", text, value_start, value_end)
                else:
                    _add_field(fields, current_section, text, value_start, value_end)
            continue

        if _UNKNOWN_HEADING_RE.fullmatch(line):
            current_section = None
            continue

        if current_section == "specimen":
            _add_field(fields, "specimen", text, line_start, line_end)
        elif current_section == "diagnosis":
            _add_field(fields, "diagnosis", text, line_start, line_end)
            _add_inline_grades(fields, text, line_start, line_end)
        elif current_section == "grade":
            _add_field(fields, "grade", text, line_start, line_end)
        elif current_section == "biomarkers":
            _add_biomarker(biomarkers, text, line_start, line_end)

        if current_section not in {"specimen", "biomarkers"}:
            _add_inline_grades(fields, text, line_start, line_end)

    for field_name in fields:
        fields[field_name].sort(
            key=lambda item: (
                item["span"]["start"],
                item["span"]["end"],
                item["value"],
            )
        )
    biomarkers.sort(
        key=lambda item: (
            item["span"]["start"],
            item["span"]["end"],
            item["name"],
            item["result"],
        )
    )
    return PathologyResult(
        specimen=fields["specimen"],
        diagnosis=fields["diagnosis"],
        grade=fields["grade"],
        biomarkers=biomarkers,
        advisory=PATHOLOGY_RESULT_ADVISORY,
    )


def extract_pathology_profile(text: str) -> PathologyResult:
    """Alias for :func:`extract_pathology_result` for profile-oriented callers."""
    return extract_pathology_result(text)


__all__ = [
    "PATHOLOGY_FIELD_NAMES",
    "PATHOLOGY_RESULT_ADVISORY",
    "PathologyBiomarker",
    "PathologyEvidenceSpan",
    "PathologyField",
    "PathologyFieldName",
    "PathologyResult",
    "extract_pathology_profile",
    "extract_pathology_result",
]
