"""Deterministic multilingual clinical section detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from openmed.clinical.lexicons import (
    available_section_languages,
    get_section_lexicon,
    normalize_section_header,
)
from openmed.processing.lists import ListItemSpan, parse_lists

UNSECTIONED_SECTION = "unsectioned"
LIST_BEARING_SECTION_LABELS = frozenset({"allergies", "medications", "problem_list"})
LIST_SECTION_LOINC_CODES = MappingProxyType(
    {
        "allergies": "48765-2",
        "medications": "10160-0",
        "problem_list": "11450-4",
    }
)
LIST_BEARING_SECTION_LOINC_CODES = frozenset(LIST_SECTION_LOINC_CODES.values())
_HEADER_DELIMITERS = (":", "：", "﹕", "꞉")
_UNDERLINE_CHARS = frozenset("-_=~")
_BULLET_PREFIXES = ("-", "*", "•")


class SectionSpan(dict[str, Any]):
    """A JSON-ready canonical half-open clinical section range.

    ``SectionSpan`` remains a dictionary so existing pipeline and service
    consumers can serialize it directly, while its required fields are also
    available as attributes for a small typed span contract.
    """

    def __init__(self, label: str, start: int, end: int, **metadata: Any) -> None:
        super().__init__(label=label, start=int(start), end=int(end), **metadata)

    @property
    def label(self) -> str:
        """Return the canonical section label."""

        return str(self["label"])

    @property
    def start(self) -> int:
        """Return the inclusive section start offset."""

        return int(self["start"])

    @property
    def end(self) -> int:
        """Return the exclusive section end offset."""

        return int(self["end"])


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    content_end: int


@dataclass(frozen=True)
class _HeaderHit:
    label: str
    start: int
    end: int
    header_start: int
    header_end: int
    content_start: int
    header: str
    language: str


def detect_sections(
    text: str,
    *,
    language: str | None = None,
    include_unsectioned: bool = True,
) -> tuple[SectionSpan, ...]:
    """Segment *text* into canonical clinical section spans.

    Headers are matched at line starts using language-pack section lexicons.
    Colon/full-width-colon headers, standalone headers, and underlined headers
    are supported without whitespace assumptions around CJK or RTL scripts.
    Returned ``label`` values are canonical section keys, so downstream section
    priors can consume them directly.
    """

    if not text:
        return ()

    lines = tuple(_iter_lines(text))
    hits = _dedupe_hits(
        hit
        for index, line in enumerate(lines)
        for hit in _line_header_hits(line, _next_line(lines, index), language)
    )
    if not hits:
        unsectioned_result = (
            (
                _section_dict(
                    label=UNSECTIONED_SECTION,
                    start=0,
                    end=len(text),
                    language=language,
                ),
            )
            if include_unsectioned and text
            else ()
        )
        _validate_section_spans(
            text,
            unsectioned_result,
            require_coverage=include_unsectioned,
        )
        return unsectioned_result

    sections: list[SectionSpan] = []
    cursor = 0
    for index, hit in enumerate(hits):
        if include_unsectioned and cursor < hit.start:
            sections.append(
                _section_dict(
                    label=UNSECTIONED_SECTION,
                    start=cursor,
                    end=hit.start,
                    language=language,
                )
            )
        section_end = hits[index + 1].start if index + 1 < len(hits) else len(text)
        if hit.start < section_end:
            sections.append(
                _section_dict(
                    label=hit.label,
                    start=hit.start,
                    end=section_end,
                    header=hit.header,
                    header_start=hit.header_start,
                    header_end=hit.header_end,
                    content_start=hit.content_start,
                    language=hit.language,
                )
            )
        cursor = section_end

    if include_unsectioned and cursor < len(text):
        sections.append(
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=cursor,
                end=len(text),
                language=language,
            )
        )
    result = tuple(section for section in sections if section["start"] < section["end"])
    _validate_section_spans(
        text,
        result,
        require_coverage=include_unsectioned,
    )
    return result


def validate_section_spans(
    text: str,
    spans: Iterable[Mapping[str, Any]],
) -> None:
    """Assert that section spans form a complete partition of ``text``.

    Spans use half-open character offsets. Each span must have a non-empty
    canonical label, remain within the document, and begin exactly where the
    previous span ends. An empty document is valid only with no spans.

    Raises:
        ValueError: If a span is malformed, out of bounds, empty, overlapping,
            or leaves any document characters uncovered.
    """

    _validate_section_spans(text, spans, require_coverage=True)


def list_section_label(section: str | Mapping[str, Any]) -> str | None:
    """Return the canonical list-bearing label for a label or coded section.

    Mapping inputs may identify a section with a canonical/user-facing
    ``label`` or a LOINC value under ``loinc_code``, ``loinc``, ``code``,
    ``coding``, or ``codes``. Coding mappings are accepted only when their
    optional system identifies LOINC.
    """

    if isinstance(section, str):
        return _canonical_list_label(section)

    label = section.get("label")
    if isinstance(label, str):
        canonical = _canonical_list_label(label)
        if canonical is not None:
            return canonical

    label_by_code = {code: label for label, code in LIST_SECTION_LOINC_CODES.items()}
    for key in ("loinc_code", "loinc", "code", "coding", "codes"):
        for code in _loinc_codes(section.get(key)):
            canonical = label_by_code.get(code)
            if canonical is not None:
                return canonical
    return None


def is_list_bearing_section(section: str | Mapping[str, Any]) -> bool:
    """Return whether a section is a problem, medication, or allergy list."""

    return list_section_label(section) is not None


def parse_section_lists(
    text: str,
    sections: Iterable[Mapping[str, Any]] | None = None,
    *,
    language: str | None = None,
) -> tuple[ListItemSpan, ...]:
    """Parse lists only inside detected or supplied list-bearing sections.

    Detected header text is excluded using ``content_start``. Returned item
    offsets remain absolute to the full source document, and parent indices
    remain valid across multiple list-bearing sections.

    Args:
        text: Full clinical document source.
        sections: Optional section spans. When omitted, :func:`detect_sections`
            supplies canonical section labels and offsets.
        language: Optional language forwarded to section detection.

    Returns:
        Offset-aligned list items ordered across the source document.
    """

    active_sections = (
        detect_sections(text, language=language)
        if sections is None
        else tuple(sections)
    )
    result: list[ListItemSpan] = []
    for index, section in enumerate(active_sections):
        if not isinstance(section, Mapping):
            raise ValueError(f"section span {index} must be a mapping")
        if not is_list_bearing_section(section):
            continue
        start = _optional_section_offset(section, "content_start", section.get("start"))
        end = _optional_section_offset(section, "end", None)
        if start is None or end is None or start < 0 or end > len(text) or start > end:
            raise ValueError(f"section span {index} has invalid list bounds")
        while start < end and text[start] in {" ", "\t"}:
            start += 1

        relative_items = parse_lists(text[start:end])
        item_index_offset = len(result)
        for item in relative_items:
            result.append(
                ListItemSpan(
                    text=item.text,
                    start=item.start + start,
                    end=item.end + start,
                    nesting_level=item.nesting_level,
                    style=item.style,
                    marker=item.marker,
                    marker_start=(
                        item.marker_start + start
                        if item.marker_start is not None
                        else None
                    ),
                    marker_end=(
                        item.marker_end + start if item.marker_end is not None else None
                    ),
                    content_start=(
                        item.content_start + start
                        if item.content_start is not None
                        else None
                    ),
                    parent_index=(
                        item.parent_index + item_index_offset
                        if item.parent_index is not None
                        else None
                    ),
                )
            )
    return tuple(result)


def _validate_section_spans(
    text: str,
    spans: Iterable[Mapping[str, Any]],
    *,
    require_coverage: bool,
) -> None:
    previous_end: int | None = None
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            raise ValueError(f"section span {index} must be a mapping")
        label = span.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"section span {index} requires a non-empty label")
        start = _section_offset(span, "start", index)
        end = _section_offset(span, "end", index)
        if start < 0 or end > len(text):
            raise ValueError(f"section span {index} is outside document bounds")
        if end <= start:
            raise ValueError(f"section span {index} must have positive length")

        if previous_end is None:
            if require_coverage and start != 0:
                raise ValueError(f"section spans leave a gap from 0 to {start}")
        elif start < previous_end:
            raise ValueError(
                f"section spans overlap at offsets {start} to {previous_end}"
            )
        elif start > previous_end:
            raise ValueError(
                f"section spans leave a gap from {previous_end} to {start}"
            )
        previous_end = end

    if require_coverage:
        covered_end = previous_end if previous_end is not None else 0
        if covered_end != len(text):
            raise ValueError(
                f"section spans leave a gap from {covered_end} to {len(text)}"
            )


def _section_offset(span: Mapping[str, Any], key: str, index: int) -> int:
    value = span.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"section span {index} requires an integer {key}")
    return value


def _optional_section_offset(
    span: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> int | None:
    value = span.get(key, fallback)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def _canonical_list_label(label: str) -> str | None:
    normalized = normalize_section_header(label)
    aliases = {
        "allergies": "allergies",
        "allergy list": "allergies",
        "drug allergies": "allergies",
        "current medications": "medications",
        "medication list": "medications",
        "medications": "medications",
        "meds": "medications",
        "problem list": "problem_list",
        "problems": "problem_list",
    }
    return aliases.get(normalized)


def _loinc_codes(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value.strip()
        return
    if isinstance(value, Mapping):
        system = value.get("system")
        code = value.get("code")
        if isinstance(code, str) and (
            system is None or (isinstance(system, str) and "loinc" in system.lower())
        ):
            yield code.strip()
        return
    if isinstance(value, Iterable):
        for entry in value:
            yield from _loinc_codes(entry)


def _iter_lines(text: str) -> Iterable[_Line]:
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        end = cursor + len(raw_line)
        content_end = end - (1 if raw_line.endswith("\n") else 0)
        if raw_line.endswith("\r\n"):
            content_end -= 1
        yield _Line(raw_line, cursor, end, content_end)
        cursor = end
    if not text.endswith(("\n", "\r")) and cursor == 0:
        yield _Line(text, 0, len(text), len(text))


def _next_line(lines: tuple[_Line, ...], index: int) -> _Line | None:
    return lines[index + 1] if index + 1 < len(lines) else None


def _line_header_hits(
    line: _Line,
    next_line: _Line | None,
    language: str | None,
) -> tuple[_HeaderHit, ...]:
    line_text = line.text.rstrip("\r\n")
    if not line_text.strip():
        return ()

    content, content_start = _strip_line_prefix(line_text, line.start)
    if not content:
        return ()

    lookups = _alias_lookups(language)
    delimiter_hit = _delimiter_header_hit(
        content,
        content_start,
        line,
        lookups,
    )
    if delimiter_hit is not None:
        return (delimiter_hit,)

    normalized = normalize_section_header(content)
    for active_language, aliases in lookups:
        label = aliases.get(normalized)
        if label is not None:
            content_start_after_header = (
                next_line.end
                if next_line is not None and _is_underline(next_line.text)
                else line.end
            )
            return (
                _HeaderHit(
                    label=label,
                    start=line.start,
                    end=line.end,
                    header_start=content_start,
                    header_end=content_start + len(content.rstrip()),
                    content_start=content_start_after_header,
                    header=content.strip(),
                    language=active_language,
                ),
            )
    return ()


def _delimiter_header_hit(
    content: str,
    content_start: int,
    line: _Line,
    lookups: tuple[tuple[str, Mapping[str, str]], ...],
) -> _HeaderHit | None:
    delimiter_index = _first_delimiter_index(content)
    if delimiter_index <= 0:
        return None
    candidate = content[:delimiter_index].strip()
    if not candidate:
        return None
    normalized = normalize_section_header(candidate)
    for active_language, aliases in lookups:
        label = aliases.get(normalized)
        if label is None:
            continue
        header_offset = content.find(candidate)
        header_start = content_start + max(header_offset, 0)
        header_end = header_start + len(candidate)
        return _HeaderHit(
            label=label,
            start=line.start,
            end=line.end,
            header_start=header_start,
            header_end=header_end,
            content_start=content_start + delimiter_index + 1,
            header=candidate,
            language=active_language,
        )
    return None


def _first_delimiter_index(content: str) -> int:
    indexes = [content.find(delimiter) for delimiter in _HEADER_DELIMITERS]
    found = [index for index in indexes if index != -1]
    return min(found) if found else -1


def _strip_line_prefix(line_text: str, line_start: int) -> tuple[str, int]:
    offset = len(line_text) - len(line_text.lstrip())
    content = line_text[offset:].rstrip()
    absolute = line_start + offset
    if not content:
        return "", absolute

    for prefix in _BULLET_PREFIXES:
        marker = f"{prefix} "
        if content.startswith(marker):
            return content[len(marker) :].lstrip(), absolute + len(marker)
    dot_index = content.find(". ")
    paren_index = content.find(") ")
    index = min((i for i in (dot_index, paren_index) if i != -1), default=-1)
    if index > 0 and content[:index].isdigit():
        stripped = content[index + 2 :].lstrip()
        return stripped, absolute + index + 2
    return content, absolute


def _is_underline(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 3 and set(stripped) <= _UNDERLINE_CHARS


def _alias_lookups(language: str | None) -> tuple[tuple[str, Mapping[str, str]], ...]:
    languages = (
        tuple(dict.fromkeys((get_section_lexicon(language).language, "en")))
        if language
        else available_section_languages()
    )
    return tuple((code, _aliases_for_language(code)) for code in languages)


def _aliases_for_language(language: str) -> dict[str, str]:
    lexicon = get_section_lexicon(language)
    aliases: dict[str, str] = {}
    for label, headers in lexicon.sections.items():
        aliases[normalize_section_header(label)] = label
        for header in headers:
            aliases[normalize_section_header(header)] = label
    return aliases


def _dedupe_hits(hits: Iterable[_HeaderHit]) -> tuple[_HeaderHit, ...]:
    by_start: dict[int, _HeaderHit] = {}
    for hit in hits:
        previous = by_start.get(hit.start)
        if previous is None or hit.header_end > previous.header_end:
            by_start[hit.start] = hit
    return tuple(by_start[start] for start in sorted(by_start))


def _section_dict(
    *,
    label: str,
    start: int,
    end: int,
    language: str | None,
    header: str | None = None,
    header_start: int | None = None,
    header_end: int | None = None,
    content_start: int | None = None,
) -> SectionSpan:
    section = SectionSpan(
        label=label,
        start=int(start),
        end=int(end),
    )
    loinc_code = LIST_SECTION_LOINC_CODES.get(label)
    if loinc_code is not None:
        section["loinc_code"] = loinc_code
    if header is not None:
        section.update(
            {
                "header": header,
                "header_start": int(
                    header_start if header_start is not None else start
                ),
                "header_end": int(header_end if header_end is not None else start),
                "content_start": int(
                    content_start if content_start is not None else start
                ),
                "language": language,
                "source": "section_header_lexicon",
            }
        )
    elif language:
        section["language"] = language
    return section


__all__ = [
    "LIST_BEARING_SECTION_LABELS",
    "LIST_BEARING_SECTION_LOINC_CODES",
    "LIST_SECTION_LOINC_CODES",
    "SectionSpan",
    "UNSECTIONED_SECTION",
    "detect_sections",
    "is_list_bearing_section",
    "list_section_label",
    "parse_section_lists",
    "validate_section_spans",
]
