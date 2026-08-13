"""Deterministic multilingual clinical section detection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from openmed.clinical.data.section_loinc_map import (
    SECTION_LOINC_MAP,
    section_codes,
    section_codings,
    section_loinc_code,
)
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
        label: SECTION_LOINC_MAP[label]
        for label in ("allergies", "medications", "problem_list")
    }
)
CONTEXT_SECTION_LOINC_CODES = MappingProxyType(
    {
        label: SECTION_LOINC_MAP[label]
        for label in ("family_history", "past_medical_history")
    }
)
SECTION_LOINC_CODES = SECTION_LOINC_MAP
LIST_BEARING_SECTION_LOINC_CODES = frozenset(LIST_SECTION_LOINC_CODES.values())
_HEADER_DELIMITERS = (":", "：", "﹕", "꞉")
_UNDERLINE_CHARS = frozenset("-_=~")
_BULLET_PREFIXES = ("-", "*", "•")
_LEARNED_REFINEMENT_CONFIDENCE = 0.82


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

    @property
    def source(self) -> str:
        """Return the detector source, defaulting to the rules stage."""

        return str(self.get("source", "rule"))

    @property
    def confidence(self) -> float:
        """Return the bounded section-boundary confidence."""

        try:
            return min(max(float(self.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def codes(self) -> list[str]:
        """Return the LOINC code list for this section."""

        raw_codes = self.get("codes")
        if isinstance(raw_codes, list):
            return [
                code if isinstance(code, str) else str(code["code"])
                for code in raw_codes
                if isinstance(code, str)
                or (isinstance(code, Mapping) and isinstance(code.get("code"), str))
            ]
        return section_codes(self.label)

    @property
    def codings(self) -> list[dict[str, str]]:
        """Return JSON-ready LOINC codings for this section."""

        raw_codings = self.get("coding")
        if isinstance(raw_codings, list):
            return [dict(code) for code in raw_codings if isinstance(code, Mapping)]
        return section_codings(self.label)


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


@dataclass(frozen=True)
class _SectionCandidate:
    span: SectionSpan
    header_start: int
    header_end: int
    registration_order: int

    @property
    def header_length(self) -> int:
        return self.header_end - self.header_start


_SectionSegmenter = Callable[[str, str | None], tuple[SectionSpan, ...]]


def detect_sections(
    text: str,
    *,
    language: str | None = None,
    include_unsectioned: bool = True,
    use_learned: bool = False,
    learned: bool | None = None,
    learned_head: Any | None = None,
    model_path: str | None = None,
) -> tuple[SectionSpan, ...]:
    """Run rules-first section detection with optional learned refinement.

    Candidate headers from the language-pack lexicon and focused section-family
    segmenters are merged deterministically. Overlapping header matches prefer
    the longest match, then the earliest section start, then registration order
    for an exact tie. Returned spans are sorted and, by default, ``unsectioned``
    spans fill any uncovered ranges so the result covers all of *text*.

    ``use_learned`` is opt-in. The learned stage is invoked only when the rules
    output contains an unsectioned gap or a low-confidence boundary. A caller
    may inject a local predictor with ``learned_head`` or point ``model_path``
    at a local MLX artifact; neither path is loaded at import time. The alias
    ``learned`` is accepted for concise call sites.
    """

    if not text:
        validate_sections(text, ())
        return ()

    if learned is not None:
        if use_learned and bool(learned) != use_learned:
            raise ValueError("learned and use_learned disagree")
        use_learned = bool(learned)
    if learned_head is not None or model_path is not None:
        use_learned = True

    result = _detect_sections_rules(
        text,
        language=language,
        include_unsectioned=include_unsectioned,
    )
    if not use_learned:
        return result

    return _refine_sections_with_learned(
        text,
        result,
        language=language,
        include_unsectioned=include_unsectioned,
        learned_head=learned_head,
        model_path=model_path,
    )


def _detect_sections_rules(
    text: str,
    *,
    language: str | None,
    include_unsectioned: bool,
) -> tuple[SectionSpan, ...]:
    """Run only the registered deterministic section segmenters."""

    candidates = _section_candidates(text, language)
    result = _assemble_sections(
        text,
        _resolve_overlapping_headers(candidates),
        language=language,
        include_unsectioned=include_unsectioned,
    )
    if include_unsectioned:
        validate_sections(text, result)
    else:
        _validate_section_spans(text, result, require_coverage=False)
    return result


def _refine_sections_with_learned(
    text: str,
    rules: tuple[SectionSpan, ...],
    *,
    language: str | None,
    include_unsectioned: bool,
    learned_head: Any | None,
    model_path: str | None,
) -> tuple[SectionSpan, ...]:
    """Add learned boundaries only inside rule gaps or weak rule spans."""

    if rules and not any(
        span.label == UNSECTIONED_SECTION
        or span.confidence < _LEARNED_REFINEMENT_CONFIDENCE
        for span in rules
    ):
        return rules

    from .learned import predict_section_candidates

    candidates = predict_section_candidates(
        text,
        language=language,
        head=learned_head,
        model_path=model_path,
    )
    accepted: list[SectionSpan] = []
    for candidate in candidates:
        start = candidate.get("start")
        end = candidate.get("end")
        label = candidate.get("label")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(label, str)
            or not label
            or start < 0
            or end <= start
            or end > len(text)
        ):
            continue

        containing = next(
            (span for span in rules if span.start <= start < span.end),
            None,
        )
        if containing is not None and (
            containing.label != UNSECTIONED_SECTION
            and containing.confidence >= _LEARNED_REFINEMENT_CONFIDENCE
        ):
            continue

        accepted.append(
            SectionSpan(
                label=label,
                start=start,
                end=end,
                **{
                    key: value
                    for key, value in candidate.items()
                    if key not in {"label", "start", "end"}
                },
            )
        )

    if not accepted:
        return rules

    boundaries: dict[int, SectionSpan] = {
        span.start: span for span in rules if span.label != UNSECTIONED_SECTION
    }
    for candidate in accepted:
        previous = boundaries.get(candidate.start)
        if previous is None or candidate.confidence > previous.confidence:
            boundaries[candidate.start] = candidate

    ordered = tuple(
        sorted(boundaries.values(), key=lambda span: (span.start, -span.confidence))
    )
    result = _assemble_refined_sections(
        text,
        ordered,
        language=language,
        include_unsectioned=include_unsectioned,
    )
    if include_unsectioned:
        validate_sections(text, result)
    else:
        _validate_section_spans(text, result, require_coverage=False)
    return result


def _assemble_refined_sections(
    text: str,
    boundaries: tuple[SectionSpan, ...],
    *,
    language: str | None,
    include_unsectioned: bool,
) -> tuple[SectionSpan, ...]:
    """Partition text at reconciled rule and learned boundary candidates."""

    if not boundaries:
        if not include_unsectioned:
            return ()
        return (
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=0,
                end=len(text),
                language=language,
            ),
        )

    sections: list[SectionSpan] = []
    cursor = 0
    for index, boundary in enumerate(boundaries):
        start = boundary.start
        if include_unsectioned and cursor < start:
            sections.append(
                _section_dict(
                    label=UNSECTIONED_SECTION,
                    start=cursor,
                    end=start,
                    language=language,
                )
            )
        end = boundaries[index + 1].start if index + 1 < len(boundaries) else len(text)
        if start >= end:
            continue
        metadata = {
            key: value
            for key, value in boundary.items()
            if key not in {"label", "start", "end"}
        }
        sections.append(
            SectionSpan(
                label=boundary.label,
                start=start,
                end=end,
                **metadata,
            )
        )
        cursor = end

    if include_unsectioned and cursor < len(text):
        sections.append(
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=cursor,
                end=len(text),
                language=language,
            )
        )
    return tuple(sections)


def _segment_lexicon_sections(
    text: str,
    language: str | None,
) -> tuple[SectionSpan, ...]:
    """Return sections detected from the multilingual header lexicon."""

    if not text:
        return ()

    lines = tuple(_iter_lines(text))
    hits = _dedupe_hits(
        hit
        for index, line in enumerate(lines)
        for hit in _line_header_hits(line, _next_line(lines, index), language)
    )
    if not hits:
        result = (
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=0,
                end=len(text),
                language=language,
            ),
        )
        validate_sections(text, result)
        return result

    sections: list[SectionSpan] = []
    cursor = 0
    for index, hit in enumerate(hits):
        if cursor < hit.start:
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

    if cursor < len(text):
        sections.append(
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=cursor,
                end=len(text),
                language=language,
            )
        )
    result = tuple(section for section in sections if section["start"] < section["end"])
    validate_sections(text, result)
    return result


def _segment_history_sections(
    text: str,
    language: str | None,
) -> tuple[SectionSpan, ...]:
    """Adapt the focused history-family segmenter to the registry contract."""

    del language
    from .history import segment_history_family

    return segment_history_family(text)


_REGISTERED_SECTION_SEGMENTERS: tuple[_SectionSegmenter, ...] = (
    _segment_lexicon_sections,
    _segment_history_sections,
)


def _section_candidates(
    text: str,
    language: str | None,
) -> tuple[_SectionCandidate, ...]:
    candidates: list[_SectionCandidate] = []
    registration_order = 0
    for segmenter in _REGISTERED_SECTION_SEGMENTERS:
        for index, raw_span in enumerate(segmenter(text, language)):
            if not isinstance(raw_span, Mapping):
                raise ValueError(
                    f"registered section segmenter span {index} must be a mapping"
                )
            label = raw_span.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"registered section segmenter span {index} requires a label"
                )
            start = _section_offset(raw_span, "start", index)
            end = _section_offset(raw_span, "end", index)
            if start < 0 or start > len(text) or end < 0 or end > len(text):
                raise ValueError(
                    f"registered section segmenter span {index} has invalid offsets"
                )
            if end <= start:
                raise ValueError(
                    f"registered section segmenter span {index} has invalid offsets"
                )
            if label == UNSECTIONED_SECTION:
                continue

            metadata = {
                key: value
                for key, value in raw_span.items()
                if isinstance(key, str) and key not in {"label", "start", "end"}
            }
            span = SectionSpan(label=label, start=start, end=end, **metadata)
            _ensure_section_metadata(span, confidence=_rule_confidence(span))
            header_start, header_end = _candidate_header_bounds(text, span)
            candidates.append(
                _SectionCandidate(
                    span=span,
                    header_start=header_start,
                    header_end=header_end,
                    registration_order=registration_order,
                )
            )
            registration_order += 1
    return tuple(candidates)


def _candidate_header_bounds(
    text: str,
    span: Mapping[str, Any],
) -> tuple[int, int]:
    start = int(span["start"])
    end = int(span["end"])
    raw_header_start = span.get("header_start")
    raw_header_end = span.get("header_end")
    if raw_header_start is not None or raw_header_end is not None:
        if (
            not isinstance(raw_header_start, int)
            or isinstance(raw_header_start, bool)
            or not isinstance(raw_header_end, int)
            or isinstance(raw_header_end, bool)
            or not start <= raw_header_start < raw_header_end <= end
        ):
            raise ValueError("section candidate has invalid header offsets")
        return raw_header_start, raw_header_end

    line_end = text.find("\n", start, end)
    if line_end == -1:
        line_end = end
    content_end = line_end - int(line_end > start and text[line_end - 1] == "\r")
    content, content_start = _strip_line_prefix(text[start:content_end], start)
    delimiter_index = _first_delimiter_index(content)
    header = (
        content[:delimiter_index].strip() if delimiter_index > 0 else content.strip()
    )
    if not header:
        return start, start + 1
    header_start = content_start + content.find(header)
    return header_start, header_start + len(header)


def _resolve_overlapping_headers(
    candidates: Iterable[_SectionCandidate],
) -> tuple[_SectionCandidate, ...]:
    selected: list[_SectionCandidate] = []
    precedence = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.header_length,
            candidate.span.start,
            candidate.registration_order,
        ),
    )
    for candidate in precedence:
        if any(_candidate_headers_overlap(candidate, other) for other in selected):
            continue
        selected.append(candidate)
    return tuple(
        sorted(
            selected,
            key=lambda candidate: (
                candidate.span.start,
                candidate.registration_order,
            ),
        )
    )


def _candidate_headers_overlap(
    left: _SectionCandidate,
    right: _SectionCandidate,
) -> bool:
    return left.span.start == right.span.start or (
        left.header_start < right.header_end and right.header_start < left.header_end
    )


def _assemble_sections(
    text: str,
    candidates: tuple[_SectionCandidate, ...],
    *,
    language: str | None,
    include_unsectioned: bool,
) -> tuple[SectionSpan, ...]:
    if not candidates:
        if not include_unsectioned:
            return ()
        return (
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=0,
                end=len(text),
                language=language,
            ),
        )

    sections: list[SectionSpan] = []
    cursor = 0
    for index, candidate in enumerate(candidates):
        start = candidate.span.start
        if include_unsectioned and cursor < start:
            sections.append(
                _section_dict(
                    label=UNSECTIONED_SECTION,
                    start=cursor,
                    end=start,
                    language=language,
                )
            )
        next_start = (
            candidates[index + 1].span.start
            if index + 1 < len(candidates)
            else len(text)
        )
        end = min(candidate.span.end, next_start)
        metadata = {
            key: value
            for key, value in candidate.span.items()
            if key not in {"label", "start", "end"}
        }
        sections.append(
            SectionSpan(
                label=candidate.span.label,
                start=start,
                end=end,
                **metadata,
            )
        )
        cursor = end
    if include_unsectioned and cursor < len(text):
        sections.append(
            _section_dict(
                label=UNSECTIONED_SECTION,
                start=cursor,
                end=len(text),
                language=language,
            )
        )
    return tuple(sections)


def validate_sections(
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


def validate_section_spans(
    text: str,
    spans: Iterable[Mapping[str, Any]],
) -> None:
    """Backward-compatible alias for :func:`validate_sections`."""

    validate_sections(text, spans)


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


def section_label_from_loinc(section: str | Mapping[str, Any]) -> str | None:
    """Return a canonical section label from supported LOINC metadata.

    Mapping inputs may carry a LOINC value under ``loinc_code``, ``loinc``,
    ``code``, ``coding``, or ``codes``. Coding mappings are accepted only when
    their optional system identifies LOINC. A bare string is treated as a
    LOINC code, not as a section heading.
    """

    values: Iterable[Any]
    if isinstance(section, str):
        values = (section,)
    else:
        values = (
            section.get(key)
            for key in ("loinc_code", "loinc", "code", "coding", "codes")
        )

    label_by_code = {code: label for label, code in SECTION_LOINC_CODES.items()}
    for value in values:
        for code in _loinc_codes(value):
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
        if start < 0 or start > len(text) or end < 0 or end > len(text):
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
        elif require_coverage and start > previous_end:
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
    loinc_code = section_loinc_code(label)
    if loinc_code is not None:
        section["loinc_code"] = loinc_code
        section["codes"] = section_codes(label)
        section["coding"] = section_codings(label)
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
                "source": "rule",
                "confidence": 0.9,
            }
        )
    elif language:
        section["language"] = language
    return section


def _rule_confidence(span: Mapping[str, Any]) -> float:
    raw_confidence = span.get("confidence")
    if raw_confidence is not None:
        try:
            return min(max(float(raw_confidence), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    if span.get("header_start") is not None:
        return 0.9
    return 0.84


def _ensure_section_metadata(span: SectionSpan, *, confidence: float) -> None:
    """Attach the shared source, confidence, and LOINC metadata contract."""

    span.setdefault("source", "rule")
    span.setdefault("confidence", round(confidence, 6))
    codings = section_codings(span.label)
    if codings:
        span.setdefault("loinc_code", codings[0]["code"])
        span.setdefault("codes", [codings[0]["code"]])
        span.setdefault("coding", codings)


__all__ = [
    "CONTEXT_SECTION_LOINC_CODES",
    "LIST_BEARING_SECTION_LABELS",
    "LIST_BEARING_SECTION_LOINC_CODES",
    "LIST_SECTION_LOINC_CODES",
    "SECTION_LOINC_MAP",
    "SECTION_LOINC_CODES",
    "SectionSpan",
    "UNSECTIONED_SECTION",
    "detect_sections",
    "is_list_bearing_section",
    "list_section_label",
    "parse_section_lists",
    "section_label_from_loinc",
    "validate_section_spans",
]
