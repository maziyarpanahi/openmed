"""Deterministic laboratory-result structuring from clinical concept spans."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.lab_values import (
    AbnormalFlag,
    ReferenceRange,
    derive_abnormal_flag,
    parse_reference_range,
)
from openmed.clinical.sections import detect_sections, validate_section_spans
from openmed.clinical.units import parse_measurement
from openmed.core.labels import LAB_TEST, LAB_VALUE, UNIT, normalize_label
from openmed.processing.advanced_ner import EntitySpan

from .candidate import SpanReference

LAB_RESULT_ADVISORY = (
    "Structured laboratory results are deterministic assistive output and are not "
    "a substitute for the originating laboratory report or clinician review."
)

_MAX_ANALYTE_DISTANCE = 80
_NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RANGE = rf"(?:{_NUMERIC}\s*(?:-|to|\u2013|\u2014)\s*{_NUMERIC}|(?:<=|>=|<|>|\u2264|\u2265)\s*{_NUMERIC})"
_UNIT = (
    r"(?:%|10(?:\^|\*\*)?\d+/[A-Za-z\u00b5\u03bc]+|"
    r"[A-Za-z\u00b5\u03bc\u00b0][A-Za-z0-9\u00b5\u03bc\u00b0%]*"
    r"(?:[./^*\u00b7-][A-Za-z0-9\u00b5\u03bc\u00b0%]+)*)"
)
_FLAG = r"(?:critical(?:\s+(?:high|low))?|crit|normal|high|low|HH|LL|H|L|C|N)"
_MEASUREMENT_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMERIC})(?![\w.])"
    rf"[ \t]*(?P<unit>{_UNIT})"
    rf"(?:[ \t]*\([ \t]*(?P<range>{_RANGE}(?:[ \t]+{_UNIT})?)[ \t]*\))?"
    rf"(?:[ \t]+(?P<flag>{_FLAG})(?![A-Za-z]))?",
    re.IGNORECASE,
)
_NUMERIC_ONLY_RE = re.compile(rf"^{_NUMERIC}$")
_LOCAL_BOUNDARY_RE = re.compile(r"[.;!?\n\r]")


@dataclass(frozen=True)
class LabResult:
    """One analyte-bound, non-FHIR laboratory result.

    Character offsets and source text are retained only for the analyte span;
    parsed attributes are normalized values derived from the same local text.
    """

    analyte: SpanReference
    value: float
    unit: str
    reference_range: ReferenceRange | None
    abnormal_flag: AbnormalFlag
    score: float
    advisory: str = LAB_RESULT_ADVISORY

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready representation."""

        return {
            "analyte": self.analyte.to_dict(),
            "value": self.value,
            "unit": self.unit,
            "reference_range": (
                dict(self.reference_range) if self.reference_range is not None else None
            ),
            "abnormal_flag": self.abnormal_flag,
            "score": self.score,
            "advisory": self.advisory,
        }


@dataclass(frozen=True)
class _Measurement:
    value: float
    unit: str
    reference_range: ReferenceRange | None
    explicit_flag: str | None
    start: int
    end: int
    unit_start: int
    unit_end: int


def extract_lab_results(
    text: str,
    spans: Iterable[EntitySpan | SpanReference | Mapping[str, Any]],
    sections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[LabResult, ...]:
    """Extract structured laboratory results near existing analyte spans.

    Measurements must contain a numeric value and unit. An optional
    parenthesized reference range and explicit ``H``/``L``/critical marker are
    parsed from the same local expression. When no explicit marker is present,
    the abnormal flag is derived from the parsed range. Each measurement is
    linked to at most one nearest analyte without crossing a section or local
    sentence/line boundary.

    Args:
        text: Original clinical text.
        spans: Existing clinical concept spans with source character offsets.
            Analytes normally use ``LAB_TEST`` (or an analyte/test alias).
            A non-numeric ``LAB_VALUE`` span is also accepted as a legacy head.
        sections: Optional precomputed contiguous section spans. When omitted,
            sections are detected locally and deterministically.

    Returns:
        Deterministically ordered analyte-bound laboratory results. A numeric
        expression without an analyte span is never emitted.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    span_items = tuple(spans)
    section_items = tuple(detect_sections(text) if sections is None else sections)
    validate_section_spans(text, section_items)
    references = _coerce_spans(text, span_items, section_items)
    analytes = tuple(reference for reference in references if _is_analyte(reference))
    if not analytes:
        return ()

    measurements = tuple(
        measurement
        for match in _MEASUREMENT_RE.finditer(text)
        if _valid_measurement(
            measurement := _measurement(match),
            references,
        )
    )
    if not measurements:
        return ()

    results = [
        _lab_result(analyte, measurement)
        for analyte, measurement in _nearest_pairs(
            text,
            analytes,
            measurements,
            section_items,
        )
    ]
    return tuple(sorted(results, key=lambda result: result.analyte.offset_key()))


def _measurement(match: re.Match[str]) -> _Measurement:
    raw_range = match.group("range")
    reference_range = parse_reference_range(raw_range) if raw_range else None
    if reference_range is not None and not any(
        reference_range[bound] is not None for bound in ("low", "high")
    ):
        reference_range = None
    return _Measurement(
        value=float(match.group("value")),
        unit=match.group("unit"),
        reference_range=reference_range,
        explicit_flag=match.group("flag"),
        start=match.start(),
        end=match.end(),
        unit_start=match.start("unit"),
        unit_end=match.end("unit"),
    )


def _valid_measurement(
    measurement: _Measurement,
    references: Sequence[SpanReference],
) -> bool:
    if parse_measurement(measurement.value, measurement.unit)["status"] == "ok":
        return True
    return any(
        normalize_label(reference.label) == UNIT
        and _spans_overlap(
            reference.start,
            reference.end,
            measurement.unit_start,
            measurement.unit_end,
        )
        for reference in references
    )


def _lab_result(analyte: SpanReference, measurement: _Measurement) -> LabResult:
    range_unit = (
        measurement.reference_range.get("unit")
        if measurement.reference_range is not None
        else None
    )
    abnormal_flag = derive_abnormal_flag(
        measurement.value,
        measurement.reference_range,
        measurement.explicit_flag,
        value_unit=measurement.unit if range_unit is not None else None,
    )
    distance = _span_gap(
        analyte.start,
        analyte.end,
        measurement.start,
        measurement.end,
    )
    proximity = max(0.0, 1.0 - (distance / (_MAX_ANALYTE_DISTANCE + 1)))
    analyte_score = analyte.score if math.isfinite(analyte.score) else 0.0
    score = (0.75 * min(1.0, max(0.0, analyte_score))) + (0.25 * proximity)
    return LabResult(
        analyte=analyte,
        value=measurement.value,
        unit=measurement.unit,
        reference_range=measurement.reference_range,
        abnormal_flag=abnormal_flag,
        score=round(score, 6),
    )


def _nearest_pairs(
    text: str,
    analytes: Sequence[SpanReference],
    measurements: Sequence[_Measurement],
    sections: Sequence[Mapping[str, Any]],
) -> tuple[tuple[SpanReference, _Measurement], ...]:
    candidates: list[tuple[int, int, int, int, int, SpanReference, _Measurement]] = []
    for analyte in analytes:
        analyte_section = _section_index(sections, analyte.start, analyte.end)
        for measurement in measurements:
            measurement_section = _section_index(
                sections,
                measurement.start,
                measurement.end,
            )
            if analyte_section is None or analyte_section != measurement_section:
                continue
            if _spans_overlap(
                analyte.start,
                analyte.end,
                measurement.start,
                measurement.end,
            ):
                continue
            distance = _span_gap(
                analyte.start,
                analyte.end,
                measurement.start,
                measurement.end,
            )
            if distance > _MAX_ANALYTE_DISTANCE:
                continue
            between = text[
                min(analyte.end, measurement.end) : max(
                    analyte.start,
                    measurement.start,
                )
            ]
            if _LOCAL_BOUNDARY_RE.search(between):
                continue
            candidates.append(
                (
                    distance,
                    0 if measurement.start >= analyte.end else 1,
                    analyte.start,
                    measurement.start,
                    measurement.end,
                    analyte,
                    measurement,
                )
            )

    selected: list[tuple[SpanReference, _Measurement]] = []
    used_analytes: set[tuple[int, int]] = set()
    used_measurements: set[tuple[int, int]] = set()
    for *_, analyte, measurement in sorted(candidates, key=lambda item: item[:5]):
        analyte_key = analyte.offset_key()
        measurement_key = measurement.start, measurement.end
        if analyte_key in used_analytes or measurement_key in used_measurements:
            continue
        selected.append((analyte, measurement))
        used_analytes.add(analyte_key)
        used_measurements.add(measurement_key)
    return tuple(selected)


def _coerce_spans(
    text: str,
    spans: Sequence[EntitySpan | SpanReference | Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
) -> tuple[SpanReference, ...]:
    references: dict[tuple[int, int, str], SpanReference] = {}
    for item in spans:
        if isinstance(item, SpanReference):
            start, end, label, score = (
                item.start,
                item.end,
                item.label,
                item.score,
            )
        elif isinstance(item, EntitySpan):
            start, end, label, score = item.start, item.end, item.label, item.score
        elif isinstance(item, Mapping):
            try:
                start = int(item.get("start", item.get("start_char", -1)))
                end = int(item.get("end", item.get("end_char", -1)))
                label = str(item.get("label", item.get("entity", "")))
                score = float(item.get("score", 1.0))
            except (TypeError, ValueError):
                continue
        else:
            continue
        if not label or start < 0 or end <= start or end > len(text):
            continue
        section_index = _section_index(sections, start, end)
        section = (
            str(sections[section_index]["label"]) if section_index is not None else None
        )
        reference = SpanReference(
            text=text[start:end],
            label=label,
            start=start,
            end=end,
            score=score,
            section=section,
        )
        references[(start, end, normalize_label(label))] = reference
    return tuple(
        sorted(
            references.values(),
            key=lambda reference: (
                reference.start,
                reference.end,
                normalize_label(reference.label),
            ),
        )
    )


def _is_analyte(span: SpanReference) -> bool:
    canonical = normalize_label(span.label)
    normalized = re.sub(r"[^a-z0-9]+", "_", span.label.casefold()).strip("_")
    if canonical == LAB_TEST or normalized in {"analyte", "lab_name"}:
        return True
    return canonical == LAB_VALUE and not _NUMERIC_ONLY_RE.fullmatch(span.text.strip())


def _section_index(
    sections: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> int | None:
    return next(
        (
            index
            for index, section in enumerate(sections)
            if int(section["start"]) <= start and end <= int(section["end"])
        ),
        None,
    )


def _spans_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start < right_end and right_start < left_end


def _span_gap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    if left_end <= right_start:
        return right_start - left_end
    if right_end <= left_start:
        return left_start - right_end
    return 0


__all__ = [
    "LAB_RESULT_ADVISORY",
    "LabResult",
    "extract_lab_results",
]
