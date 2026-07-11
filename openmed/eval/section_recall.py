"""Section-level character recall for clinical note de-identification."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.eval.metrics import (
    EvalSpan,
    RateMetric,
    compute_character_recall,
    normalize_eval_span,
    normalize_eval_spans,
)

UNSECTIONED_SECTION = "unsectioned"
_SECTION_BOUNDARY_NAME_KEYS = ("section", "section_name", "name", "title", "label")
_SECTION_TAG_KEYS = ("section", "section_name", "clinical_section")


@dataclass(frozen=True)
class SectionSpan:
    """A named character range for a clinical note section."""

    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the non-negative section length."""
        return max(self.end - self.start, 0)


@dataclass(frozen=True)
class SectionRecallMetrics:
    """Character recall counts for one clinical note section."""

    recall: float
    covered_chars: int
    total_chars: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a deterministic JSON-ready metric payload."""
        return {
            "covered_chars": self.covered_chars,
            "recall": self.recall,
            "total_chars": self.total_chars,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class SectionDetectionMetrics:
    """Boundary and label metrics for detected clinical sections."""

    boundary_recall: float
    label_precision: float
    label_recall: float
    label_f1: float
    gold_sections: int
    predicted_sections: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a deterministic JSON-ready metric payload."""

        return {
            "boundary_recall": self.boundary_recall,
            "gold_sections": self.gold_sections,
            "label_f1": self.label_f1,
            "label_precision": self.label_precision,
            "label_recall": self.label_recall,
            "predicted_sections": self.predicted_sections,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class SectionRecallReport:
    """Serializable per-section character recall report."""

    per_section: Mapping[str, SectionRecallMetrics]
    overall: SectionRecallMetrics
    worst_sections: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report payload."""
        return {
            "overall": self.overall.to_dict(),
            "per_section": {
                section: metrics.to_dict()
                for section, metrics in sorted(self.per_section.items())
            },
            "worst_sections": list(self.worst_sections),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to a deterministic Markdown table."""
        lines = [
            "| Section | Covered Chars | Total Chars | Recall |",
            "|---|---:|---:|---:|",
        ]
        for section, metrics in _rank_sections(self.per_section):
            lines.append(
                "| "
                f"{_markdown_cell(section)} | "
                f"{metrics.covered_chars} | "
                f"{metrics.total_chars} | "
                f"{_format_rate(metrics.recall)} |"
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def compute_section_recall(
    note: str,
    section_spans: Iterable[Any] | None,
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    unsectioned_name: str = UNSECTIONED_SECTION,
) -> SectionRecallReport:
    """Compute label-aware character recall by clinical note section.

    Args:
        note: Source clinical note text. The text is used only to normalize
            span snippets for the shared eval metric helpers.
        section_spans: Explicit section boundaries as ``SectionSpan`` values,
            mappings, objects with ``name``/``start``/``end`` attributes, or
            three-item tuples. When ``None``, section names are read from gold
            span metadata/top-level fields and untagged spans are assigned to
            ``unsectioned_name``.
        gold_spans: Gold PHI spans.
        predicted_spans: Predicted PHI spans.
        default_language: Language passed through to eval span normalization.
        default_device: Device passed through to eval span normalization.
        unsectioned_name: Bucket for gold characters outside all declared
            sections.

    Returns:
        A report containing only section names and aggregate counts/rates.
    """
    if not unsectioned_name.strip():
        raise ValueError("unsectioned_name must be non-empty")

    raw_gold = list(gold_spans)
    gold = [
        normalize_eval_span(
            span,
            default_language=default_language,
            default_device=default_device,
            source_text=note,
        )
        for span in raw_gold
    ]
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=note,
    )

    if section_spans is None:
        section_gold = _gold_by_section_tags(
            raw_gold,
            gold,
            unsectioned_name=unsectioned_name,
        )
        section_names = sorted(section_gold) or [unsectioned_name]
    else:
        sections = _normalize_sections(section_spans, note_length=len(note))
        section_gold = _gold_by_section_ranges(
            gold,
            sections,
            unsectioned_name=unsectioned_name,
        )
        section_names = sorted(
            {section.name for section in sections}
            | set(section_gold)
            | {unsectioned_name}
        )

    per_section = {
        section: _section_metrics(
            section_gold.get(section, []),
            predicted,
            note=note,
            default_language=default_language,
            default_device=default_device,
        )
        for section in section_names
    }
    overall = _metric_to_section_metrics(
        compute_character_recall(
            gold,
            predicted,
            default_language=default_language,
            default_device=default_device,
            source_text=note,
        )
    )
    worst_sections = _worst_sections(per_section)

    return SectionRecallReport(
        per_section=per_section,
        overall=overall,
        worst_sections=worst_sections,
    )


def compute_section_detection_metrics(
    note: str,
    gold_sections: Iterable[Any],
    predicted_sections: Iterable[Any],
) -> SectionDetectionMetrics:
    """Score exact section boundary recall and canonical label F1."""

    gold = _normalize_sections(gold_sections, note_length=len(note))
    predicted = _normalize_sections(predicted_sections, note_length=len(note))

    gold_boundaries = {(section.start, section.end) for section in gold}
    predicted_boundaries = {(section.start, section.end) for section in predicted}
    boundary_true_positive = len(gold_boundaries & predicted_boundaries)
    boundary_recall = _safe_rate(boundary_true_positive, len(gold_boundaries))

    gold_labeled = {(section.name, section.start, section.end) for section in gold}
    predicted_labeled = {
        (section.name, section.start, section.end) for section in predicted
    }
    label_true_positive = len(gold_labeled & predicted_labeled)
    label_precision = _safe_rate(label_true_positive, len(predicted_labeled))
    label_recall = _safe_rate(label_true_positive, len(gold_labeled))
    label_f1 = _f1(label_precision, label_recall)

    return SectionDetectionMetrics(
        boundary_recall=boundary_recall,
        label_precision=label_precision,
        label_recall=label_recall,
        label_f1=label_f1,
        gold_sections=len(gold),
        predicted_sections=len(predicted),
    )


def _section_metrics(
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
    *,
    note: str,
    default_language: str,
    default_device: str,
) -> SectionRecallMetrics:
    recall = compute_character_recall(
        gold_spans,
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=note,
    )
    return _metric_to_section_metrics(recall)


def _metric_to_section_metrics(metric: RateMetric) -> SectionRecallMetrics:
    return SectionRecallMetrics(
        recall=float(metric.rate),
        covered_chars=int(metric.numerator),
        total_chars=int(metric.denominator),
    )


def _normalize_sections(
    section_spans: Iterable[Any],
    *,
    note_length: int,
) -> list[SectionSpan]:
    sections = [
        _normalize_section(span, note_length=note_length) for span in section_spans
    ]
    sections.sort(key=lambda span: (span.start, span.end, span.name))
    for previous, current in zip(sections, sections[1:]):
        if previous.end > current.start:
            raise ValueError(
                "section_spans must not overlap: "
                f"{previous.name!r} overlaps {current.name!r}"
            )
    return sections


def _normalize_section(raw: Any, *, note_length: int) -> SectionSpan:
    if isinstance(raw, SectionSpan):
        section = raw
    elif isinstance(raw, Mapping):
        name = _first_present(raw, _SECTION_BOUNDARY_NAME_KEYS)
        section = SectionSpan(
            name=_coerce_section_name(name),
            start=_coerce_int(raw.get("start"), field="start"),
            end=_coerce_int(raw.get("end"), field="end"),
        )
    elif _is_three_item_sequence(raw):
        section = _section_from_sequence(raw)
    else:
        name = _first_attr(raw, _SECTION_BOUNDARY_NAME_KEYS)
        section = SectionSpan(
            name=_coerce_section_name(name),
            start=_coerce_int(getattr(raw, "start", None), field="start"),
            end=_coerce_int(getattr(raw, "end", None), field="end"),
        )

    if section.start < 0:
        raise ValueError("section start must be non-negative")
    if section.end < section.start:
        raise ValueError("section end must be greater than or equal to start")
    if section.end > note_length:
        raise ValueError("section end must not exceed note length")
    return section


def _section_from_sequence(raw: Sequence[Any]) -> SectionSpan:
    first, second, third = raw
    if isinstance(first, str):
        name, start, end = first, second, third
    elif isinstance(third, str):
        start, end, name = first, second, third
    else:
        raise ValueError(
            "section tuple must be (name, start, end) or (start, end, name)"
        )
    return SectionSpan(
        name=_coerce_section_name(name),
        start=_coerce_int(start, field="start"),
        end=_coerce_int(end, field="end"),
    )


def _gold_by_section_ranges(
    gold_spans: Sequence[EvalSpan],
    sections: Sequence[SectionSpan],
    *,
    unsectioned_name: str,
) -> dict[str, list[EvalSpan]]:
    by_section: defaultdict[str, list[EvalSpan]] = defaultdict(list)
    for span in gold_spans:
        cursor = span.start
        for section in sections:
            if section.end <= cursor:
                continue
            if section.start >= span.end:
                break
            if cursor < section.start:
                gap_end = min(section.start, span.end)
                _append_fragment(by_section, unsectioned_name, span, cursor, gap_end)
                cursor = gap_end
            overlap_start = max(cursor, section.start)
            overlap_end = min(span.end, section.end)
            _append_fragment(by_section, section.name, span, overlap_start, overlap_end)
            cursor = max(cursor, overlap_end)
        _append_fragment(by_section, unsectioned_name, span, cursor, span.end)
    return dict(by_section)


def _gold_by_section_tags(
    raw_gold: Sequence[Any],
    gold_spans: Sequence[EvalSpan],
    *,
    unsectioned_name: str,
) -> dict[str, list[EvalSpan]]:
    by_section: defaultdict[str, list[EvalSpan]] = defaultdict(list)
    for raw, span in zip(raw_gold, gold_spans):
        section = _section_name_from_span(raw, span) or unsectioned_name
        by_section[section].append(span)
    return dict(by_section)


def _append_fragment(
    by_section: defaultdict[str, list[EvalSpan]],
    section: str,
    span: EvalSpan,
    start: int,
    end: int,
) -> None:
    if start >= end:
        return
    by_section[section].append(
        EvalSpan(
            start=start,
            end=end,
            label=span.label,
            text=span.text,
            language=span.language,
            device=span.device,
            metadata=span.metadata,
        )
    )


def _section_name_from_span(raw: Any, span: EvalSpan) -> str | None:
    if isinstance(raw, Mapping):
        section = _first_present(raw, _SECTION_TAG_KEYS)
        if section is not None:
            return _coerce_section_name(section)
    else:
        section = _first_attr(raw, _SECTION_TAG_KEYS)
        if section is not None:
            return _coerce_section_name(section)
    metadata_section = _first_present(span.metadata, _SECTION_TAG_KEYS)
    if metadata_section is None:
        return None
    return _coerce_section_name(metadata_section)


def _worst_sections(
    per_section: Mapping[str, SectionRecallMetrics],
) -> tuple[str, ...]:
    populated = {
        section: metrics
        for section, metrics in per_section.items()
        if metrics.total_chars > 0
    }
    if not populated:
        return ()
    worst_recall = min(metrics.recall for metrics in populated.values())
    return tuple(
        section
        for section, metrics in sorted(populated.items())
        if metrics.recall == worst_recall
    )


def _rank_sections(
    per_section: Mapping[str, SectionRecallMetrics],
) -> list[tuple[str, SectionRecallMetrics]]:
    return sorted(
        per_section.items(),
        key=lambda item: (item[1].recall, item[0]),
    )


def _is_three_item_sequence(raw: Any) -> bool:
    return (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes, bytearray))
        and len(raw) == 3
    )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _first_attr(raw: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        value = getattr(raw, key, None)
        if value is not None:
            return value
    return None


def _coerce_section_name(value: Any) -> str:
    if value is None:
        raise ValueError("section must include a name")
    name = str(value).strip()
    if not name:
        raise ValueError("section name must be non-empty")
    return name


def _coerce_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"section {field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"section {field} must be an integer") from exc


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def _format_rate(value: float) -> str:
    return f"{value:.6f}"


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


__all__ = [
    "UNSECTIONED_SECTION",
    "SectionDetectionMetrics",
    "SectionRecallMetrics",
    "SectionRecallReport",
    "SectionSpan",
    "compute_section_detection_metrics",
    "compute_section_recall",
]
