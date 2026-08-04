"""Label-by-language PHI leakage heatmaps for evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.metrics import (
    _contains_surface,  # noqa: PLC2701
    _covered_char_count,  # noqa: PLC2701
    _grapheme_count,  # noqa: PLC2701
    _grapheme_overlap_tally,  # noqa: PLC2701
    _iter_extraction_offsets,  # noqa: PLC2701
    _iter_extraction_text_values,  # noqa: PLC2701
    _surface_index,  # noqa: PLC2701
    compute_leakage_rate,
    normalize_eval_spans,
)


@dataclass(frozen=True, init=False)
class LeakageHeatmapCell:
    """Grapheme leakage rate for one canonical label and language cell.

    The ``*_chars`` names remain compatibility aliases for grapheme counts.
    """

    canonical_label: str
    language: str
    leaked_chars: int
    total_chars: int
    rate: float

    def __init__(
        self,
        canonical_label: str | None = None,
        language: str = "",
        leaked_chars: int = 0,
        total_chars: int = 0,
        rate: float = 0.0,
        *,
        label: str | None = None,
    ) -> None:
        resolved_label = canonical_label if canonical_label is not None else label
        if resolved_label is None:
            msg = "canonical_label or label is required"
            raise TypeError(msg)

        object.__setattr__(self, "canonical_label", resolved_label)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "leaked_chars", int(leaked_chars))
        object.__setattr__(self, "total_chars", int(total_chars))
        object.__setattr__(self, "rate", float(rate))

    @property
    def label(self) -> str:
        """Return the canonical label using the interim heatmap API name."""
        return self.canonical_label

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a PHI-free dictionary representation of this cell."""
        return {
            "canonical_label": self.canonical_label,
            "label": self.canonical_label,
            "language": self.language,
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
            "rate": self.rate,
        }

    def __getitem__(self, key: str) -> int | float | str:
        return self.to_dict()[key]


HeatmapCell = LeakageHeatmapCell


@dataclass(frozen=True)
class LeakageHeatmapTotal:
    """Leakage aggregate for one heatmap row or column."""

    leaked_chars: int
    total_chars: int
    rate: float
    canonical_label: str | None = None
    language: str | None = None

    @property
    def label(self) -> str | None:
        """Return the canonical label using the interim heatmap API name."""
        return self.canonical_label

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a PHI-free dictionary representation of this aggregate."""
        payload: dict[str, int | float | str] = {
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
            "rate": self.rate,
        }
        if self.canonical_label is not None:
            payload["canonical_label"] = self.canonical_label
            payload["label"] = self.canonical_label
        if self.language is not None:
            payload["language"] = self.language
        return payload

    def __getitem__(self, key: str) -> int | float | str:
        return self.to_dict()[key]


class _HeatmapCells(dict[str, dict[str, LeakageHeatmapCell]]):
    """Nested cell map with tuple-key lookup compatibility."""

    def __getitem__(
        self, key: str | tuple[str, str]
    ) -> dict[str, LeakageHeatmapCell] | LeakageHeatmapCell:
        if isinstance(key, tuple):
            label, language = key
            return super().__getitem__(label)[language]
        return super().__getitem__(key)

    def get(
        self,
        key: object,
        default: Any = None,
    ) -> dict[str, LeakageHeatmapCell] | LeakageHeatmapCell | Any:
        if isinstance(key, tuple) and len(key) == 2:
            label, language = key
            language_cells = super().get(label)
            if language_cells is None:
                return default
            return language_cells.get(language, default)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            label, language = key
            language_cells = super().get(label)
            return language_cells is not None and language in language_cells
        return super().__contains__(key)


@dataclass(frozen=True)
class LeakageHeatmap:
    """PHI-free leakage matrix keyed by canonical label and language."""

    labels: tuple[str, ...]
    languages: tuple[str, ...]
    cells: Mapping[str, Mapping[str, LeakageHeatmapCell]]
    row_totals: Mapping[str, LeakageHeatmapTotal]
    column_totals: Mapping[str, LeakageHeatmapTotal]
    worst_cells: tuple[LeakageHeatmapCell, ...]
    total: LeakageHeatmapTotal

    @property
    def worst(self) -> list[LeakageHeatmapCell]:
        """Return worst cells using the interim heatmap API name."""
        return list(self.worst_cells)

    @property
    def col_totals(self) -> Mapping[str, LeakageHeatmapTotal]:
        """Return column totals using the interim heatmap API name."""
        return self.column_totals

    def to_dict(self) -> dict[str, Any]:
        """Return labels, languages, rates, and counts without source spans."""
        cell_payload = {
            label: {
                language: cell.to_dict() for language, cell in language_cells.items()
            }
            for label, language_cells in self.cells.items()
        }
        row_payload = {
            label: total.to_dict() for label, total in self.row_totals.items()
        }
        column_payload = {
            language: total.to_dict() for language, total in self.column_totals.items()
        }
        worst_payload = [cell.to_dict() for cell in self.worst_cells]
        return {
            "labels": list(self.labels),
            "languages": list(self.languages),
            "cells": cell_payload,
            "row_totals": row_payload,
            "column_totals": column_payload,
            "col_totals": column_payload,
            "worst_cells": worst_payload,
            "worst": worst_payload,
            "total": self.total.to_dict(),
        }

    def to_markdown(self) -> str:
        """Render a deterministic Markdown matrix with explicit empty cells."""
        header = ["Canonical label", *self.languages, "Total"]
        rows = [
            _markdown_row(header),
            _markdown_row(["---"] * len(header)),
        ]
        for label in self.labels:
            rows.append(
                _markdown_row(
                    [
                        f"`{label}`",
                        *[
                            _format_cell(self.cells[label][language])
                            for language in self.languages
                        ],
                        _format_total(self.row_totals[label]),
                    ]
                )
            )
        rows.append(
            _markdown_row(
                [
                    "**Total**",
                    *[
                        _format_total(self.column_totals[language])
                        for language in self.languages
                    ],
                    _format_total(self.total),
                ]
            )
        )
        return "\n".join(rows)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ScriptLeakageHeatmapCell:
    """PHI-free grapheme leakage counts for one Unicode script."""

    script: str
    leaked_graphemes: int
    total_graphemes: int
    rate: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-compatible script cell."""
        return {
            "script": self.script,
            "leaked_graphemes": self.leaked_graphemes,
            "total_graphemes": self.total_graphemes,
            "rate": self.rate,
        }

    def __getitem__(self, key: str) -> int | float | str:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ScriptLeakageHeatmap:
    """Grapheme leakage heatmap covering every script in the gold spans."""

    scripts: tuple[str, ...]
    cells: Mapping[str, ScriptLeakageHeatmapCell]
    worst_scripts: tuple[ScriptLeakageHeatmapCell, ...]
    total: ScriptLeakageHeatmapCell

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free, JSON-compatible payload."""
        return {
            "scripts": list(self.scripts),
            "cells": {script: self.cells[script].to_dict() for script in self.scripts},
            "worst_scripts": [cell.to_dict() for cell in self.worst_scripts],
            "total": self.total.to_dict(),
            "unit": "grapheme_cluster",
        }

    def to_markdown(self) -> str:
        """Render the per-script leakage heatmap as deterministic Markdown."""
        rows = [
            _markdown_row(["Script", "Leaked graphemes", "Total graphemes", "Rate"]),
            _markdown_row(["---", "---:", "---:", "---:"]),
        ]
        for script in self.scripts:
            cell = self.cells[script]
            rows.append(
                _markdown_row(
                    [
                        f"`{script}`",
                        str(cell.leaked_graphemes),
                        str(cell.total_graphemes),
                        f"{cell.rate:.3f}",
                    ]
                )
            )
        rows.append(
            _markdown_row(
                [
                    "**Total**",
                    str(self.total.leaked_graphemes),
                    str(self.total.total_graphemes),
                    f"{self.total.rate:.3f}",
                ]
            )
        )
        return "\n".join(rows)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def compute_leakage_heatmap(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    worst_n: int = 10,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> LeakageHeatmap:
    """Compute leakage rates for every canonical-label/language cell.

    Args:
        gold_spans: Gold PHI spans in any format accepted by
            ``normalize_eval_spans``.
        predicted_spans: Predicted spans in any format accepted by
            ``normalize_eval_spans``.
        worst_n: Number of non-empty highest-leakage cells to include.
        default_language: Language applied to spans without a language.
        default_device: Device applied to spans without a device.
        source_text: Optional source text used only by span normalization.

    Returns:
        A PHI-free heatmap with cell rates, row/column totals, and worst cells.
    """
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    leaked_counts = [
        max(_grapheme_count(span) - _covered_char_count(span, predicted), 0)
        for span in gold
    ]
    return _heatmap_from_leaked_counts(gold, leaked_counts, worst_n=worst_n)


def compute_script_leakage_heatmap(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    worst_n: int = 10,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> ScriptLeakageHeatmap:
    """Compute grapheme leakage for every Unicode script in the gold set."""
    leakage = compute_leakage_rate(
        gold_spans,
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    scripts = tuple(
        sorted(
            script
            for script, total in leakage.total_chars_by_script.items()
            if total > 0
        )
    )
    cells = {
        script: ScriptLeakageHeatmapCell(
            script=script,
            leaked_graphemes=leakage.leaked_chars_by_script.get(script, 0),
            total_graphemes=leakage.total_chars_by_script[script],
            rate=leakage.by_script[script],
        )
        for script in scripts
    }
    limit = max(int(worst_n), 0)
    worst = tuple(
        sorted(
            cells.values(),
            key=lambda cell: (-cell.rate, cell.script),
        )[:limit]
    )
    return ScriptLeakageHeatmap(
        scripts=scripts,
        cells=cells,
        worst_scripts=worst,
        total=ScriptLeakageHeatmapCell(
            script="all",
            leaked_graphemes=leakage.leaked_graphemes,
            total_graphemes=leakage.total_graphemes,
            rate=leakage.overall,
        ),
    )


def compute_extraction_reemission_heatmap(
    gold_spans: Iterable[Any],
    extraction_outputs: Any,
    *,
    worst_n: int = 10,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> LeakageHeatmap:
    """Compute a PHI-free heatmap for extraction or grounding re-emissions."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    leaked_intervals: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    surfaces = _surface_index(gold)

    for text in _iter_extraction_text_values(extraction_outputs):
        folded = text.casefold()
        for span_index, surface in surfaces:
            if _contains_surface(folded, surface):
                span = gold[span_index]
                leaked_intervals[span_index].append((span.start, span.end))

    for start, end in _iter_extraction_offsets(extraction_outputs):
        for span_index, span in enumerate(gold):
            overlap_start = max(start, span.start)
            overlap_end = min(end, span.end)
            if overlap_start < overlap_end:
                leaked_intervals[span_index].append((overlap_start, overlap_end))

    leaked_counts = [
        _grapheme_overlap_tally(span, leaked_intervals.get(index, ())).matched
        for index, span in enumerate(gold)
    ]
    return _heatmap_from_leaked_counts(gold, leaked_counts, worst_n=worst_n)


def _heatmap_from_leaked_counts(
    gold: Iterable[Any],
    leaked_counts: Iterable[int],
    *,
    worst_n: int,
) -> LeakageHeatmap:
    leaked_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
    total_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
    leaked_by_label: defaultdict[str, int] = defaultdict(int)
    total_by_label: defaultdict[str, int] = defaultdict(int)
    leaked_by_language: defaultdict[str, int] = defaultdict(int)
    total_by_language: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    leaked_chars = 0
    for span, raw_leaked in zip(gold, leaked_counts):
        total = _grapheme_count(span)
        leaked = min(max(int(raw_leaked), 0), total)
        key = (span.label, span.language)

        leaked_by_cell[key] += leaked
        total_by_cell[key] += total
        leaked_by_label[span.label] += leaked
        total_by_label[span.label] += total
        leaked_by_language[span.language] += leaked
        total_by_language[span.language] += total
        leaked_chars += leaked
        total_chars += total

    labels = _ordered_keys(CANONICAL_LABELS, total_by_label, leaked_by_label)
    languages = _ordered_keys(
        SUPPORTED_LANGUAGES,
        total_by_language,
        leaked_by_language,
    )
    cells = _HeatmapCells(
        {
            label: {
                language: _cell(
                    label,
                    language,
                    leaked_by_cell[(label, language)],
                    total_by_cell[(label, language)],
                )
                for language in languages
            }
            for label in labels
        }
    )
    row_totals = {
        label: _total(
            leaked_by_label[label],
            total_by_label[label],
            canonical_label=label,
            language="all",
        )
        for label in labels
    }
    column_totals = {
        language: _total(
            leaked_by_language[language],
            total_by_language[language],
            canonical_label="all",
            language=language,
        )
        for language in languages
    }

    ranked_cells = sorted(
        (
            cell
            for language_cells in cells.values()
            for cell in language_cells.values()
            if cell.total_chars > 0
        ),
        key=lambda cell: (-cell.rate, cell.canonical_label, cell.language),
    )
    limit = max(int(worst_n), 0)

    return LeakageHeatmap(
        labels=tuple(labels),
        languages=tuple(languages),
        cells=cells,
        row_totals=row_totals,
        column_totals=column_totals,
        worst_cells=tuple(ranked_cells[:limit]),
        total=_total(leaked_chars, total_chars),
    )


def render_leakage_heatmap_markdown(heatmap: LeakageHeatmap) -> str:
    """Render a leakage heatmap as a deterministic Markdown matrix."""
    return heatmap.to_markdown()


def render_script_leakage_heatmap_markdown(
    heatmap: ScriptLeakageHeatmap,
) -> str:
    """Render a per-script leakage heatmap as deterministic Markdown."""
    return heatmap.to_markdown()


def _ordered_keys(
    required: Iterable[str],
    *maps: Mapping[str, Any],
) -> tuple[str, ...]:
    keys = set(required)
    for item in maps:
        keys.update(item)
    return tuple(sorted(keys))


def _cell(
    canonical_label: str,
    language: str,
    leaked_chars: int,
    total_chars: int,
) -> LeakageHeatmapCell:
    return LeakageHeatmapCell(
        canonical_label=canonical_label,
        language=language,
        leaked_chars=int(leaked_chars),
        total_chars=int(total_chars),
        rate=_safe_rate(leaked_chars, total_chars),
    )


def _total(
    leaked_chars: int,
    total_chars: int,
    *,
    canonical_label: str | None = None,
    language: str | None = None,
) -> LeakageHeatmapTotal:
    return LeakageHeatmapTotal(
        canonical_label=canonical_label,
        language=language,
        leaked_chars=int(leaked_chars),
        total_chars=int(total_chars),
        rate=_safe_rate(leaked_chars, total_chars),
    )


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_cell(cell: LeakageHeatmapCell) -> str:
    return f"{cell.leaked_chars}/{cell.total_chars} ({cell.rate:.3f})"


def _format_total(total: LeakageHeatmapTotal) -> str:
    return f"{total.leaked_chars}/{total.total_chars} ({total.rate:.3f})"


def _markdown_row(values: Iterable[str]) -> str:
    return "| " + " | ".join(values) + " |"
