"""Label-by-language PHI leakage heatmaps for evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.metrics import (
    _covered_char_count,  # noqa: PLC2701
    normalize_eval_spans,
)


@dataclass(frozen=True, init=False)
class LeakageHeatmapCell:
    """Leakage rate and counts for one canonical label and language cell."""

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

    leaked_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
    total_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
    leaked_by_label: defaultdict[str, int] = defaultdict(int)
    total_by_label: defaultdict[str, int] = defaultdict(int)
    leaked_by_language: defaultdict[str, int] = defaultdict(int)
    total_by_language: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    leaked_chars = 0
    for span in gold:
        covered = _covered_char_count(span, predicted)
        leaked = max(span.length - covered, 0)
        key = (span.label, span.language)

        leaked_by_cell[key] += leaked
        total_by_cell[key] += span.length
        leaked_by_label[span.label] += leaked
        total_by_label[span.label] += span.length
        leaked_by_language[span.language] += leaked
        total_by_language[span.language] += span.length
        leaked_chars += leaked
        total_chars += span.length

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
