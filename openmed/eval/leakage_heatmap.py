"""Two-dimensional leakage heatmap: leakage rate by (label, language) cell."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.metrics import (
    EvalSpan,
    _covered_char_count,  # noqa: PLC2701
    _safe_rate,  # noqa: PLC2701
    normalize_eval_spans,
)


@dataclass(frozen=True)
class HeatmapCell:
    """Leakage rate for a single (label, language) pair."""

    label: str
    language: str
    rate: float
    leaked_chars: int
    total_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "language": self.language,
            "rate": self.rate,
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
        }


@dataclass(frozen=True)
class LeakageHeatmap:
    """Leakage rates for every (canonical_label, language) cell.

    ``cells`` is keyed by ``(label, language)`` and contains only cells with
    at least one gold character. ``worst`` lists up to *worst_n* cells sorted
    by descending rate, with ties broken by ``(label, language)`` lexicographic
    order for determinism.

    ``row_totals`` aggregates each label across all languages (language="all").
    ``col_totals`` aggregates each language across all labels (label="all").
    """

    cells: dict[tuple[str, str], HeatmapCell]
    worst: list[HeatmapCell]
    labels: list[str]
    languages: list[str]
    row_totals: dict[str, HeatmapCell]
    col_totals: dict[str, HeatmapCell]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": {
                f"{label}|{lang}": cell.to_dict()
                for (label, lang), cell in self.cells.items()
            },
            "worst": [cell.to_dict() for cell in self.worst],
            "labels": self.labels,
            "languages": self.languages,
            "row_totals": {k: v.to_dict() for k, v in self.row_totals.items()},
            "col_totals": {k: v.to_dict() for k, v in self.col_totals.items()},
        }

    def to_markdown(self) -> str:
        """Render a Markdown matrix: labels as rows, languages as columns."""
        langs = self.languages
        header = "| label \\ lang | " + " | ".join(langs) + " | **Total** |"
        sep = "| --- |" + " --- |" * len(langs) + " --- |"
        rows = [header, sep]
        for label in self.labels:
            parts = [label]
            for lang in langs:
                cell = self.cells.get((label, lang))
                if cell is None or cell.total_chars == 0:
                    parts.append("—")
                else:
                    parts.append(f"{cell.rate:.1%}")
            row_total = self.row_totals.get(label)
            parts.append(f"**{row_total.rate:.1%}**" if row_total else "—")
            rows.append("| " + " | ".join(parts) + " |")
        col_parts = ["**Total**"]
        for lang in langs:
            ct = self.col_totals.get(lang)
            col_parts.append(f"**{ct.rate:.1%}**" if ct else "—")
        col_parts.append("")
        rows.append("| " + " | ".join(col_parts) + " |")
        return "\n".join(rows)


def compute_leakage_heatmap(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
    worst_n: int = 5,
) -> LeakageHeatmap:
    """Compute character-weighted leakage for each (label, language) cell.

    Reuses the same leakage definition as ``compute_leakage_rate``: a gold
    character is leaked when no same-label prediction covers it.
    """
    gold: list[EvalSpan] = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted: list[EvalSpan] = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )

    leaked: defaultdict[tuple[str, str], int] = defaultdict(int)
    total: defaultdict[tuple[str, str], int] = defaultdict(int)

    for span in gold:
        key = (span.label, span.language)
        covered = _covered_char_count(span, predicted)
        total[key] += span.length
        leaked[key] += max(span.length - covered, 0)

    seen_labels: set[str] = set()
    seen_langs: set[str] = set()
    for label, lang in total:
        seen_labels.add(label)
        seen_langs.add(lang)

    label_order = [l for l in CANONICAL_LABELS if l in seen_labels] + sorted(
        seen_labels - set(CANONICAL_LABELS)
    )
    lang_order = [l for l in SUPPORTED_LANGUAGES if l in seen_langs] + sorted(
        seen_langs - set(SUPPORTED_LANGUAGES)
    )

    cells: dict[tuple[str, str], HeatmapCell] = {}
    for key, total_chars in total.items():
        label, lang = key
        leaked_chars = leaked[key]
        cells[key] = HeatmapCell(
            label=label,
            language=lang,
            rate=_safe_rate(leaked_chars, total_chars, zero_denominator=0.0),
            leaked_chars=leaked_chars,
            total_chars=total_chars,
        )

    row_leaked: defaultdict[str, int] = defaultdict(int)
    row_total: defaultdict[str, int] = defaultdict(int)
    col_leaked: defaultdict[str, int] = defaultdict(int)
    col_total: defaultdict[str, int] = defaultdict(int)
    for (label, lang), cell in cells.items():
        row_leaked[label] += cell.leaked_chars
        row_total[label] += cell.total_chars
        col_leaked[lang] += cell.leaked_chars
        col_total[lang] += cell.total_chars

    row_totals = {
        label: HeatmapCell(
            label=label,
            language="all",
            rate=_safe_rate(row_leaked[label], row_total[label], zero_denominator=0.0),
            leaked_chars=row_leaked[label],
            total_chars=row_total[label],
        )
        for label in label_order
        if row_total[label] > 0
    }
    col_totals = {
        lang: HeatmapCell(
            label="all",
            language=lang,
            rate=_safe_rate(col_leaked[lang], col_total[lang], zero_denominator=0.0),
            leaked_chars=col_leaked[lang],
            total_chars=col_total[lang],
        )
        for lang in lang_order
        if col_total[lang] > 0
    }

    worst = sorted(
        cells.values(),
        key=lambda c: (-c.rate, c.label, c.language),
    )[:worst_n]

    return LeakageHeatmap(
        cells=cells,
        worst=worst,
        labels=label_order,
        languages=lang_order,
        row_totals=row_totals,
        col_totals=col_totals,
    )
