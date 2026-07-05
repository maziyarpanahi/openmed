"""Dependency-light notebook rendering for grounded clinical spans."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_LABEL_COLORS = {
    "condition": "#d7f0ff",
    "finding": "#fce7b2",
    "medication": "#dcfce7",
    "procedure": "#ede9fe",
    "lab": "#fbcfe8",
}
_DEFAULT_COLOR = "#e5e7eb"


@dataclass(frozen=True)
class ClinicalExtractionWidget:
    """Small HTML display object for grounded spans in one note."""

    text: str
    rows: tuple[Mapping[str, Any], ...]
    title: str = "OpenMed clinical extraction"

    def _repr_html_(self) -> str:
        """Return notebook-renderable HTML."""

        return render_span_html(self.text, self.rows, title=self.title)

    def to_html(self) -> str:
        """Return the same deterministic HTML used by notebook display."""

        return self._repr_html_()


def render_span_html(
    text: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str = "OpenMed clinical extraction",
) -> str:
    """Render ``text`` with highlighted grounded spans.

    Rows are expected to follow ``FLAT_TABLE_COLUMNS``. Each rendered span
    carries ``data-start``, ``data-end``, ``data-label``, and ``data-code`` so
    tests and lightweight notebook consumers can inspect the DOM without a JS
    dependency.
    """

    safe_rows = _valid_rows(tuple(rows), len(text))
    chunks: list[str] = []
    cursor = 0

    for row in safe_rows:
        start = int(row["start"])
        end = int(row["end"])
        label = str(row.get("entity_label") or "")
        code = str(row.get("code") or "")
        display = str(row.get("display") or row.get("normalized_text") or "")
        color = _LABEL_COLORS.get(label.lower(), _DEFAULT_COLOR)

        chunks.append(html.escape(text[cursor:start]))
        chunks.append(
            '<mark class="openmed-span" '
            f'data-start="{start}" data-end="{end}" '
            f'data-label="{html.escape(label, quote=True)}" '
            f'data-code="{html.escape(code, quote=True)}" '
            f'title="{html.escape(_tooltip(label, code, display), quote=True)}" '
            f'style="background:{color};padding:0 0.15rem;border-radius:3px;">'
            f"{html.escape(text[start:end])}"
            "</mark>"
        )
        cursor = end

    chunks.append(html.escape(text[cursor:]))
    legend = _legend(safe_rows)
    return (
        '<div class="openmed-clinical-widget">'
        "<style>"
        ".openmed-clinical-widget{font-family:system-ui,-apple-system,Segoe UI,"
        "sans-serif;line-height:1.5;color:#111827}"
        ".openmed-clinical-widget pre{white-space:pre-wrap;font:inherit;"
        "margin:0.5rem 0 0}"
        ".openmed-clinical-widget .openmed-title{font-weight:600;margin:0 0 "
        "0.35rem}"
        ".openmed-clinical-widget .openmed-legend{display:flex;gap:0.5rem;"
        "flex-wrap:wrap;font-size:0.85rem;color:#374151}"
        ".openmed-clinical-widget .openmed-chip{display:inline-flex;gap:0.25rem;"
        "align-items:center}"
        "</style>"
        f'<div class="openmed-title">{html.escape(title)}</div>'
        f"{legend}"
        f"<pre>{''.join(chunks)}</pre>"
        "</div>"
    )


def _valid_rows(
    rows: tuple[Mapping[str, Any], ...],
    text_length: int,
) -> tuple[Mapping[str, Any], ...]:
    valid: list[Mapping[str, Any]] = []
    occupied_end = -1
    for row in sorted(
        rows, key=lambda item: (_offset(item.get("start")), _offset(item.get("end")))
    ):
        start = _offset(row.get("start"))
        end = _offset(row.get("end"))
        if start < 0 or end <= start or end > text_length or start < occupied_end:
            continue
        valid.append(row)
        occupied_end = end
    return tuple(valid)


def _offset(value: Any) -> int:
    if isinstance(value, bool) or value in (None, ""):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _tooltip(label: str, code: str, display: str) -> str:
    parts = [part for part in (label, code, display) if part]
    return " | ".join(parts)


def _legend(rows: tuple[Mapping[str, Any], ...]) -> str:
    labels = []
    seen: set[str] = set()
    for row in rows:
        label = str(row.get("entity_label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        color = _LABEL_COLORS.get(label.lower(), _DEFAULT_COLOR)
        labels.append(
            '<span class="openmed-chip">'
            f'<span style="background:{color};width:0.75rem;height:0.75rem;'
            'display:inline-block;border-radius:999px"></span>'
            f"{html.escape(label)}"
            "</span>"
        )
    if not labels:
        return ""
    return f'<div class="openmed-legend">{"".join(labels)}</div>'


__all__ = ["ClinicalExtractionWidget", "render_span_html"]
