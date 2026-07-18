"""Self-contained HTML rendering for re-identification risk reports."""

from __future__ import annotations

import html as html_mod
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["render_risk_dashboard", "write_risk_dashboard"]

_DEFAULT_TITLE = "OpenMed Risk Dashboard"

_CSS = """
:root {
  color-scheme: light;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  background: #f7f8fa;
  color: #1f2933;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: #f7f8fa;
}

main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 24px 40px;
}

header {
  margin-bottom: 24px;
}

h1,
h2 {
  margin: 0;
  line-height: 1.2;
}

h1 {
  font-size: 2rem;
  font-weight: 760;
}

h2 {
  margin-bottom: 12px;
  font-size: 1.1rem;
}

section {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #d8dde6;
}

table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border: 1px solid #d8dde6;
}

th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid #e6eaf0;
  text-align: left;
  vertical-align: top;
}

th {
  width: 20%;
  background: #eef2f6;
  color: #344054;
  font-size: 0.78rem;
  letter-spacing: 0;
  text-transform: uppercase;
}

td {
  color: #1f2933;
  overflow-wrap: anywhere;
}

tr:last-child td,
tr:last-child th {
  border-bottom: 0;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.metric {
  min-height: 96px;
  padding: 16px;
  background: #ffffff;
  border: 1px solid #d8dde6;
}

.metric-label {
  display: block;
  margin-bottom: 8px;
  color: #526170;
  font-size: 0.82rem;
}

.metric-value {
  display: block;
  color: #111827;
  font-size: 1.8rem;
  font-weight: 760;
  line-height: 1.1;
}

.empty {
  margin: 0;
  padding: 12px;
  background: #ffffff;
  border: 1px solid #d8dde6;
  color: #526170;
}

.subtle {
  color: #526170;
}
""".strip()


def render_risk_dashboard(
    risk: Mapping[str, Any],
    *,
    kanon: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Render a deterministic, self-contained HTML risk dashboard.

    Args:
        risk: Mapping returned by :func:`openmed.risk.risk_report`.
        kanon: Optional mapping returned by :func:`openmed.risk.kanon_report`
            or :func:`openmed.risk.enforce_kanon`.
        title: Optional document and page title. Defaults to a stable title.

    Returns:
        A complete HTML document with inline CSS and no external assets.
    """

    document_title = title or _DEFAULT_TITLE
    body = [
        _render_header(document_title),
        _render_headline_metrics(risk),
        _render_singletons(risk.get("singleton_records") or ()),
        _render_quasi_identifiers(risk.get("quasi_identifiers") or ()),
    ]
    if kanon is not None:
        body.append(_render_kanon(kanon))

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{_escape(document_title)}</title>",
            "<style>",
            _CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            *body,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_risk_dashboard(
    risk: Mapping[str, Any],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write a rendered risk dashboard and return the output path."""

    output_path = Path(path)
    output_path.write_text(render_risk_dashboard(risk, **kwargs), encoding="utf-8")
    return output_path


def _render_header(title: str) -> str:
    return "\n".join(
        [
            "<header>",
            f"<h1>{_escape(title)}</h1>",
            (
                '<p class="subtle">'
                "Residual disclosure risk summary for de-identified records."
                "</p>"
            ),
            "</header>",
        ]
    )


def _render_headline_metrics(risk: Mapping[str, Any]) -> str:
    metrics = [
        ("Leakage rate", _format_rate(risk.get("leakage_rate"))),
        ("Re-identification rate", _format_rate(risk.get("reid_rate"))),
        ("Minimum k", _format_count(risk.get("k_min"))),
    ]
    cards = [
        "\n".join(
            [
                '<article class="metric">',
                f'<span class="metric-label">{_escape(label)}</span>',
                f'<strong class="metric-value">{_escape(value)}</strong>',
                "</article>",
            ]
        )
        for label, value in metrics
    ]
    return "\n".join(
        [
            '<section aria-labelledby="headline-risk-metrics">',
            '<h2 id="headline-risk-metrics">Headline Metrics</h2>',
            '<div class="metric-grid">',
            *cards,
            "</div>",
            "</section>",
        ]
    )


def _render_singletons(singletons: Any) -> str:
    rows = [
        record for record in _as_sequence(singletons) if isinstance(record, Mapping)
    ]
    rows.sort(key=_singleton_sort_key)

    if not rows:
        table = '<p class="empty">No singleton records reported.</p>'
    else:
        table = _table(
            ["Record ID", "Record index", "Effective k", "Quasi-identifier key"],
            [
                [
                    _display(record.get("record_id")),
                    _display(record.get("record_index")),
                    _display(record.get("effective_k")),
                    _format_quasi_identifier_key(record.get("quasi_identifier_key")),
                ]
                for record in rows
            ],
        )

    return "\n".join(
        [
            '<section aria-labelledby="singleton-records">',
            '<h2 id="singleton-records">Singleton Records</h2>',
            table,
            "</section>",
        ]
    )


def _render_quasi_identifiers(quasi_identifiers: Any) -> str:
    rows = _top_quasi_identifier_rows(quasi_identifiers)
    if not rows:
        table = '<p class="empty">No quasi-identifiers reported.</p>'
    else:
        table = _table(
            ["Category", "Value", "Count", "Records", "Sources"],
            [
                [
                    category,
                    value,
                    str(count),
                    ", ".join(records),
                    ", ".join(sources),
                ]
                for category, value, count, records, sources in rows
            ],
        )

    return "\n".join(
        [
            '<section aria-labelledby="top-quasi-identifiers">',
            '<h2 id="top-quasi-identifiers">Top Quasi-identifiers</h2>',
            table,
            "</section>",
        ]
    )


def _render_kanon(kanon: Mapping[str, Any]) -> str:
    if isinstance(kanon.get("kanon"), Mapping):
        return _render_kanon_enforcement(kanon)

    size_distribution = _as_sequence(kanon.get("class_size_distribution") or ())
    class_rows = _equivalence_class_rows(kanon.get("equivalence_classes") or ())

    sections = [
        '<section aria-labelledby="kanon-summary">',
        '<h2 id="kanon-summary">K-Anonymity Equivalence Classes</h2>',
        '<div class="metric-grid">',
        _metric("Records", _format_count(kanon.get("record_count"))),
        _metric("Minimum k", _format_count(kanon.get("k"))),
        _metric("Class count", _format_count(kanon.get("class_count"))),
        "</div>",
    ]

    if size_distribution:
        sections.extend(
            [
                "<h2>Class Size Distribution</h2>",
                _table(
                    ["Class size", "Class count"],
                    [
                        [_display(size), _display(count)]
                        for size, count in _sorted_size_distribution(size_distribution)
                    ],
                ),
            ]
        )

    if class_rows:
        sections.extend(
            [
                "<h2>Equivalence Classes</h2>",
                _table(
                    ["Key", "Size", "Members", "l-diversity", "t-closeness"],
                    class_rows,
                ),
            ]
        )
    else:
        sections.append('<p class="empty">No equivalence classes reported.</p>')

    sections.append("</section>")
    return "\n".join(sections)


def _render_kanon_enforcement(enforcement: Mapping[str, Any]) -> str:
    kanon = _mapping(enforcement.get("kanon"))
    generalization = _mapping(enforcement.get("generalization"))
    bounds = _mapping(enforcement.get("bounds"))
    self_check = _mapping(bounds.get("numeric_self_check"))
    selected_levels = _mapping(generalization.get("levels"))

    sections = [
        '<section aria-labelledby="kanon-enforcement">',
        '<h2 id="kanon-enforcement">K-Anonymity Enforcement</h2>',
        '<div class="metric-grid">',
        _metric("Target k", _format_count(enforcement.get("target_k"))),
        _metric("Measured k", _format_count(kanon.get("k"))),
        _metric("Released", _format_count(enforcement.get("released_count"))),
        _metric("Suppressed", _format_count(enforcement.get("suppressed_count"))),
        _metric(
            "Max re-id bound",
            _format_rate(bounds.get("max_reidentification_upper_bound")),
        ),
        _metric("Bound check", "pass" if self_check.get("passed") else "fail"),
        "</div>",
    ]

    if selected_levels:
        sections.extend(
            [
                "<h2>Selected Generalization</h2>",
                _table(
                    ["Field", "Level", "Name", "Loss"],
                    [
                        [
                            field,
                            _display(_mapping(level).get("level")),
                            _display(_mapping(level).get("name")),
                            _display(_mapping(level).get("loss")),
                        ]
                        for field, level in sorted(selected_levels.items())
                    ],
                ),
            ]
        )

    sections.extend(
        [
            "<h2>Enforced Equivalence Classes</h2>",
            _table(
                ["Key", "Size", "Members", "l-diversity", "t-closeness"],
                _equivalence_class_rows(kanon.get("equivalence_classes") or ()),
            )
            if kanon.get("equivalence_classes")
            else '<p class="empty">No equivalence classes reported.</p>',
        ]
    )

    suppressed = _as_sequence(enforcement.get("suppressed_records") or ())
    if suppressed:
        sections.extend(
            [
                "<h2>Suppressed Records</h2>",
                _table(
                    ["Offset", "Record hash", "Reason"],
                    [
                        [
                            _display(_mapping(record).get("offset")),
                            _display(_mapping(record).get("record_hash")),
                            _display(_mapping(record).get("reason")),
                        ]
                        for record in suppressed
                        if isinstance(record, Mapping)
                    ],
                ),
            ]
        )

    sections.append("</section>")
    return "\n".join(sections)


def _metric(label: str, value: str) -> str:
    return "\n".join(
        [
            '<article class="metric">',
            f'<span class="metric-label">{_escape(label)}</span>',
            f'<strong class="metric-value">{_escape(value)}</strong>',
            "</article>",
        ]
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_escape(cell)}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return "\n".join(
        [
            "<table>",
            "<thead>",
            f"<tr>{header_html}</tr>",
            "</thead>",
            "<tbody>",
            *body,
            "</tbody>",
            "</table>",
        ]
    )


def _top_quasi_identifier_rows(
    quasi_identifiers: Any,
) -> list[tuple[str, str, int, list[str], list[str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    records: dict[tuple[str, str], set[str]] = {}
    sources: dict[tuple[str, str], set[str]] = {}

    for item in _as_sequence(quasi_identifiers):
        if not isinstance(item, Mapping):
            continue
        category = _display(item.get("category"))
        value = _display(item.get("value", item.get("normalized_value")))
        key = (category, value)
        counts[key] += 1
        records.setdefault(key, set()).add(_record_reference(item))
        source = item.get("source")
        if source is not None:
            sources.setdefault(key, set()).add(_display(source))

    rows = [
        (
            category,
            value,
            count,
            sorted(records.get((category, value), set())),
            sorted(sources.get((category, value), set())),
        )
        for (category, value), count in counts.items()
    ]
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    return rows[:10]


def _record_reference(item: Mapping[str, Any]) -> str:
    record_id = item.get("record_id")
    if record_id is not None:
        return _display(record_id)
    return _display(item.get("record_index"))


def _format_quasi_identifier_key(value: Any) -> str:
    parts = []
    for entry in _as_sequence(value):
        if not isinstance(entry, Mapping):
            parts.append(_display(entry))
            continue
        category = _display(entry.get("category"))
        values = ", ".join(_display(item) for item in _as_sequence(entry.get("values")))
        parts.append(f"{category}: {values}" if values else category)
    return "; ".join(parts) if parts else ""


def _equivalence_class_rows(classes: Any) -> list[list[str]]:
    rows = []
    for cls in _as_sequence(classes):
        if not isinstance(cls, Mapping):
            continue
        rows.append(
            [
                _display(cls.get("key")),
                _display(cls.get("size")),
                _display(cls.get("members")),
                _display(cls.get("l_diversity")),
                _display(cls.get("t_closeness")),
            ]
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return rows


def _sorted_size_distribution(distribution: Sequence[Any]) -> list[tuple[Any, Any]]:
    rows = []
    for entry in distribution:
        if (
            isinstance(entry, Sequence)
            and not isinstance(entry, str)
            and len(entry) >= 2
        ):
            rows.append((entry[0], entry[1]))
    rows.sort(key=lambda row: (_sort_value(row[0]), _sort_value(row[1])))
    return rows


def _singleton_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _sort_value(record.get("record_id")),
        _sort_value(record.get("record_index")),
    )


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _format_rate(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.1%}"
    return _display(value)


def _format_count(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _display(value)


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return json.dumps(list(value), sort_keys=True, ensure_ascii=True)
    return str(value)


def _sort_value(value: Any) -> str:
    return _display(value)


def _escape(value: Any) -> str:
    return html_mod.escape(str(value), quote=True)
