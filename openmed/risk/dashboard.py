"""HTML dashboards for local-sensitive diagnostics and aggregate release evidence."""

from __future__ import annotations

import html as html_mod
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .release import AnonymizationResult, ReleaseAssessment

__all__ = [
    "render_release_assessment_dashboard",
    "render_risk_dashboard",
    "write_release_assessment_dashboard",
    "write_risk_dashboard",
]

_DEFAULT_TITLE = "OpenMed Risk Dashboard"
_DEFAULT_RELEASE_TITLE = "OpenMed Release Assessment Dashboard"
_SAFE_METADATA_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

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

.notice {
  padding: 12px 14px;
  border-left: 4px solid #9b2c2c;
  background: #fff5f5;
  color: #742a2a;
}

.safe-notice {
  padding: 12px 14px;
  border-left: 4px solid #2b6cb0;
  background: #ebf8ff;
  color: #2a4365;
}
""".strip()


def render_risk_dashboard(
    risk: Mapping[str, Any],
    *,
    kanon: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Render a local-sensitive, self-contained diagnostic dashboard.

    This legacy renderer intentionally displays raw quasi-identifier values,
    record references, equivalence-class keys, and member indices. Keep its
    output on trusted local storage. Use
    :func:`render_release_assessment_dashboard` for shareable aggregate
    evidence.

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

    return _render_document(document_title, body)


def write_risk_dashboard(
    risk: Mapping[str, Any],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write a local-sensitive diagnostic dashboard and return its path."""

    output_path = Path(path)
    output_path.write_text(render_risk_dashboard(risk, **kwargs), encoding="utf-8")
    return output_path


def render_release_assessment_dashboard(
    assessment: ReleaseAssessment | AnonymizationResult | Mapping[str, Any],
    *,
    title: str | None = None,
) -> str:
    """Render only allow-listed aggregate release evidence.

    Args:
        assessment: A :class:`ReleaseAssessment`, an
            :class:`AnonymizationResult`, or the mapping returned by
            ``to_dict()`` / ``to_safe_dict()`` on those objects. Detailed
            ``risk_report`` and ``kanon_report`` mappings are rejected.
        title: Optional document and page title.

    Returns:
        A complete, deterministic HTML document with no external assets.

    Raises:
        TypeError: If ``assessment`` is not a supported aggregate artifact.
    """

    payload = _release_dashboard_payload(assessment)
    document_title = title or _DEFAULT_RELEASE_TITLE
    body = [_render_release_header(document_title)]
    artifact = payload.get("artifact")
    if artifact == "deidentification_release_assessment":
        body.append(_render_release_assessment(payload, section_id="assessment"))
    else:
        body.extend(_render_anonymization_summary(payload))
    return _render_document(document_title, body)


def write_release_assessment_dashboard(
    assessment: ReleaseAssessment | AnonymizationResult | Mapping[str, Any],
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write an aggregate release-evidence dashboard and return its path."""

    output_path = Path(path)
    output_path.write_text(
        render_release_assessment_dashboard(assessment, title=title),
        encoding="utf-8",
    )
    return output_path


def _render_document(title: str, body: Sequence[str]) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{_escape(title)}</title>",
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


def _render_header(title: str) -> str:
    return "\n".join(
        [
            "<header>",
            f"<h1>{_escape(title)}</h1>",
            (
                '<p class="subtle">'
                "Record-level residual disclosure-risk diagnostics."
                "</p>"
            ),
            (
                '<p class="notice"><strong>Local-sensitive diagnostic.</strong> '
                "This report can contain raw quasi-identifier values, record "
                "references, class keys, and member indices. Keep it on trusted "
                "local storage.</p>"
            ),
            "</header>",
        ]
    )


def _render_release_header(title: str) -> str:
    return "\n".join(
        [
            "<header>",
            f"<h1>{_escape(title)}</h1>",
            (
                '<p class="subtle">'
                "Aggregate disclosure-risk evidence for a declared release policy."
                "</p>"
            ),
            (
                '<p class="safe-notice"><strong>Aggregate evidence only.</strong> '
                "Qualified expert review is required. This dashboard is not an "
                "Expert Determination or compliance certificate.</p>"
            ),
            "</header>",
        ]
    )


def _release_dashboard_payload(value: Any) -> Mapping[str, Any]:
    value_type = type(value)
    if (
        value_type.__module__ == "openmed.risk.release"
        and value_type.__name__ == "ReleaseAssessment"
    ):
        payload = value.to_dict()
    elif (
        value_type.__module__ == "openmed.risk.release"
        and value_type.__name__ == "AnonymizationResult"
    ):
        payload = value.to_safe_dict()
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise TypeError(
            "assessment must be a ReleaseAssessment, AnonymizationResult, "
            "or their safe aggregate mapping"
        )

    artifact = payload.get("artifact") if isinstance(payload, Mapping) else None
    if artifact not in {
        "deidentification_release_assessment",
        "deidentification_anonymization_summary",
    }:
        raise TypeError(
            "assessment mapping must be a safe release assessment or "
            "anonymization summary artifact"
        )
    return payload


def _render_release_assessment(
    assessment: Mapping[str, Any],
    *,
    section_id: str,
    heading: str = "Release Assessment",
) -> str:
    policy = _mapping(assessment.get("policy"))
    kanon = _mapping(assessment.get("k_anonymity"))
    identity_risk = _mapping(assessment.get("sample_identity_risk"))
    meets_policy = assessment.get("meets_policy")
    verdict = (
        "meets declared policy"
        if meets_policy is True
        else "does not meet declared policy"
        if meets_policy is False
        else "not reported"
    )
    sections = [
        f'<section aria-labelledby="{_escape(section_id)}">',
        f'<h2 id="{_escape(section_id)}">{_escape(heading)}</h2>',
        '<div class="metric-grid">',
        _metric("Result", verdict),
        _metric("Rows", _format_count(assessment.get("row_count"))),
        _metric(
            "Privacy units",
            _format_count(assessment.get("privacy_unit_count")),
        ),
        _metric("Achieved k", _format_count(kanon.get("achieved_k"))),
        _metric("Target k", _format_count(policy.get("target_k"))),
        _metric("Max sample identity risk", _format_rate(identity_risk.get("max"))),
        "</div>",
        "<h2>Declared Policy</h2>",
        _table(
            ["Field", "Value"],
            [
                [
                    "Quasi-identifiers",
                    ", ".join(
                        _safe_metadata_strings(assessment.get("quasi_identifiers"))
                    ),
                ],
                [
                    "Sensitive attributes",
                    ", ".join(
                        _safe_metadata_strings(assessment.get("sensitive_attributes"))
                    ),
                ],
                ["Target l", _format_count(policy.get("target_l"))],
                ["Target t", _format_rate(policy.get("target_t"))],
                [
                    "Warnings",
                    _format_count(
                        assessment.get(
                            "warning_count",
                            len(_as_sequence(assessment.get("warnings"))),
                        )
                    ),
                ],
            ],
        ),
    ]
    distribution = _safe_distribution_rows(kanon.get("class_size_distribution"))
    if distribution:
        sections.extend(
            [
                "<h2>Class Size Distribution</h2>",
                _table(
                    ["Class size", "Class count"],
                    [[str(size), str(count)] for size, count in distribution],
                ),
            ]
        )
    attribute_rows = _safe_attribute_rows(assessment.get("attribute_disclosure"))
    if attribute_rows:
        sections.extend(
            [
                "<h2>Attribute Disclosure</h2>",
                _table(
                    [
                        "Attribute",
                        "Achieved l",
                        "l violations",
                        "Achieved t",
                        "t violations",
                    ],
                    attribute_rows,
                ),
            ]
        )
    sections.append("</section>")
    return "\n".join(sections)


def _render_anonymization_summary(
    summary: Mapping[str, Any],
) -> list[str]:
    sections: list[str] = []
    before = summary.get("before")
    after = summary.get("after")
    if isinstance(before, Mapping):
        sections.append(
            _render_release_assessment(
                before,
                section_id="assessment-before",
                heading="Before Anonymization",
            )
        )
    if isinstance(after, Mapping):
        sections.append(
            _render_release_assessment(
                after,
                section_id="assessment-after",
                heading="After Anonymization",
            )
        )

    generalization = _mapping(summary.get("generalization"))
    utility = _mapping(summary.get("utility"))
    sections.append(
        "\n".join(
            [
                '<section aria-labelledby="transformation-summary">',
                '<h2 id="transformation-summary">Transformation and Utility</h2>',
                '<div class="metric-grid">',
                _metric(
                    "Information loss",
                    _format_rate(generalization.get("information_loss")),
                ),
                _metric(
                    "Row suppression",
                    _format_rate(utility.get("row_suppression_rate")),
                ),
                _metric(
                    "Privacy-unit suppression",
                    _format_rate(utility.get("privacy_unit_suppression_rate")),
                ),
                _metric(
                    "QI cell change",
                    _format_rate(utility.get("quasi_identifier_cell_change_rate")),
                ),
                _metric(
                    "Released rows",
                    _format_count(utility.get("released_rows")),
                ),
                "</div>",
                *_render_safe_generalization_levels(generalization.get("levels")),
                "</section>",
            ]
        )
    )
    return sections


def _render_safe_generalization_levels(value: Any) -> list[str]:
    rows = []
    for item in _as_sequence(value):
        if not isinstance(item, Mapping):
            continue
        attribute = item.get("attribute")
        if not isinstance(attribute, str) or not _SAFE_METADATA_IDENTIFIER_RE.fullmatch(
            attribute
        ):
            continue
        rows.append(
            [
                attribute,
                _format_count(item.get("level")),
                _format_rate(item.get("loss")),
            ]
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    if not rows:
        return []
    return [
        "<h2>Selected Generalization</h2>",
        _table(["Attribute", "Level", "Loss"], rows),
    ]


def _safe_metadata_strings(value: Any) -> list[str]:
    return sorted(
        item
        for item in _as_sequence(value)
        if isinstance(item, str) and _SAFE_METADATA_IDENTIFIER_RE.fullmatch(item)
    )


def _safe_distribution_rows(value: Any) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            size = item.get("size")
            count = item.get("class_count")
        elif (
            isinstance(item, Sequence) and not isinstance(item, str) and len(item) >= 2
        ):
            size, count = item[0], item[1]
        else:
            continue
        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and isinstance(count, int)
            and not isinstance(count, bool)
        ):
            rows.append((size, count))
    return sorted(rows)


def _safe_attribute_rows(value: Any) -> list[list[str]]:
    rows = []
    for item in _as_sequence(value):
        if not isinstance(item, Mapping):
            continue
        attribute = item.get("attribute")
        if not isinstance(attribute, str) or not _SAFE_METADATA_IDENTIFIER_RE.fullmatch(
            attribute
        ):
            continue
        l_diversity = _mapping(item.get("l_diversity"))
        t_closeness = _mapping(item.get("t_closeness"))
        rows.append(
            [
                attribute,
                _format_number(l_diversity.get("achieved")),
                _format_count(l_diversity.get("violating_classes")),
                _format_number(t_closeness.get("achieved")),
                _format_count(t_closeness.get("violating_classes")),
            ]
        )
    rows.sort(key=lambda row: row[0])
    return rows


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
    if type(value) is int:
        return f"{value:.1%}"
    if type(value) is float and math.isfinite(value):
        return f"{value:.1%}"
    return ""


def _format_count(value: Any) -> str:
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return ""


def _format_number(value: Any) -> str:
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return str(value)
    return ""


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
