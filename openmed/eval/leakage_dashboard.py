"""Per-language PHI leakage dashboards over benchmark runs.

The dashboard consumes the count evidence already emitted by leakage metrics
and :mod:`openmed.eval.leakage_heatmap`.  It deliberately discards all other
benchmark fields so model names, fixture identifiers, metadata, and source text
cannot enter the generated artifacts.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.leakage_heatmap import LeakageHeatmap
from openmed.eval.report import BenchmarkReport

LEAKAGE_DASHBOARD_SCHEMA_VERSION = 1
LEAKAGE_DASHBOARD_ARTIFACT_TYPE = "openmed.per_language_leakage_dashboard"
DEFAULT_WORST_LABEL_COUNT = 5

__all__ = [
    "DEFAULT_WORST_LABEL_COUNT",
    "LEAKAGE_DASHBOARD_ARTIFACT_TYPE",
    "LEAKAGE_DASHBOARD_SCHEMA_VERSION",
    "LanguageLeakageSummary",
    "LeakageDashboard",
    "LeakageDashboardArtifactPaths",
    "LeakageTrendPoint",
    "ResidualLabelSummary",
    "aggregate_leakage_runs",
    "build_leakage_dashboard",
    "render_leakage_dashboard",
    "render_leakage_dashboard_html",
    "render_leakage_dashboard_json",
    "write_leakage_dashboard",
    "write_leakage_dashboard_artifacts",
]


@dataclass(frozen=True)
class LeakageTrendPoint:
    """PHI-free leakage counts for one language in one benchmark run."""

    run_index: int
    leaked_chars: int
    total_chars: int
    rate: float

    def to_dict(self) -> dict[str, int | float]:
        """Return this trend point as a deterministic JSON-ready mapping."""
        return {
            "leaked_chars": self.leaked_chars,
            "rate": self.rate,
            "run_index": self.run_index,
            "total_chars": self.total_chars,
        }


@dataclass(frozen=True)
class ResidualLabelSummary:
    """Residual leaked-character evidence for one canonical PII label."""

    label: str
    leaked_chars: int
    total_chars: int
    rate: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return this label summary as a deterministic JSON-ready mapping."""
        return {
            "label": self.label,
            "leaked_chars": self.leaked_chars,
            "rate": self.rate,
            "total_chars": self.total_chars,
        }


@dataclass(frozen=True)
class LanguageLeakageSummary:
    """Aggregate leakage, trend, and residual categories for one language."""

    language: str
    leaked_chars: int
    total_chars: int
    rate: float
    residual_by_label: tuple[ResidualLabelSummary, ...]
    worst_labels: tuple[ResidualLabelSummary, ...]
    trend: tuple[LeakageTrendPoint, ...]
    threshold: float | None = None

    @property
    def threshold_passed(self) -> bool | None:
        """Return the language gate result, or ``None`` when not configured."""
        if self.threshold is None:
            return None
        return self.rate <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return only allow-listed labels, counts, rates, and run offsets."""
        return {
            "gate_passed": self.threshold_passed,
            "language": self.language,
            "leakage": {
                "leaked_chars": self.leaked_chars,
                "rate": self.rate,
                "total_chars": self.total_chars,
            },
            "leakage_rate": self.rate,
            "residual_count": self.leaked_chars,
            "residual_counts_by_category": {
                item.label: item.leaked_chars for item in self.residual_by_label
            },
            "residuals_by_label": [item.to_dict() for item in self.residual_by_label],
            "threshold": self.threshold,
            "trend": [point.to_dict() for point in self.trend],
            "worst_labels": [item.to_dict() for item in self.worst_labels],
        }


@dataclass(frozen=True)
class LeakageDashboard:
    """Deterministic per-language leakage dashboard artifact."""

    run_count: int
    languages: tuple[LanguageLeakageSummary, ...]
    thresholds: Mapping[str, float]

    @classmethod
    def from_runs(
        cls,
        runs: Iterable[BenchmarkReport | LeakageHeatmap | Mapping[str, Any]],
        *,
        worst_n: int = DEFAULT_WORST_LABEL_COUNT,
        thresholds: float | Mapping[str, float] | None = None,
    ) -> "LeakageDashboard":
        """Build a dashboard from benchmark reports or leakage heatmaps."""
        return build_leakage_dashboard(
            runs,
            worst_n=worst_n,
            thresholds=thresholds,
        )

    @property
    def gate_passed(self) -> bool | None:
        """Return the combined configured threshold result."""
        configured = [
            summary.threshold_passed
            for summary in self.languages
            if summary.threshold_passed is not None
        ]
        if not configured:
            return None
        return all(configured)

    def to_dict(self) -> dict[str, Any]:
        """Return the companion JSON payload used by CI and other tooling."""
        by_language = {
            summary.language: summary.to_dict() for summary in self.languages
        }
        violations = [
            {
                "language": summary.language,
                "observed": summary.rate,
                "threshold": summary.threshold,
            }
            for summary in self.languages
            if summary.threshold_passed is False
        ]
        return {
            "artifact_type": LEAKAGE_DASHBOARD_ARTIFACT_TYPE,
            "by_language": by_language,
            "gate": {
                "configured": bool(self.thresholds),
                "passed": self.gate_passed,
                "violations": violations,
            },
            "languages": [summary.language for summary in self.languages],
            "run_count": self.run_count,
            "schema_version": LEAKAGE_DASHBOARD_SCHEMA_VERSION,
            "thresholds": {
                "max_leakage_rate_by_language": dict(self.thresholds),
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the companion payload with stable key ordering."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_html(self) -> str:
        """Render a self-contained HTML dashboard."""
        return render_leakage_dashboard_html(self)


@dataclass(frozen=True)
class LeakageDashboardArtifactPaths:
    """Paths written for a dashboard's HTML and companion JSON artifacts."""

    html: Path
    json: Path


def build_leakage_dashboard(
    runs: Iterable[BenchmarkReport | LeakageHeatmap | Mapping[str, Any]],
    *,
    worst_n: int = DEFAULT_WORST_LABEL_COUNT,
    thresholds: float | Mapping[str, float] | None = None,
) -> LeakageDashboard:
    """Aggregate one or more benchmark runs into per-language leakage panels.

    A run may be a :class:`BenchmarkReport`, a serialized report mapping, a
    :class:`LeakageHeatmap`, or its serialized mapping.  Multi-language runs
    must carry the OM-403 ``leakage_heatmap`` payload because one-dimensional
    leakage slices cannot correctly attribute residual labels to languages.

    Args:
        runs: Ordered benchmark evidence. Input order defines trend offsets.
        worst_n: Maximum number of residual labels listed per language.
        thresholds: Optional maximum leakage rate for every language, or a
            mapping from language code to maximum rate. ``"*"`` provides a
            default which explicit language entries may override.

    Returns:
        A PHI-free dashboard covering every wired PII language.

    Raises:
        ValueError: If no runs are supplied or count evidence is invalid or
            insufficient for safe per-language aggregation.
    """
    if isinstance(worst_n, bool) or not isinstance(worst_n, int) or worst_n < 0:
        raise ValueError("worst_n must be a non-negative integer")

    run_payloads = tuple(_run_cells(run) for run in runs)
    if not run_payloads:
        raise ValueError("at least one benchmark run is required")

    resolved_thresholds = _resolve_thresholds(thresholds)
    aggregate: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    trends: defaultdict[str, list[LeakageTrendPoint]] = defaultdict(list)

    for run_index, cells in enumerate(run_payloads):
        run_totals: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for language, label_cells in cells.items():
            for label, (leaked_chars, total_chars) in label_cells.items():
                aggregate[language][label][0] += leaked_chars
                aggregate[language][label][1] += total_chars
                run_totals[language][0] += leaked_chars
                run_totals[language][1] += total_chars

        for language in sorted(SUPPORTED_LANGUAGES):
            leaked_chars, total_chars = run_totals[language]
            trends[language].append(
                LeakageTrendPoint(
                    run_index=run_index,
                    leaked_chars=leaked_chars,
                    total_chars=total_chars,
                    rate=_safe_rate(leaked_chars, total_chars),
                )
            )

    summaries: list[LanguageLeakageSummary] = []
    for language in sorted(SUPPORTED_LANGUAGES):
        label_summaries = tuple(
            ResidualLabelSummary(
                label=label,
                leaked_chars=counts[0],
                total_chars=counts[1],
                rate=_safe_rate(counts[0], counts[1]),
            )
            for label, counts in sorted(aggregate[language].items())
            if counts[0] or counts[1]
        )
        leaked_chars = sum(item.leaked_chars for item in label_summaries)
        total_chars = sum(item.total_chars for item in label_summaries)
        worst_labels = tuple(
            sorted(
                (item for item in label_summaries if item.leaked_chars > 0),
                key=lambda item: (-item.rate, item.label),
            )[:worst_n]
        )
        summaries.append(
            LanguageLeakageSummary(
                language=language,
                leaked_chars=leaked_chars,
                total_chars=total_chars,
                rate=_safe_rate(leaked_chars, total_chars),
                residual_by_label=label_summaries,
                worst_labels=worst_labels,
                trend=tuple(trends[language]),
                threshold=resolved_thresholds.get(language),
            )
        )

    return LeakageDashboard(
        run_count=len(run_payloads),
        languages=tuple(summaries),
        thresholds=resolved_thresholds,
    )


def aggregate_leakage_runs(
    runs: Iterable[BenchmarkReport | LeakageHeatmap | Mapping[str, Any]],
    *,
    worst_n: int = DEFAULT_WORST_LABEL_COUNT,
    thresholds: float | Mapping[str, float] | None = None,
) -> LeakageDashboard:
    """Alias for :func:`build_leakage_dashboard`."""
    return build_leakage_dashboard(
        runs,
        worst_n=worst_n,
        thresholds=thresholds,
    )


def render_leakage_dashboard_json(
    dashboard: LeakageDashboard,
    *,
    indent: int = 2,
) -> str:
    """Render the deterministic companion JSON document."""
    return dashboard.to_json(indent=indent)


def render_leakage_dashboard_html(dashboard: LeakageDashboard) -> str:
    """Render a deterministic, self-contained, sortable HTML dashboard."""
    summary_rows = "\n".join(_summary_row(summary) for summary in dashboard.languages)
    panels = "\n".join(_language_panel(summary) for summary in dashboard.languages)
    gate = _gate_text(dashboard.gate_passed)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            "<title>OpenMed Per-language Leakage Dashboard</title>",
            "<style>",
            _CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            "<h1>Per-language Leakage Dashboard</h1>",
            (
                '<p class="subtle">'
                f"{dashboard.run_count} benchmark run(s); threshold gate: "
                f"{_escape(gate)}."
                "</p>"
            ),
            "</header>",
            '<section aria-labelledby="language-summary">',
            '<h2 id="language-summary">Language summary</h2>',
            '<div class="table-scroll">',
            '<table id="leakage-summary-table">',
            "<thead><tr>",
            _sortable_header("Language", "language"),
            _sortable_header("Leakage", "rate"),
            _sortable_header("Residual chars", "residual"),
            _sortable_header("Measured PHI chars", "total"),
            "<th>Top residual labels</th>",
            "<th>Gate</th>",
            "</tr></thead>",
            "<tbody>",
            summary_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            panels,
            "</main>",
            "<script>",
            _SORT_SCRIPT,
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def render_leakage_dashboard(dashboard: LeakageDashboard) -> str:
    """Alias for :func:`render_leakage_dashboard_html`."""
    return render_leakage_dashboard_html(dashboard)


def write_leakage_dashboard(
    dashboard: LeakageDashboard,
    html_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> LeakageDashboardArtifactPaths:
    """Write self-contained HTML and companion JSON dashboard artifacts."""
    output_html = Path(html_path)
    output_json = (
        Path(json_path) if json_path is not None else output_html.with_suffix(".json")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(dashboard.to_html(), encoding="utf-8")
    output_json.write_text(dashboard.to_json() + "\n", encoding="utf-8")
    return LeakageDashboardArtifactPaths(html=output_html, json=output_json)


def write_leakage_dashboard_artifacts(
    dashboard: LeakageDashboard,
    html_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> LeakageDashboardArtifactPaths:
    """Alias for :func:`write_leakage_dashboard`."""
    return write_leakage_dashboard(
        dashboard,
        html_path,
        json_path=json_path,
    )


def _run_cells(
    run: BenchmarkReport | LeakageHeatmap | Mapping[str, Any],
) -> dict[str, dict[str, tuple[int, int]]]:
    if isinstance(run, LeakageHeatmap):
        return _cells_from_heatmap(run.to_dict())

    payload: Mapping[str, Any]
    if isinstance(run, BenchmarkReport):
        payload = run.metrics
    elif isinstance(run, Mapping):
        payload = run
    else:
        raise ValueError("benchmark runs must be reports, heatmaps, or mappings")

    heatmap = _find_heatmap(payload)
    if heatmap is not None:
        return _cells_from_heatmap(heatmap)

    metrics = _as_mapping(payload.get("metrics")) or payload
    leakage = _as_mapping(metrics.get("leakage"))
    if leakage is None:
        raise ValueError("benchmark run lacks leakage count evidence")
    heatmap = _find_heatmap(leakage)
    if heatmap is not None:
        return _cells_from_heatmap(heatmap)
    return _cells_from_single_language_metrics(leakage)


def _find_heatmap(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if _as_mapping(payload.get("cells")) is not None:
        return payload
    for key in ("leakage_heatmap", "heatmap"):
        candidate = _as_mapping(payload.get(key))
        if candidate is not None and _as_mapping(candidate.get("cells")) is not None:
            return candidate
    metrics = _as_mapping(payload.get("metrics"))
    if metrics is not None and metrics is not payload:
        return _find_heatmap(metrics)
    return None


def _cells_from_heatmap(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, tuple[int, int]]]:
    raw_cells = _as_mapping(payload.get("cells"))
    if raw_cells is None:
        raise ValueError("leakage heatmap lacks cells")

    result: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for raw_label, raw_language_cells in raw_cells.items():
        label = _canonical_label(raw_label)
        language_cells = _as_mapping(raw_language_cells)
        if language_cells is None:
            raise ValueError(f"leakage heatmap label {label} must map languages")
        for raw_language, raw_cell in language_cells.items():
            language = _language(raw_language)
            cell = _as_mapping(raw_cell)
            if cell is None:
                raise ValueError(f"leakage heatmap cell {label}/{language} is invalid")
            counts = _counts(cell, context=f"heatmap cell {label}/{language}")
            if counts != (0, 0):
                result[language][label] = counts

    column_totals = _as_mapping(
        payload.get("column_totals") or payload.get("col_totals")
    )
    if column_totals is not None:
        for raw_language, raw_total in column_totals.items():
            language = _language(raw_language)
            total = _as_mapping(raw_total)
            if total is None:
                raise ValueError(f"column total for {language} is invalid")
            expected = _counts(total, context=f"column total {language}")
            observed = (
                sum(counts[0] for counts in result[language].values()),
                sum(counts[1] for counts in result[language].values()),
            )
            if expected != observed:
                raise ValueError(
                    f"column total for {language} does not match heatmap cells"
                )
    return {language: dict(cells) for language, cells in result.items()}


def _cells_from_single_language_metrics(
    leakage: Mapping[str, Any],
) -> dict[str, dict[str, tuple[int, int]]]:
    leaked_by_language = _count_mapping(
        leakage.get("leaked_chars_by_language"),
        context="leaked_chars_by_language",
        keys="language",
    )
    total_by_language = _count_mapping(
        leakage.get("total_chars_by_language"),
        context="total_chars_by_language",
        keys="language",
    )
    leaked_by_label = _count_mapping(
        leakage.get("leaked_chars_by_label"),
        context="leaked_chars_by_label",
        keys="label",
    )
    total_by_label = _count_mapping(
        leakage.get("total_chars_by_label"),
        context="total_chars_by_label",
        keys="label",
    )

    active_languages = sorted(
        language
        for language in set(leaked_by_language) | set(total_by_language)
        if leaked_by_language.get(language, 0) or total_by_language.get(language, 0)
    )
    has_label_evidence = any(leaked_by_label.values()) or any(total_by_label.values())
    if not active_languages and not has_label_evidence:
        return {}
    if len(active_languages) != 1:
        raise ValueError(
            "multi-language benchmark runs require a leakage_heatmap payload"
        )

    language = active_languages[0]
    language_counts = (
        leaked_by_language.get(language, 0),
        total_by_language.get(language, 0),
    )
    label_cells = {
        label: (leaked_by_label.get(label, 0), total_by_label.get(label, 0))
        for label in sorted(set(leaked_by_label) | set(total_by_label))
        if leaked_by_label.get(label, 0) or total_by_label.get(label, 0)
    }
    label_counts = (
        sum(counts[0] for counts in label_cells.values()),
        sum(counts[1] for counts in label_cells.values()),
    )
    if language_counts != label_counts:
        raise ValueError("language leakage counts do not match label counts")
    return {language: label_cells}


def _count_mapping(
    value: Any,
    *,
    context: str,
    keys: str,
) -> dict[str, int]:
    mapping = _as_mapping(value)
    if mapping is None:
        raise ValueError(f"{context} is required")
    result: dict[str, int] = {}
    for raw_key, raw_count in mapping.items():
        key = _language(raw_key) if keys == "language" else _canonical_label(raw_key)
        result[key] = _count(raw_count, field=context)
    return result


def _counts(value: Mapping[str, Any], *, context: str) -> tuple[int, int]:
    leaked = _count(value.get("leaked_chars"), field=f"{context}.leaked_chars")
    total = _count(value.get("total_chars"), field=f"{context}.total_chars")
    if leaked > total:
        raise ValueError(f"{context} leaked_chars cannot exceed total_chars")
    return leaked, total


def _count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative integer")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a non-negative integer")
    count = int(value)
    if count != value or count < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return count


def _canonical_label(value: Any) -> str:
    label = str(value)
    if label not in CANONICAL_LABELS:
        raise ValueError(f"unsupported leakage label: {label!r}")
    return label


def _language(value: Any) -> str:
    language = str(value)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported PII language: {language!r}")
    return language


def _resolve_thresholds(
    thresholds: float | Mapping[str, float] | None,
) -> dict[str, float]:
    if thresholds is None:
        return {}
    if isinstance(thresholds, Mapping):
        default = thresholds.get("*")
        result = {
            language: _unit_rate(default, field="thresholds.*")
            for language in sorted(SUPPORTED_LANGUAGES)
            if default is not None
        }
        for raw_language, raw_value in thresholds.items():
            if raw_language == "*":
                continue
            language = _language(raw_language)
            result[language] = _unit_rate(
                raw_value,
                field=f"thresholds.{language}",
            )
        return dict(sorted(result.items()))
    value = _unit_rate(thresholds, field="thresholds")
    return {language: value for language in sorted(SUPPORTED_LANGUAGES)}


def _unit_rate(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be between 0 and 1")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return rate


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _safe_rate(leaked_chars: int, total_chars: int) -> float:
    if total_chars == 0:
        return 0.0
    return leaked_chars / total_chars


def _summary_row(summary: LanguageLeakageSummary) -> str:
    top_labels = ", ".join(item.label for item in summary.worst_labels) or "None"
    return "".join(
        [
            (
                f'<tr data-language="{_escape(summary.language)}" '
                f'data-rate="{summary.rate:.17g}" '
                f'data-residual="{summary.leaked_chars}" '
                f'data-total="{summary.total_chars}">'
            ),
            (
                f'<td><a href="#language-{_escape(summary.language)}">'
                f"{_escape(summary.language)}</a></td>"
            ),
            f"<td>{summary.rate:.6f}</td>",
            f"<td>{summary.leaked_chars}</td>",
            f"<td>{summary.total_chars}</td>",
            f"<td>{_escape(top_labels)}</td>",
            f"<td>{_escape(_gate_text(summary.threshold_passed))}</td>",
            "</tr>",
        ]
    )


def _language_panel(summary: LanguageLeakageSummary) -> str:
    residual_rows = [
        [
            item.label,
            str(item.leaked_chars),
            str(item.total_chars),
            f"{item.rate:.6f}",
        ]
        for item in summary.residual_by_label
    ]
    worst_rows = [
        [
            str(rank),
            item.label,
            str(item.leaked_chars),
            str(item.total_chars),
            f"{item.rate:.6f}",
        ]
        for rank, item in enumerate(summary.worst_labels, start=1)
    ]
    trend_rows = [
        [
            str(point.run_index),
            str(point.leaked_chars),
            str(point.total_chars),
            f"{point.rate:.6f}",
        ]
        for point in summary.trend
    ]
    return "\n".join(
        [
            (
                f'<section class="language-panel" '
                f'id="language-{_escape(summary.language)}">'
            ),
            f"<h2>Language: {_escape(summary.language)}</h2>",
            '<div class="metric-grid">',
            _metric("Leakage", f"{summary.rate:.6f}"),
            _metric("Residual chars", str(summary.leaked_chars)),
            _metric("Measured PHI chars", str(summary.total_chars)),
            _metric("Gate", _gate_text(summary.threshold_passed)),
            "</div>",
            "<h3>Leakage trend</h3>",
            _table(
                ["Run offset", "Residual chars", "Measured PHI chars", "Leakage"],
                trend_rows,
            ),
            "<h3>Residual-PHI categories</h3>",
            _table_or_empty(
                ["Canonical label", "Residual chars", "Measured chars", "Leakage"],
                residual_rows,
                "No measured PHI categories for this language.",
            ),
            "<h3>Worst labels</h3>",
            _table_or_empty(
                [
                    "Rank",
                    "Canonical label",
                    "Residual chars",
                    "Measured chars",
                    "Leakage",
                ],
                worst_rows,
                "No residual leakage for this language.",
            ),
            "</section>",
        ]
    )


def _metric(label: str, value: str) -> str:
    return "".join(
        [
            '<article class="metric">',
            f'<span class="metric-label">{_escape(label)}</span>',
            f'<strong class="metric-value">{_escape(value)}</strong>',
            "</article>",
        ]
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            '<div class="table-scroll"><table>',
            "<thead><tr>",
            *[f"<th>{_escape(header)}</th>" for header in headers],
            "</tr></thead>",
            "<tbody>",
            *[
                "<tr>"
                + "".join(f"<td>{_escape(value)}</td>" for value in row)
                + "</tr>"
                for row in rows
            ],
            "</tbody>",
            "</table></div>",
        ]
    )


def _table_or_empty(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_escape(empty)}</p>'
    return _table(headers, rows)


def _sortable_header(label: str, key: str) -> str:
    return (
        '<th scope="col">'
        f'<button type="button" class="sort" data-sort="{_escape(key)}">'
        f"{_escape(label)}</button></th>"
    )


def _gate_text(value: bool | None) -> str:
    if value is None:
        return "Not configured"
    return "Pass" if value else "Fail"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


_CSS = """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background: #f7f8fa;
  color: #1f2933;
}
* { box-sizing: border-box; }
body { margin: 0; background: #f7f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }
header { margin-bottom: 24px; }
h1, h2, h3 { line-height: 1.2; }
h1 { margin: 0; font-size: 2rem; }
h2 { margin-top: 0; }
h3 { margin: 24px 0 10px; font-size: 1rem; }
section { margin-top: 24px; }
.language-panel { padding-top: 24px; border-top: 1px solid #d8dde6; }
.subtle { color: #526170; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 9px 11px; border: 1px solid #d8dde6; text-align: left; }
th { background: #eef2f6; color: #344054; font-size: .8rem; }
.sort { padding: 0; border: 0; background: transparent; color: inherit;
  font: inherit; font-weight: 700; cursor: pointer; }
.sort::after { content: " ↕"; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.metric { min-height: 88px; padding: 14px; background: #fff; border: 1px solid #d8dde6; }
.metric-label { display: block; color: #526170; font-size: .82rem; }
.metric-value { display: block; margin-top: 8px; font-size: 1.45rem; }
.empty { padding: 11px; background: #fff; border: 1px solid #d8dde6; color: #526170; }
a { color: #175cd3; }
""".strip()


_SORT_SCRIPT = """
(() => {
  const table = document.getElementById("leakage-summary-table");
  const body = table.querySelector("tbody");
  let activeKey = "language";
  let direction = 1;
  table.querySelectorAll("button[data-sort]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      direction = key === activeKey ? -direction : 1;
      activeKey = key;
      const numeric = key !== "language";
      const rows = Array.from(body.querySelectorAll("tr"));
      rows.sort((left, right) => {
        const a = left.dataset[key];
        const b = right.dataset[key];
        const comparison = numeric
          ? Number(a) - Number(b)
          : a.localeCompare(b);
        return comparison * direction;
      });
      rows.forEach((row) => body.appendChild(row));
    });
  });
})();
""".strip()
