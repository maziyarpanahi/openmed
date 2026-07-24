"""Regression-escape tracking across ordered benchmark release reports.

The tracker deliberately emits metric-only evidence. Source report metadata,
fixture payloads, predictions, and record-level values are never copied into
the JSON summary or dashboard.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openmed.eval.history import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    REGRESSION,
    diff_against_baseline,
)
from openmed.eval.report import BenchmarkReport

GATED = "gated"
ESCAPED = "escaped"
RECOVERED = "recovered"
REGRESSION_TRACKER_SCHEMA_VERSION = "openmed.regression_tracker.v1"

_DEFAULT_TITLE = "OpenMed Regression Escape Tracker"
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}\Z")
_GATED_STATUSES = frozenset({"blocked", "failed", "gated", "rejected"})
_RELEASED_STATUSES = frozenset(
    {"approved", "escaped", "passed", "published", "released", "shipped"}
)

ReportLike = BenchmarkReport | Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class RegressionEvent:
    """One gated, open escaped, or recovered metric regression."""

    event_id: str
    suite: str
    metric: str
    direction: str
    classification: str
    baseline_release: str
    detected_release: str
    baseline_value: float
    detected_value: float
    delta: float
    relative_delta: float | None = None
    recovery_release: str | None = None
    recovery_value: float | None = None
    time_to_recovery_releases: int | None = None
    time_to_recovery_days: int | None = None

    @property
    def escaped(self) -> bool:
        """Return whether the regression reached a shipped release."""

        return self.classification in {ESCAPED, RECOVERED}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready event payload."""

        return {
            "baseline_release": self.baseline_release,
            "baseline_value": self.baseline_value,
            "classification": self.classification,
            "delta": self.delta,
            "detected_release": self.detected_release,
            "detected_value": self.detected_value,
            "direction": self.direction,
            "escaped": self.escaped,
            "event_id": self.event_id,
            "metric": self.metric,
            "recovery_release": self.recovery_release,
            "recovery_value": self.recovery_value,
            "relative_delta": self.relative_delta,
            "suite": self.suite,
            "time_to_recovery_days": self.time_to_recovery_days,
            "time_to_recovery_releases": self.time_to_recovery_releases,
        }


@dataclass(frozen=True)
class RegressionHotspot:
    """Aggregate recurrence and recovery evidence for one suite metric."""

    suite: str
    metric: str
    regression_count: int
    escape_count: int
    gated_count: int
    recovered_count: int
    open_escape_count: int
    mean_time_to_recovery_releases: float | None = None
    mean_time_to_recovery_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready hotspot payload."""

        return {
            "escape_count": self.escape_count,
            "gated_count": self.gated_count,
            "mean_time_to_recovery_days": self.mean_time_to_recovery_days,
            "mean_time_to_recovery_releases": (self.mean_time_to_recovery_releases),
            "metric": self.metric,
            "open_escape_count": self.open_escape_count,
            "recovered_count": self.recovered_count,
            "regression_count": self.regression_count,
            "suite": self.suite,
        }


@dataclass(frozen=True)
class RegressionTrackerSummary:
    """Release-readiness summary for regression escapes over time."""

    release_count: int
    events: tuple[RegressionEvent, ...] = field(default_factory=tuple)
    hotspots: tuple[RegressionHotspot, ...] = field(default_factory=tuple)
    schema_version: str = REGRESSION_TRACKER_SCHEMA_VERSION

    @property
    def gated_count(self) -> int:
        """Return the number of regressions stopped by a gate."""

        return sum(event.classification == GATED for event in self.events)

    @property
    def escape_count(self) -> int:
        """Return the number of regressions that reached a release."""

        return sum(event.escaped for event in self.events)

    @property
    def recovered_count(self) -> int:
        """Return the number of escaped regressions later recovered."""

        return sum(event.classification == RECOVERED for event in self.events)

    @property
    def open_escape_count(self) -> int:
        """Return the number of escaped regressions still unresolved."""

        return sum(event.classification == ESCAPED for event in self.events)

    @property
    def release_ready(self) -> bool:
        """Return whether no unresolved regression escapes remain."""

        return self.open_escape_count == 0

    @property
    def mean_time_to_recovery_releases(self) -> float | None:
        """Return mean shipped-release distance for recovered escapes."""

        return _mean(
            event.time_to_recovery_releases
            for event in self.events
            if event.classification == RECOVERED
        )

    @property
    def mean_time_to_recovery_days(self) -> float | None:
        """Return mean calendar days for recoveries with dated releases."""

        return _mean(
            event.time_to_recovery_days
            for event in self.events
            if event.classification == RECOVERED
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON summary for release-readiness checks."""

        return {
            "escape_count": self.escape_count,
            "events": [event.to_dict() for event in self.events],
            "gated_count": self.gated_count,
            "hotspots": [hotspot.to_dict() for hotspot in self.hotspots],
            "mean_time_to_recovery_days": self.mean_time_to_recovery_days,
            "mean_time_to_recovery_releases": (self.mean_time_to_recovery_releases),
            "open_escape_count": self.open_escape_count,
            "recovered_count": self.recovered_count,
            "regression_count": len(self.events),
            "release_count": self.release_count,
            "release_ready": self.release_ready,
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the summary to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic release-readiness JSON and return its path."""

        output_path = Path(path)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


@dataclass
class _EventState:
    suite: str
    metric: str
    direction: str
    gated: bool
    baseline_release: str
    detected_release: str
    baseline_value: float
    detected_value: float
    delta: float
    relative_delta: float | None
    detected_release_index: int
    detected_date: date | None
    recovery_release: str | None = None
    recovery_value: float | None = None
    recovery_release_index: int | None = None
    recovery_date: date | None = None

    def finalize(self, event_id: str) -> RegressionEvent:
        if self.gated:
            classification = GATED
        elif self.recovery_release is not None:
            classification = RECOVERED
        else:
            classification = ESCAPED

        release_distance = None
        if self.recovery_release_index is not None:
            release_distance = self.recovery_release_index - self.detected_release_index

        day_distance = None
        if self.detected_date is not None and self.recovery_date is not None:
            day_distance = (self.recovery_date - self.detected_date).days

        return RegressionEvent(
            event_id=event_id,
            suite=self.suite,
            metric=self.metric,
            direction=self.direction,
            classification=classification,
            baseline_release=self.baseline_release,
            detected_release=self.detected_release,
            baseline_value=self.baseline_value,
            detected_value=self.detected_value,
            delta=self.delta,
            relative_delta=self.relative_delta,
            recovery_release=self.recovery_release,
            recovery_value=self.recovery_value,
            time_to_recovery_releases=release_distance,
            time_to_recovery_days=day_distance,
        )


@dataclass(frozen=True)
class _ReleaseObservation:
    payload: Mapping[str, Any]
    raw_suite: str
    suite: str
    release: str
    release_index: int
    released_on: date | None
    gated: bool
    metrics: Mapping[str, float]


def track_regression_escapes(
    reports_in_order: Iterable[ReportLike],
    *,
    metric_directions: Mapping[str, str] | None = None,
    regression_thresholds: Mapping[str, float] | None = None,
) -> RegressionTrackerSummary:
    """Track gated, escaped, and recovered regressions over release history.

    Reports must be provided in release order. Multiple suite reports may share
    a release identifier. The identifier is read from ``release`` or ``version``
    at the top level or in ``metadata``; ISO ``released``/``generated_at`` values
    and finally a deterministic sequence identifier are fallbacks.

    A candidate is considered gated when ``gate_passed`` is false,
    ``released`` is false, or ``gate_status``/``release_status`` is one of
    ``failed``, ``blocked``, ``gated``, or ``rejected``. Missing gate metadata
    means the historical report represents a shipped release.

    Args:
        reports_in_order: Benchmark reports or JSON report paths in release order.
        metric_directions: Optional exact flattened metric direction overrides.
        regression_thresholds: Optional absolute deltas to ignore per metric.

    Returns:
        A deterministic, metric-only release-readiness summary.
    """

    thresholds = _validate_thresholds(regression_thresholds)
    observations, release_count = _observations(reports_in_order)
    last_shipped: dict[str, _ReleaseObservation] = {}
    open_events: dict[tuple[str, str], list[_EventState]] = defaultdict(list)
    event_states: list[_EventState] = []

    for current in observations:
        previous = last_shipped.get(current.raw_suite)
        if previous is None:
            if not current.gated:
                last_shipped[current.raw_suite] = current
            continue

        if not current.gated:
            _recover_open_events(current, open_events, thresholds)

        diff = diff_against_baseline(
            current.payload,
            previous.payload,
            metric_directions=metric_directions,
            rank_limit=None,
        )
        for raw_metric, delta in sorted(diff.metrics.items()):
            if delta.verdict != REGRESSION:
                continue
            if abs(delta.delta) <= thresholds.get(raw_metric, 0.0):
                continue
            event = _EventState(
                suite=current.suite,
                metric=_safe_output_identifier(raw_metric, "metric"),
                direction=delta.direction,
                gated=current.gated,
                baseline_release=previous.release,
                detected_release=current.release,
                baseline_value=delta.baseline,
                detected_value=delta.current,
                delta=delta.delta,
                relative_delta=delta.relative_delta,
                detected_release_index=current.release_index,
                detected_date=current.released_on,
            )
            event_states.append(event)
            if not current.gated:
                open_events[(current.raw_suite, raw_metric)].append(event)

        if not current.gated:
            last_shipped[current.raw_suite] = current

    events = tuple(
        event.finalize(f"regression-{index:04d}")
        for index, event in enumerate(event_states, start=1)
    )
    return RegressionTrackerSummary(
        release_count=release_count,
        events=events,
        hotspots=_rank_hotspots(events),
    )


def render_regression_dashboard(summary: RegressionTrackerSummary) -> str:
    """Render a deterministic, self-contained regression tracker dashboard."""

    cards = [
        ("Releases analyzed", str(summary.release_count)),
        ("Regressions", str(len(summary.events))),
        ("Escaped", str(summary.escape_count)),
        ("Open escapes", str(summary.open_escape_count)),
        ("Recovered", str(summary.recovered_count)),
        (
            "Mean recovery (releases)",
            _display_number(summary.mean_time_to_recovery_releases),
        ),
        (
            "Mean recovery (days)",
            _display_number(summary.mean_time_to_recovery_days),
        ),
    ]
    card_html = "\n".join(
        "\n".join(
            [
                '<article class="metric-card">',
                f'<span class="metric-label">{html.escape(label)}</span>',
                f'<strong class="metric-value">{html.escape(value)}</strong>',
                "</article>",
            ]
        )
        for label, value in cards
    )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{_DEFAULT_TITLE}</title>",
            "<style>",
            _DASHBOARD_CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            f"<h1>{_DEFAULT_TITLE}</h1>",
            '<p class="subtle">Metric-only release readiness and recovery evidence.</p>',
            "</header>",
            '<section aria-labelledby="readiness">',
            '<h2 id="readiness">Release readiness</h2>',
            f'<p class="readiness {_readiness_class(summary)}">',
            "READY" if summary.release_ready else "BLOCKED",
            "</p>",
            '<div class="metric-grid">',
            card_html,
            "</div>",
            "</section>",
            '<section aria-labelledby="hotspots">',
            '<h2 id="hotspots">Recurring-regression hotspots</h2>',
            _hotspot_table(summary.hotspots),
            "</section>",
            '<section aria-labelledby="events">',
            '<h2 id="events">Regression history</h2>',
            _event_table(summary.events),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_regression_dashboard(
    summary: RegressionTrackerSummary,
    path: str | Path,
) -> Path:
    """Write a rendered regression tracker dashboard and return its path."""

    output_path = Path(path)
    output_path.write_text(render_regression_dashboard(summary), encoding="utf-8")
    return output_path


def _observations(
    reports_in_order: Iterable[ReportLike],
) -> tuple[list[_ReleaseObservation], int]:
    observations: list[_ReleaseObservation] = []
    release_indexes: dict[str, int] = {}
    closed_releases: set[str] = set()
    previous_release: str | None = None

    for report_index, report in enumerate(reports_in_order):
        payload = _report_payload(report)
        metadata = _mapping(payload.get("metadata"))
        raw_suite = _required_identifier(payload.get("suite"), "suite")
        raw_release = _release_identifier(payload, metadata, report_index)
        release = _safe_output_identifier(raw_release, "release")
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("benchmark report metrics must be an object")

        if raw_release != previous_release:
            if raw_release in release_indexes:
                raise ValueError(
                    f"release reports must be contiguous in input order: {release}"
                )
            if previous_release is not None:
                closed_releases.add(previous_release)
            release_indexes[raw_release] = len(release_indexes)
            previous_release = raw_release
        elif raw_release in closed_releases:
            raise ValueError(
                f"release reports must be contiguous in input order: {release}"
            )

        observations.append(
            _ReleaseObservation(
                payload=payload,
                raw_suite=raw_suite,
                suite=_safe_output_identifier(raw_suite, "suite"),
                release=release,
                release_index=release_indexes[raw_release],
                released_on=_release_date(payload, metadata),
                gated=_is_gated(payload, metadata),
                metrics=_numeric_metrics(raw_metrics),
            )
        )
    return observations, len(release_indexes)


def _recover_open_events(
    current: _ReleaseObservation,
    open_events: Mapping[tuple[str, str], list[_EventState]],
    thresholds: Mapping[str, float],
) -> None:
    for raw_metric, value in current.metrics.items():
        events = open_events.get((current.raw_suite, raw_metric), ())
        tolerance = thresholds.get(raw_metric, 0.0)
        for event in events:
            if event.recovery_release is not None:
                continue
            if not _meets_baseline(
                value,
                event.baseline_value,
                event.direction,
                tolerance,
            ):
                continue
            event.recovery_release = current.release
            event.recovery_value = value
            event.recovery_release_index = current.release_index
            event.recovery_date = current.released_on


def _rank_hotspots(
    events: Sequence[RegressionEvent],
) -> tuple[RegressionHotspot, ...]:
    grouped: dict[tuple[str, str], list[RegressionEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.suite, event.metric)].append(event)

    hotspots = []
    for (suite, metric), rows in grouped.items():
        recovered = [row for row in rows if row.classification == RECOVERED]
        hotspots.append(
            RegressionHotspot(
                suite=suite,
                metric=metric,
                regression_count=len(rows),
                escape_count=sum(row.escaped for row in rows),
                gated_count=sum(row.classification == GATED for row in rows),
                recovered_count=len(recovered),
                open_escape_count=sum(row.classification == ESCAPED for row in rows),
                mean_time_to_recovery_releases=_mean(
                    row.time_to_recovery_releases for row in recovered
                ),
                mean_time_to_recovery_days=_mean(
                    row.time_to_recovery_days for row in recovered
                ),
            )
        )

    return tuple(
        sorted(
            hotspots,
            key=lambda row: (
                -row.escape_count,
                -row.regression_count,
                -row.open_escape_count,
                row.suite,
                row.metric,
            ),
        )
    )


def _report_payload(report: ReportLike) -> dict[str, Any]:
    if isinstance(report, BenchmarkReport):
        return report.to_dict()
    if isinstance(report, (str, Path)):
        path = Path(report)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"benchmark release report must be an object: {path}")
        return dict(payload)
    if hasattr(report, "to_dict") and callable(report.to_dict):
        payload = report.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(report, Mapping):
        return dict(report)
    raise TypeError(f"unsupported benchmark release report: {type(report).__name__}")


def _release_identifier(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    report_index: int,
) -> str:
    for key in ("release", "version"):
        value = metadata.get(key, payload.get(key))
        if value is not None and value != "":
            return str(value)
    for value in (
        metadata.get("released"),
        payload.get("released"),
        payload.get("generated_at"),
    ):
        if isinstance(value, str) and value:
            return value
    return f"release-{report_index + 1}"


def _release_date(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> date | None:
    for key in ("released_at", "released"):
        value = metadata.get(key, payload.get(key))
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return _parse_date(payload.get("generated_at"))


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None


def _is_gated(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    for source in (payload, metadata):
        if "gate_passed" in source:
            gate_passed = source["gate_passed"]
            if not isinstance(gate_passed, bool):
                raise ValueError("gate_passed must be a boolean")
            return not gate_passed

        for key in ("gate_status", "release_status"):
            if key not in source:
                continue
            status = str(source[key]).strip().lower()
            if status in _GATED_STATUSES:
                return True
            if status in _RELEASED_STATUSES:
                return False
            raise ValueError(f"unsupported {key}: {source[key]}")

        if isinstance(source.get("released"), bool):
            return not source["released"]
    return False


def _numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, Mapping):
        metrics: dict[str, float] = {}
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_numeric_metrics(value[key], child_prefix))
        return metrics
    if _is_number(value) and prefix:
        return {prefix: float(value)}
    return {}


def _validate_thresholds(
    thresholds: Mapping[str, float] | None,
) -> dict[str, float]:
    if thresholds is None:
        return {}
    validated = {}
    for metric, value in thresholds.items():
        if not _is_number(value) or float(value) < 0:
            raise ValueError(f"regression threshold must be non-negative: {metric}")
        validated[str(metric)] = float(value)
    return validated


def _meets_baseline(
    current: float,
    baseline: float,
    direction: str,
    tolerance: float,
) -> bool:
    if direction == HIGHER_IS_BETTER:
        return current >= baseline - tolerance
    if direction == LOWER_IS_BETTER:
        return current <= baseline + tolerance
    raise ValueError(f"unsupported metric direction: {direction}")


def _safe_output_identifier(value: str, kind: str) -> str:
    if _SAFE_IDENTIFIER.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}"


def _required_identifier(value: Any, kind: str) -> str:
    if value is None or value == "":
        raise ValueError(f"benchmark report {kind} is required")
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Iterable[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _readiness_class(summary: RegressionTrackerSummary) -> str:
    return "ready" if summary.release_ready else "blocked"


def _display_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _display_optional(value: Any) -> str:
    if value is None:
        return "—"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for label in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(_display_optional(value))}</td>" for value in row
        )
        body.append(f"<tr>{cells}</tr>")
    return "\n".join(
        [
            "<table>",
            f"<thead><tr>{header}</tr></thead>",
            "<tbody>",
            *body,
            "</tbody>",
            "</table>",
        ]
    )


def _hotspot_table(hotspots: Sequence[RegressionHotspot]) -> str:
    if not hotspots:
        return '<p class="empty">No regression hotspots detected.</p>'
    return _table(
        [
            "Suite",
            "Metric",
            "Regressions",
            "Escaped",
            "Open",
            "Recovered",
            "Mean recovery",
        ],
        [
            [
                row.suite,
                row.metric,
                row.regression_count,
                row.escape_count,
                row.open_escape_count,
                row.recovered_count,
                _display_number(row.mean_time_to_recovery_releases),
            ]
            for row in hotspots
        ],
    )


def _event_table(events: Sequence[RegressionEvent]) -> str:
    if not events:
        return '<p class="empty">No regressions detected.</p>'
    return _table(
        [
            "Event",
            "Suite",
            "Metric",
            "Status",
            "Detected",
            "Recovered",
            "Recovery releases",
            "Recovery days",
        ],
        [
            [
                event.event_id,
                event.suite,
                event.metric,
                event.classification,
                event.detected_release,
                event.recovery_release,
                event.time_to_recovery_releases,
                event.time_to_recovery_days,
            ]
            for event in events
        ],
    )


_DASHBOARD_CSS = """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: #f6f8fb;
  color: #172033;
}
* { box-sizing: border-box; }
body { margin: 0; background: #f6f8fb; }
main { max-width: 1160px; margin: 0 auto; padding: 32px 24px 48px; }
header { margin-bottom: 24px; }
h1, h2 { margin: 0; line-height: 1.2; }
h1 { font-size: 2rem; }
h2 { margin-bottom: 12px; font-size: 1.1rem; }
section { margin-top: 24px; }
.subtle { color: #596579; }
.readiness { display: inline-block; padding: 7px 11px; font-weight: 750; }
.readiness.ready { color: #146c43; background: #dff5e9; }
.readiness.blocked { color: #9f2d20; background: #fde7e3; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
  gap: 12px;
}
.metric-card { padding: 16px; border: 1px solid #d9dfeb; background: #fff; }
.metric-label { display: block; color: #596579; font-size: .8rem; }
.metric-value { display: block; margin-top: 7px; font-size: 1.7rem; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 10px 12px; border: 1px solid #e2e6ee; text-align: left; }
th { background: #edf1f7; font-size: .78rem; }
td { overflow-wrap: anywhere; }
.empty { padding: 12px; border: 1px solid #d9dfeb; background: #fff; }
""".strip()


__all__ = [
    "ESCAPED",
    "GATED",
    "RECOVERED",
    "REGRESSION_TRACKER_SCHEMA_VERSION",
    "RegressionEvent",
    "RegressionHotspot",
    "RegressionTrackerSummary",
    "render_regression_dashboard",
    "track_regression_escapes",
    "write_regression_dashboard",
]
