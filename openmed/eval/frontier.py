"""Throughput-versus-accuracy Pareto frontier report.

Device tiers trade speed for quality: a smaller or more aggressively quantized
variant runs more documents per second but usually redacts less accurately (and
may leak more). The eval harness already measures both sides -- throughput lives
in :class:`~openmed.eval.perf.PerfReport` and accuracy/leakage live in
:class:`~openmed.eval.report.BenchmarkReport` -- but there is no combined view
that says *which* variants are actually worth shipping.

This module answers that question. Given a set of measured configuration points
(system/variant, throughput in docs/sec, an accuracy metric, and a leakage rate)
it computes the Pareto-optimal (non-dominated) set on the speed-vs-quality plane
and flags every dominated point with the point that dominates it. It reuses the
already-measured numbers -- it never re-runs a benchmark -- and serializes
deterministically to JSON and Markdown like the other eval reports.

Optimization sense
------------------
- ``throughput`` (docs/sec) -- higher is better.
- ``accuracy`` (F1 / recall, in ``[0, 1]``) -- higher is better.
- ``leakage`` (leaked-character rate, in ``[0, 1]``) -- lower is better, so it is
  compared as ``-leakage``. Leakage evidence is required; an unmeasured variant
  cannot appear safer than a measured one.

A point *B* dominates a point *A* when *B* is at least as good as *A* on every
objective and strictly better on at least one. Dominated points are off the
frontier; every point that nothing dominates is on the frontier.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Default keys, in priority order, used to pull an accuracy scalar out of a
# ``BenchmarkReport.metrics`` bundle. All are "higher is better".
DEFAULT_ACCURACY_KEYS: tuple[str, ...] = (
    "exact_span_f1.f1",
    "relaxed_span_f1.f1",
    "character_recall.overall",
    "f1",
)

# Default keys, in priority order, used to pull a leakage scalar out of a
# ``BenchmarkReport.metrics`` bundle. All are "lower is better".
DEFAULT_LEAKAGE_KEYS: tuple[str, ...] = (
    "leakage.overall",
    "leakage_rate.overall",
    "leakage",
)


@dataclass(frozen=True)
class FrontierPoint:
    """One measured configuration on the throughput-vs-accuracy plane.

    Attributes:
        label: Stable identifier for the system/variant (e.g.
            ``"clinical-e5-small@int8"``). Used to reference dominators.
        throughput: Documents processed per second (higher is better).
        accuracy: Quality metric such as span F1 or character recall in
            ``[0, 1]`` (higher is better).
        leakage: Measured leaked-character rate in ``[0, 1]`` (lower is better).
            This evidence is required so missing measurements cannot be treated
            as perfect zero leakage.
        metadata: Optional provenance (model, threshold, quantization, batch,
            device, tier, source report ids). Never holds raw PHI.
    """

    label: str
    throughput: float
    accuracy: float
    leakage: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize a measured point at every construction path."""
        object.__setattr__(self, "label", _coerce_label(self.label))
        object.__setattr__(
            self,
            "throughput",
            _coerce_nonnegative(self.throughput, field_name="throughput"),
        )
        object.__setattr__(
            self,
            "accuracy",
            _coerce_unit_interval(self.accuracy, field_name="accuracy"),
        )
        object.__setattr__(
            self,
            "leakage",
            _coerce_unit_interval(self.leakage, field_name="leakage"),
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def objectives(self) -> tuple[float, float, float]:
        """Return objectives as a maximization tuple ``(tput, acc, -leak)``."""
        return (self.throughput, self.accuracy, -self.leakage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable point dictionary with stable keys."""
        return {
            "accuracy": self.accuracy,
            "label": self.label,
            "leakage": self.leakage,
            "metadata": dict(self.metadata),
            "throughput": self.throughput,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrontierPoint":
        """Build a point from a JSON-compatible mapping."""
        raw_label = data.get("label")
        if raw_label is None:
            raw_label = data.get("variant")
        if raw_label is None:
            raw_label = data.get("system")
        label = _coerce_label(raw_label)
        throughput = _coerce_nonnegative(
            data.get("throughput", data.get("docs_per_second")),
            field_name="throughput",
        )
        accuracy = _coerce_unit_interval(data.get("accuracy"), field_name="accuracy")
        if data.get("leakage") is None:
            raise ValueError("leakage evidence is required")
        leakage = _coerce_unit_interval(data.get("leakage"), field_name="leakage")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            label=label,
            throughput=throughput,
            accuracy=accuracy,
            leakage=leakage,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class FrontierEntry:
    """A point plus its computed frontier status."""

    point: FrontierPoint
    on_frontier: bool
    dominated_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entry dictionary with stable keys."""
        return {
            "dominated_by": self.dominated_by,
            "on_frontier": self.on_frontier,
            "point": self.point.to_dict(),
        }


@dataclass(frozen=True)
class FrontierReport:
    """Throughput-versus-accuracy Pareto frontier over measured configs."""

    entries: Sequence[FrontierEntry]
    accuracy_metric: str = "accuracy"
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frontier(self) -> list[FrontierEntry]:
        """Return only the non-dominated entries, in report order."""
        return [entry for entry in self.entries if entry.on_frontier]

    @property
    def dominated(self) -> list[FrontierEntry]:
        """Return only the dominated entries, in report order."""
        return [entry for entry in self.entries if not entry.on_frontier]

    @property
    def point_count(self) -> int:
        """Return the total number of measured configuration points."""
        return len(self.entries)

    @property
    def frontier_count(self) -> int:
        """Return the number of non-dominated (on-frontier) points."""
        return len(self.frontier)

    @property
    def dominated_count(self) -> int:
        """Return the number of dominated (off-frontier) points."""
        return len(self.dominated)

    def chart_data(self) -> dict[str, Any]:
        """Return a stdlib-only structure for a simple speed-vs-quality plot.

        The payload names the axes and splits the measured points into the
        connected frontier line (sorted by throughput) and the scattered
        dominated points, so a caller can render it with any plotting tool
        without OpenMed taking on a plotting dependency.
        """
        frontier_series = sorted(
            (entry.point for entry in self.frontier),
            key=lambda point: (point.throughput, point.accuracy),
        )
        return {
            "x_axis": {"key": "throughput", "label": "Throughput (docs/sec)"},
            "y_axis": {"key": "accuracy", "label": self.accuracy_metric},
            "frontier": [
                {
                    "label": point.label,
                    "throughput": point.throughput,
                    "accuracy": point.accuracy,
                    "leakage": point.leakage,
                }
                for point in frontier_series
            ],
            "dominated": [
                {
                    "label": entry.point.label,
                    "throughput": entry.point.throughput,
                    "accuracy": entry.point.accuracy,
                    "leakage": entry.point.leakage,
                    "dominated_by": entry.dominated_by,
                }
                for entry in self.dominated
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable report dictionary."""
        return {
            "accuracy_metric": self.accuracy_metric,
            "chart_data": self.chart_data(),
            "dominated_count": self.dominated_count,
            "entries": [entry.to_dict() for entry in self.entries],
            "frontier_count": self.frontier_count,
            "frontier_labels": [entry.point.label for entry in self.frontier],
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
            "point_count": self.point_count,
            "schema_version": 1,
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
        """Serialize the report to deterministic Markdown."""
        lines = [
            "# Throughput vs Accuracy Frontier",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Accuracy Metric | `{self.accuracy_metric}` |",
            f"| Points | {len(self.entries)} |",
            f"| On Frontier | {len(self.frontier)} |",
            f"| Dominated | {len(self.dominated)} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Configurations",
                "",
                "| Variant | Throughput (docs/sec) | Accuracy | Leakage | Frontier"
                " | Dominated By |",
                "|---|---:|---:|---:|:---:|---|",
            ]
        )
        for entry in self.entries:
            point = entry.point
            status = "yes" if entry.on_frontier else "no"
            dominated_by = f"`{entry.dominated_by}`" if entry.dominated_by else ""
            lines.append(
                f"| `{point.label}` | {_format_number(point.throughput)} | "
                f"{_format_number(point.accuracy)} | "
                f"{_format_number(point.leakage)} | {status} | {dominated_by} |"
            )

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def frontier_report(
    points: Iterable[FrontierPoint | Mapping[str, Any]],
    *,
    accuracy_metric: str = "accuracy",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FrontierReport:
    """Compute the throughput-versus-accuracy Pareto frontier over *points*.

    Each point carries a throughput (docs/sec, higher is better), an accuracy
    metric (higher is better), and a measured leakage rate (lower is better).
    A point is *dominated* when another point is at least as good on every
    objective and strictly better on at least one; every point that nothing
    dominates is *on the frontier*.

    When two points have identical objectives they are considered mutual ties:
    neither dominates the other, so both stay on the frontier. Input order is
    preserved in the report, and the deterministic dominator chosen for a
    dominated point is the strongest available (best objectives, ties broken by
    label) so serialization is stable regardless of input order.

    Args:
        points: Measured configuration points, either :class:`FrontierPoint`
            instances or JSON-compatible mappings (see
            :meth:`FrontierPoint.from_mapping`).
        accuracy_metric: Human-readable name of the accuracy axis, recorded in
            the report and used as the chart's y-axis label.
        generated_at: Optional ISO-8601 timestamp for the report.
        metadata: Optional report-level provenance.

    Returns:
        A :class:`FrontierReport` with one entry per input point, each flagged
        on-frontier or dominated (with its dominator's label recorded).
    """
    resolved = [
        (
            point
            if isinstance(point, FrontierPoint)
            else FrontierPoint.from_mapping(point)
        )
        for point in points
    ]

    seen_labels: set[str] = set()
    for point in resolved:
        if point.label in seen_labels:
            raise ValueError(f"duplicate point label: {point.label!r}")
        seen_labels.add(point.label)

    entries: list[FrontierEntry] = []
    for index, candidate in enumerate(resolved):
        dominators = [
            other
            for other_index, other in enumerate(resolved)
            if other_index != index and _dominates(other, candidate)
        ]
        if dominators:
            dominator = _select_dominator(dominators)
            entries.append(
                FrontierEntry(
                    point=candidate,
                    on_frontier=False,
                    dominated_by=dominator.label,
                )
            )
        else:
            entries.append(FrontierEntry(point=candidate, on_frontier=True))

    return FrontierReport(
        entries=entries,
        accuracy_metric=accuracy_metric,
        generated_at=generated_at,
        metadata=dict(metadata or {}),
    )


def frontier_point_from_reports(
    perf: Any,
    benchmark: Any,
    *,
    label: str | None = None,
    accuracy_keys: Sequence[str] = DEFAULT_ACCURACY_KEYS,
    leakage_keys: Sequence[str] = DEFAULT_LEAKAGE_KEYS,
) -> FrontierPoint:
    """Assemble one :class:`FrontierPoint` from already-measured reports.

    Throughput is read from a :class:`~openmed.eval.perf.PerfReport` (its
    ``docs_per_second``) and accuracy/leakage are read from a
    :class:`~openmed.eval.report.BenchmarkReport`'s ``metrics`` bundle. Nothing
    is re-measured; this only reshapes numbers the harness already produced.

    Args:
        perf: A ``PerfReport`` (or any object exposing ``docs_per_second``).
        benchmark: A ``BenchmarkReport`` (or any object exposing a ``metrics``
            mapping).
        label: Variant label; defaults to the benchmark/perf ``model_name``.
        accuracy_keys: Dotted metric keys tried in order for the accuracy scalar.
        leakage_keys: Dotted metric keys tried in order for the leakage scalar.

    Returns:
        A :class:`FrontierPoint` combining the two reports.

    Raises:
        ValueError: If required evidence or a stable label is missing, or if a
            measured value is invalid.
    """
    throughput = _coerce_nonnegative(
        getattr(perf, "docs_per_second", None), field_name="docs_per_second"
    )
    metrics = getattr(benchmark, "metrics", None)
    if not isinstance(metrics, Mapping):
        raise ValueError("benchmark metrics must be a mapping")

    accuracy_value, accuracy_key = _first_metric(metrics, accuracy_keys)
    if accuracy_value is None:
        raise ValueError(
            "no accuracy metric found in benchmark report under keys: "
            f"{list(accuracy_keys)}"
        )
    leakage_value, leakage_key = _first_metric(metrics, leakage_keys)
    if leakage_value is None:
        raise ValueError(
            "no leakage metric found in benchmark report under keys: "
            f"{list(leakage_keys)}"
        )

    resolved_label = label
    if resolved_label is None:
        resolved_label = getattr(benchmark, "model_name", None)
    if resolved_label is None:
        resolved_label = getattr(perf, "model_name", None)
    if resolved_label is None:
        raise ValueError("label is required when reports have no model_name")

    metadata: dict[str, Any] = {"accuracy_key": accuracy_key}
    metadata["leakage_key"] = leakage_key
    for source, obj in (("perf", perf), ("benchmark", benchmark)):
        for attr in ("model_name", "device", "tier", "canonical_tier", "suite"):
            value = getattr(obj, attr, None)
            if value is not None:
                metadata[f"{source}_{attr}"] = value

    return FrontierPoint(
        label=resolved_label,
        throughput=throughput,
        accuracy=accuracy_value,
        leakage=leakage_value,
        metadata=metadata,
    )


def _dominates(better: FrontierPoint, worse: FrontierPoint) -> bool:
    """Return ``True`` if *better* Pareto-dominates *worse*.

    Domination requires *better* to be at least as good on every objective and
    strictly better on at least one. Identical objective tuples are ties and do
    not dominate either way.
    """
    lhs = better.objectives()
    rhs = worse.objectives()
    at_least_as_good = all(left >= right for left, right in zip(lhs, rhs))
    strictly_better = any(left > right for left, right in zip(lhs, rhs))
    return at_least_as_good and strictly_better


def _select_dominator(dominators: Sequence[FrontierPoint]) -> FrontierPoint:
    """Pick a deterministic, strongest dominator from *dominators*.

    Ordering is by objectives descending, with the label as a final tie-break so
    the choice does not depend on input order. Among equally strong dominators
    the lexicographically smallest label wins.
    """
    return min(
        dominators,
        key=lambda point: (
            tuple(-value for value in point.objectives()),
            point.label,
        ),
    )


def _first_metric(
    metrics: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[float | None, str | None]:
    """Return the first numeric metric found under dotted *keys*."""
    for key in keys:
        value = _nested_get(metrics, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value), key
    return None, None


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted key path against a nested mapping."""
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_float(value: Any, *, field_name: str) -> float:
    """Coerce *value* to a finite ``float`` or raise a clear error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def _coerce_nonnegative(value: Any, *, field_name: str) -> float:
    """Coerce a finite, non-negative measured value."""
    result = _coerce_float(value, field_name=field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return result


def _coerce_unit_interval(value: Any, *, field_name: str) -> float:
    """Coerce a finite metric in the closed unit interval."""
    result = _coerce_float(value, field_name=field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _coerce_label(value: Any) -> str:
    """Return a non-empty stable label or reject ambiguous identifiers."""
    if not isinstance(value, str):
        raise ValueError("label must be a non-empty string")
    label = value.strip()
    if not label:
        raise ValueError("label must be a non-empty string")
    if "\n" in label or "\r" in label:
        raise ValueError("label must not contain line breaks")
    return label


def _format_number(value: float) -> str:
    """Format a metric for stable Markdown output."""
    return f"{value:.6g}"


__all__ = [
    "DEFAULT_ACCURACY_KEYS",
    "DEFAULT_LEAKAGE_KEYS",
    "FrontierEntry",
    "FrontierPoint",
    "FrontierReport",
    "frontier_point_from_reports",
    "frontier_report",
]
