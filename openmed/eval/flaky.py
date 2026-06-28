"""Repeated-run variance checks for benchmark metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

DEFAULT_FLAKY_TOLERANCE = 1e-12

RunCallable = Callable[[], Any]
Tolerance = float | Mapping[str, float]


@dataclass(frozen=True)
class FlakyMetricReport:
    """Variance evidence for one metric across repeated runs."""

    metric: str
    minimum: float
    maximum: float
    spread: float
    tolerance: float
    stable: bool
    values: tuple[float, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        """Return the stable/flaky verdict."""
        return "stable" if self.stable else "flaky"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metric report."""
        return {
            "max": self.maximum,
            "metric": self.metric,
            "min": self.minimum,
            "spread": self.spread,
            "stable": self.stable,
            "tolerance": self.tolerance,
            "values": list(self.values),
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class FlakyReport:
    """Repeated-run stability report for a scoring callable."""

    n_runs: int
    metrics: Mapping[str, FlakyMetricReport] = field(default_factory=dict)

    @property
    def stable(self) -> bool:
        """Return whether every metric stayed within tolerance."""
        return all(metric.stable for metric in self.metrics.values())

    @property
    def verdict(self) -> str:
        """Return the overall stable/flaky verdict."""
        return "stable" if self.stable else "flaky"

    @property
    def flaky_metrics(self) -> tuple[str, ...]:
        """Return metric names whose spread exceeded tolerance."""
        return tuple(name for name, metric in self.metrics.items() if not metric.stable)

    def metric(self, name: str) -> FlakyMetricReport:
        """Return the named metric report."""
        return self.metrics[name]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload with stable metric ordering."""
        return {
            "flaky_metrics": list(self.flaky_metrics),
            "metrics": {
                name: metric.to_dict() for name, metric in self.metrics.items()
            },
            "n_runs": self.n_runs,
            "stable": self.stable,
            "verdict": self.verdict,
        }


def detect_flaky_eval(
    run_callable: RunCallable,
    n_runs: int,
    tolerance: Tolerance = DEFAULT_FLAKY_TOLERANCE,
) -> FlakyReport:
    """Detect flaky benchmark metrics by repeating the same scoring callable.

    Args:
        run_callable: Zero-argument callable that returns either a metric
            mapping or an object with a ``metrics`` mapping, such as
            ``BenchmarkReport``.
        n_runs: Number of times to run ``run_callable``. Must be positive.
        tolerance: Scalar tolerance for every metric, or a per-metric mapping.
            Metrics missing from a mapping use ``DEFAULT_FLAKY_TOLERANCE``.

    Returns:
        A ``FlakyReport`` with per-metric min, max, spread, tolerance, and
        stable/flaky verdict.

    Raises:
        ValueError: If ``n_runs`` is invalid, metrics are missing between runs,
            no numeric metrics are returned, or a metric/tolerance is non-finite.
        TypeError: If a run result cannot be interpreted as a metric mapping.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    runs = [_extract_numeric_metrics(run_callable(), index) for index in range(n_runs)]
    metric_names = sorted({name for run in runs for name in run})
    if not metric_names:
        raise ValueError("run_callable did not return any numeric metrics")

    reports: dict[str, FlakyMetricReport] = {}
    for name in metric_names:
        values = _values_for_metric(runs, name)
        minimum = min(values)
        maximum = max(values)
        spread = maximum - minimum
        resolved_tolerance = _resolve_tolerance(tolerance, name)
        reports[name] = FlakyMetricReport(
            metric=name,
            minimum=minimum,
            maximum=maximum,
            spread=spread,
            tolerance=resolved_tolerance,
            stable=spread <= resolved_tolerance,
            values=tuple(values),
        )

    return FlakyReport(n_runs=n_runs, metrics=reports)


def _extract_numeric_metrics(result: Any, run_index: int) -> dict[str, float]:
    source = getattr(result, "metrics", result)
    if not isinstance(source, Mapping):
        raise TypeError(
            "run_callable must return a metrics mapping or an object with "
            f"a metrics mapping; run {run_index + 1} returned {type(result).__name__}"
        )

    metrics: dict[str, float] = {}
    _flatten_numeric_metrics(source, prefix="", output=metrics)
    return metrics


def _flatten_numeric_metrics(
    values: Mapping[str, Any],
    *,
    prefix: str,
    output: dict[str, float],
) -> None:
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_numeric_metrics(value, prefix=name, output=output)
            continue

        parsed = _numeric_metric(value, name)
        if parsed is not None:
            output[name] = parsed


def _numeric_metric(value: Any, metric_name: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metric {metric_name!r} must be finite")
    return parsed


def _values_for_metric(
    runs: list[dict[str, float]],
    metric_name: str,
) -> list[float]:
    values: list[float] = []
    for index, metrics in enumerate(runs, start=1):
        if metric_name not in metrics:
            raise ValueError(f"run {index} did not return metric {metric_name!r}")
        values.append(metrics[metric_name])
    return values


def _resolve_tolerance(tolerance: Tolerance, metric_name: str) -> float:
    if isinstance(tolerance, Mapping):
        value = tolerance.get(metric_name, DEFAULT_FLAKY_TOLERANCE)
    else:
        value = tolerance

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"tolerance for metric {metric_name!r} must be numeric")

    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise ValueError(
            f"tolerance for metric {metric_name!r} must be a finite non-negative value"
        )
    return parsed
