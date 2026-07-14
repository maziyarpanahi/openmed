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

from openmed.core.audit import stable_hash

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

_CONFIGURATION_ID_KEYS: tuple[str, ...] = (
    "configuration_id",
    "config_id",
    "config_hash",
)
_SAFE_METADATA_METRIC_KEYS = frozenset({"accuracy_key", "leakage_key"})
_SAFE_METADATA_HASH_KEYS = frozenset(
    {
        "additional_metadata_hash",
        "benchmark_evidence_hash",
        "configuration_hash",
        "device_hash",
        "model_hash",
        "perf_evidence_hash",
    }
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
        accuracy_metric: Stable metric key defining what ``accuracy`` measures.
            All points in one frontier must use exactly the same key.
        leakage: Measured leaked-character rate in ``[0, 1]`` (lower is better).
            This evidence is required so missing measurements cannot be treated
            as perfect zero leakage.
        metadata: Optional provenance. Arbitrary metadata is represented only by
            a deterministic hash; the allow-listed report-join provenance fields
            contain metric keys or SHA-256 digests and never raw PHI.
    """

    label: str
    throughput: float
    accuracy: float
    leakage: float
    accuracy_metric: str = "accuracy"
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
            "accuracy_metric",
            _coerce_metric_key(self.accuracy_metric, field_name="accuracy_metric"),
        )
        object.__setattr__(
            self,
            "leakage",
            _coerce_unit_interval(self.leakage, field_name="leakage"),
        )
        object.__setattr__(self, "metadata", _protect_metadata(self.metadata))

    def objectives(self) -> tuple[float, float, float]:
        """Return objectives as a maximization tuple ``(tput, acc, -leak)``."""
        return (self.throughput, self.accuracy, -self.leakage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable point dictionary with stable keys."""
        return {
            "accuracy": self.accuracy,
            "accuracy_metric": self.accuracy_metric,
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
        accuracy_metric = _coerce_metric_key(
            data.get("accuracy_metric", "accuracy"), field_name="accuracy_metric"
        )
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
            accuracy_metric=accuracy_metric,
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

    def __post_init__(self) -> None:
        """Normalize the metric identifier and protect report metadata."""
        object.__setattr__(
            self,
            "accuracy_metric",
            _coerce_metric_key(self.accuracy_metric, field_name="accuracy_metric"),
        )
        object.__setattr__(self, "metadata", _protect_metadata(self.metadata))

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
            f"| Accuracy Metric | {_markdown_code(self.accuracy_metric)} |",
            f"| Points | {len(self.entries)} |",
            f"| On Frontier | {len(self.frontier)} |",
            f"| Dominated | {len(self.dominated)} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | {_markdown_code(self.generated_at)} |")

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
            dominated_by = (
                _markdown_code(entry.dominated_by) if entry.dominated_by else ""
            )
            lines.append(
                f"| {_markdown_code(point.label)} | "
                f"{_format_number(point.throughput)} | "
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
    accuracy_metric: str | None = None,
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
        accuracy_metric: Exact metric key for the accuracy axis. When omitted,
            it is inferred from the points. An explicit value must match every
            point, preventing unlike metrics from being compared.
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

    point_metrics = {point.accuracy_metric for point in resolved}
    if len(point_metrics) > 1:
        raise ValueError(
            "cannot compare unlike accuracy metrics: "
            f"{', '.join(sorted(point_metrics))}"
        )
    inferred_metric = next(iter(point_metrics), "accuracy")
    resolved_metric = (
        inferred_metric
        if accuracy_metric is None
        else _coerce_metric_key(accuracy_metric, field_name="accuracy_metric")
    )
    if point_metrics and resolved_metric != inferred_metric:
        raise ValueError(
            "accuracy_metric does not match point metric: "
            f"{resolved_metric!r} != {inferred_metric!r}"
        )

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
        accuracy_metric=resolved_metric,
        generated_at=generated_at,
        metadata=metadata or {},
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
        perf: A ``PerfReport`` (or compatible object) that identifies its model,
            device, and configuration and exposes ``docs_per_second``.
        benchmark: A ``BenchmarkReport`` (or compatible object) identifying the
            same model, device, and configuration and exposing a ``metrics``
            mapping. Configuration identity may be an attribute or a metadata
            value named ``configuration_id``, ``config_id``, or ``config_hash``.
        label: Variant label; defaults to the benchmark/perf ``model_name``.
        accuracy_keys: Dotted metric keys tried in order for the accuracy scalar.
        leakage_keys: Dotted metric keys tried in order for the leakage scalar.

    Returns:
        A :class:`FrontierPoint` combining the two reports.

    Raises:
        ValueError: If report identity or required evidence is missing or
            mismatched, a stable label is missing, or a measured value is invalid.
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

    perf_identity = _report_identity(perf, source="perf")
    benchmark_identity = _report_identity(benchmark, source="benchmark")
    identity_fields = ("model_name", "device", "configuration_id")
    mismatches = [
        field_name
        for field_name in identity_fields
        if perf_identity[field_name] != benchmark_identity[field_name]
    ]
    if mismatches:
        raise ValueError(
            f"perf and benchmark report identity mismatch for: {', '.join(mismatches)}"
        )

    resolved_label = label
    if resolved_label is None:
        resolved_label = benchmark_identity["model_name"]

    resolved_accuracy_key = _coerce_metric_key(accuracy_key, field_name="accuracy_key")
    resolved_leakage_key = _coerce_metric_key(leakage_key, field_name="leakage_key")
    identity = {field_name: perf_identity[field_name] for field_name in identity_fields}
    metadata: dict[str, Any] = {
        "accuracy_key": resolved_accuracy_key,
        "benchmark_evidence_hash": stable_hash(
            {
                **identity,
                "accuracy": accuracy_value,
                "accuracy_key": resolved_accuracy_key,
                "leakage": leakage_value,
                "leakage_key": resolved_leakage_key,
            }
        ),
        "configuration_hash": stable_hash(
            {"configuration_id": identity["configuration_id"]}
        ),
        "device_hash": stable_hash({"device": identity["device"]}),
        "leakage_key": resolved_leakage_key,
        "model_hash": stable_hash({"model_name": identity["model_name"]}),
        "perf_evidence_hash": stable_hash({**identity, "docs_per_second": throughput}),
    }

    return FrontierPoint(
        label=resolved_label,
        throughput=throughput,
        accuracy=accuracy_value,
        leakage=leakage_value,
        accuracy_metric=resolved_accuracy_key,
        metadata=metadata,
    )


def _report_identity(report: Any, *, source: str) -> dict[str, str]:
    """Return the complete identity required to join measured reports."""
    model_name = _required_identity_value(
        getattr(report, "model_name", None),
        field_name=f"{source}.model_name",
    )
    device = _required_identity_value(
        getattr(report, "device", None),
        field_name=f"{source}.device",
    )
    configuration_id = _report_configuration_id(report, source=source)
    return {
        "configuration_id": configuration_id,
        "device": device,
        "model_name": model_name,
    }


def _report_configuration_id(report: Any, *, source: str) -> str:
    """Resolve one unambiguous configuration id from a report."""
    candidates: list[str] = []
    for key in _CONFIGURATION_ID_KEYS:
        value = getattr(report, key, None)
        if value is not None:
            candidates.append(
                _required_identity_value(value, field_name=f"{source}.{key}")
            )

    metadata = getattr(report, "metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}.metadata must be a mapping")
    for key in _CONFIGURATION_ID_KEYS:
        value = metadata.get(key)
        if value is not None:
            candidates.append(
                _required_identity_value(
                    value,
                    field_name=f"{source}.metadata.{key}",
                )
            )

    unique = set(candidates)
    if not unique:
        raise ValueError(
            f"{source} report requires configuration identity under one of: "
            f"{', '.join(_CONFIGURATION_ID_KEYS)}"
        )
    if len(unique) > 1:
        raise ValueError(f"{source} report has conflicting configuration identities")
    return unique.pop()


def _required_identity_value(value: Any, *, field_name: str) -> str:
    """Return a non-empty report identity value without normalizing semantics."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    result = value.strip()
    if "\n" in result or "\r" in result:
        raise ValueError(f"{field_name} must not contain line breaks")
    return result


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


def _coerce_metric_key(value: Any, *, field_name: str) -> str:
    """Return a conservative metric identifier safe for report serialization."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty metric key")
    key = value.strip()
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not key or len(key) > 128 or any(char not in allowed for char in key):
        raise ValueError(
            f"{field_name} must use only letters, numbers, '.', '_', or '-'"
        )
    return key


def _protect_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Allow-list safe provenance and hash every other metadata field."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")

    protected: dict[str, Any] = {}
    additional: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if key in _SAFE_METADATA_METRIC_KEYS:
            protected[key] = _coerce_metric_key(value, field_name=f"metadata.{key}")
        elif key in _SAFE_METADATA_HASH_KEYS:
            protected[key] = _coerce_sha256(value, field_name=f"metadata.{key}")
        else:
            additional[key] = _json_hash_material(value, path=f"metadata.{key}")

    if additional:
        previous_hash = protected.pop("additional_metadata_hash", None)
        material: dict[str, Any] = {"metadata": additional}
        if previous_hash is not None:
            material["previous_hash"] = previous_hash
        protected["additional_metadata_hash"] = stable_hash(material)
    return protected


def _json_hash_material(value: Any, *, path: str) -> Any:
    """Validate and normalize metadata into deterministic JSON hash material."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            result[key] = _json_hash_material(child, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_hash_material(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(f"{path} must be JSON-serializable")


def _coerce_sha256(value: Any, *, field_name: str) -> str:
    """Require a canonical SHA-256 digest for safe provenance metadata."""
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return value


def _markdown_code(value: Any) -> str:
    """Render one table-safe inline-code span, including pipes and backticks."""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|")
    longest_run = 0
    current_run = 0
    for char in text:
        if char == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * (longest_run + 1)
    if text.startswith(("`", " ")) or text.endswith(("`", " ")):
        text = f" {text} "
    return f"{fence}{text}{fence}"


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
