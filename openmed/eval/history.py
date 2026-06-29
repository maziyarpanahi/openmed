"""Cross-release benchmark history diffing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

from openmed.core.baseline import (
    BASELINE_PATH,
    BaselineMiss,
    baseline_key,
    get_baseline,
    require_baseline,
)
from openmed.eval.report import BenchmarkReport

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

REGRESSION = "regression"
IMPROVEMENT = "improvement"
UNCHANGED = "unchanged"
FLAKINESS_LEDGER_SCHEMA_VERSION = "openmed.flakiness_ledger.v1"

_LOWER_IS_BETTER_MARKERS = (
    "critical_leakage_count",
    "false_negative",
    "false_positive",
    "leakage",
    "leaked",
    "loss",
    "memory",
    "over_redaction",
    "p50",
    "p95",
    "p99",
    "peak_rss",
    "quant_recall_delta",
    "ram",
    "rss",
    "_ms",
)

ReportLike = BenchmarkReport | Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class MetricDelta:
    """One numeric metric's current-vs-baseline delta."""

    metric: str
    baseline: float
    current: float
    delta: float
    relative_delta: float | None
    direction: str
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        payload: dict[str, Any] = {
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "direction": self.direction,
            "metric": self.metric,
            "verdict": self.verdict,
        }
        if self.relative_delta is not None:
            payload["relative_delta"] = self.relative_delta
        return payload


@dataclass(frozen=True)
class BenchmarkHistoryDiff:
    """Diff between a current benchmark report and a baseline report."""

    metrics: Mapping[str, MetricDelta]
    largest_regressions: tuple[MetricDelta, ...] = field(default_factory=tuple)
    largest_improvements: tuple[MetricDelta, ...] = field(default_factory=tuple)
    baseline_key: str | None = None
    baseline_released: str | None = None
    current_generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "baseline_key": self.baseline_key,
            "baseline_released": self.baseline_released,
            "current_generated_at": self.current_generated_at,
            "largest_improvements": [
                delta.to_dict() for delta in self.largest_improvements
            ],
            "largest_regressions": [
                delta.to_dict() for delta in self.largest_regressions
            ],
            "metrics": {
                metric: self.metrics[metric].to_dict()
                for metric in sorted(self.metrics)
            },
        }


@dataclass(frozen=True)
class MetricHistoryPoint:
    """One ordered value for a benchmark metric history."""

    index: int
    metric: str
    value: float
    generated_at: str | None = None
    release: str | None = None
    suite: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "generated_at": self.generated_at,
            "index": self.index,
            "metric": self.metric,
            "model_name": self.model_name,
            "release": self.release,
            "suite": self.suite,
            "value": self.value,
        }


@dataclass(frozen=True)
class FlakinessLedgerEntry:
    """Persistent quarantine state for one release gate."""

    gate: str
    quarantined: bool
    stable_runs: int = 0
    unstable_runs: int = 0
    reason: str = ""
    last_flip_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "gate": self.gate,
            "last_flip_rate": self.last_flip_rate,
            "quarantined": self.quarantined,
            "reason": self.reason,
            "stable_runs": self.stable_runs,
            "unstable_runs": self.unstable_runs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlakinessLedgerEntry":
        """Create an entry from a persisted JSON object."""

        return cls(
            gate=str(data.get("gate", "")),
            quarantined=bool(data.get("quarantined", False)),
            stable_runs=max(0, int(data.get("stable_runs", 0))),
            unstable_runs=max(0, int(data.get("unstable_runs", 0))),
            reason=str(data.get("reason", "")),
            last_flip_rate=_finite_float(data.get("last_flip_rate"), default=0.0),
        )


@dataclass(frozen=True)
class FlakinessLedger:
    """Persistent release-gate flakiness quarantine ledger."""

    entries: Mapping[str, FlakinessLedgerEntry] = field(default_factory=dict)
    schema_version: str = FLAKINESS_LEDGER_SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | Path) -> "FlakinessLedger":
        """Load a ledger from *path*, returning an empty ledger if absent."""

        ledger_path = Path(path)
        if not ledger_path.is_file():
            return cls()
        with ledger_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"flakiness ledger payload must be an object: {path}")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlakinessLedger":
        """Create a ledger from a JSON-ready mapping."""

        raw_entries = data.get("entries") or {}
        entries: dict[str, FlakinessLedgerEntry] = {}
        if isinstance(raw_entries, Mapping):
            for gate, value in raw_entries.items():
                if not isinstance(value, Mapping):
                    continue
                item = dict(value)
                item.setdefault("gate", gate)
                entry = FlakinessLedgerEntry.from_dict(item)
                if entry.gate:
                    entries[entry.gate] = entry
        return cls(entries=entries)

    @property
    def quarantined_gates(self) -> tuple[str, ...]:
        """Return gate names currently held in quarantine."""

        return tuple(
            sorted(gate for gate, entry in self.entries.items() if entry.quarantined)
        )

    def entry(self, gate: str) -> FlakinessLedgerEntry | None:
        """Return the ledger entry for *gate*, if one exists."""

        return self.entries.get(gate)

    def record_gate(
        self,
        gate: str,
        *,
        unstable: bool,
        reason: str,
        flip_rate: float,
        stability_window: int,
    ) -> tuple["FlakinessLedger", FlakinessLedgerEntry]:
        """Return an updated ledger and entry after one gate stability sweep."""

        if stability_window < 1:
            raise ValueError("stability_window must be at least 1")

        previous = self.entries.get(gate)
        if unstable:
            entry = FlakinessLedgerEntry(
                gate=gate,
                quarantined=True,
                stable_runs=0,
                unstable_runs=(previous.unstable_runs if previous else 0) + 1,
                reason=reason or "unstable gate verdict",
                last_flip_rate=flip_rate,
            )
        elif previous is not None and previous.quarantined:
            stable_runs = previous.stable_runs + 1
            quarantined = stable_runs < stability_window
            if quarantined:
                policy_reason = (
                    f"awaiting stability window {stable_runs}/{stability_window}"
                )
            else:
                policy_reason = "stability window satisfied"
            entry = FlakinessLedgerEntry(
                gate=gate,
                quarantined=quarantined,
                stable_runs=stable_runs,
                unstable_runs=previous.unstable_runs,
                reason=policy_reason,
                last_flip_rate=flip_rate,
            )
        else:
            entry = FlakinessLedgerEntry(
                gate=gate,
                quarantined=False,
                stable_runs=(previous.stable_runs if previous else 0) + 1,
                unstable_runs=previous.unstable_runs if previous else 0,
                reason=reason or "stable",
                last_flip_rate=flip_rate,
            )

        entries = dict(self.entries)
        entries[gate] = entry
        return FlakinessLedger(entries=entries), entry

    def save(self, path: str | Path) -> Path:
        """Persist the ledger to *path* using deterministic JSON."""

        ledger_path = Path(path)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ledger_path

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "entries": {
                gate: self.entries[gate].to_dict() for gate in sorted(self.entries)
            },
            "schema_version": self.schema_version,
        }


def diff_against_baseline(
    current_report: ReportLike,
    baseline_report: ReportLike | None = None,
    *,
    baseline_path: str | Path = BASELINE_PATH,
    metric_directions: Mapping[str, str] | None = None,
    rank_limit: int | None = 5,
) -> BenchmarkHistoryDiff:
    """Diff a current benchmark report against a baseline.

    If ``baseline_report`` is omitted, the last-green baseline store is read
    from ``baseline_path`` and resolved using the current report's family, tier,
    and format metadata. ``baseline_report`` may also be a baseline store, a
    single baseline entry, a full ``BenchmarkReport`` payload, raw metrics, or a
    JSON path to any of those payloads.
    """

    if rank_limit is not None and rank_limit < 0:
        raise ValueError("rank_limit must be non-negative or None")

    current_payload = _payload(current_report)
    baseline_payload = _resolve_baseline_payload(
        current_payload,
        baseline_report,
        baseline_path=baseline_path,
    )
    current_metrics = _numeric_metrics(_metrics(current_payload))
    baseline_metrics = _numeric_metrics(_metrics(baseline_payload))

    deltas = {
        metric: _metric_delta(
            metric,
            baseline=baseline_metrics[metric],
            current=current_metrics[metric],
            metric_directions=metric_directions,
        )
        for metric in sorted(current_metrics.keys() & baseline_metrics.keys())
    }
    regressions = _rank_changes(
        deltas.values(),
        verdict=REGRESSION,
        rank_limit=rank_limit,
    )
    improvements = _rank_changes(
        deltas.values(),
        verdict=IMPROVEMENT,
        rank_limit=rank_limit,
    )
    identity = _identity(current_payload)
    return BenchmarkHistoryDiff(
        metrics=deltas,
        largest_regressions=regressions,
        largest_improvements=improvements,
        baseline_key=(
            str(baseline_payload.get("key"))
            if baseline_payload.get("key") is not None
            else baseline_key(identity["family"], identity["tier"], identity["format"])
        ),
        baseline_released=_optional_str(baseline_payload.get("released")),
        current_generated_at=_optional_str(current_payload.get("generated_at")),
    )


def metric_history(
    reports_in_order: Iterable[ReportLike],
    metric: str,
) -> list[MetricHistoryPoint]:
    """Return one metric's values across reports in caller-provided order."""

    history: list[MetricHistoryPoint] = []
    for index, report in enumerate(reports_in_order):
        payload = _payload(report)
        value = _lookup_numeric_metric(_metrics(payload), metric)
        metadata = _mapping(payload.get("metadata"))
        history.append(
            MetricHistoryPoint(
                index=index,
                metric=metric,
                value=value,
                generated_at=_optional_str(payload.get("generated_at")),
                release=_optional_str(
                    metadata.get("released") or metadata.get("release")
                ),
                suite=_optional_str(payload.get("suite")),
                model_name=_optional_str(payload.get("model_name")),
            )
        )
    return history


def _metric_delta(
    metric: str,
    *,
    baseline: float,
    current: float,
    metric_directions: Mapping[str, str] | None,
) -> MetricDelta:
    delta = current - baseline
    relative_delta = None if baseline == 0 else delta / abs(baseline)
    direction = _direction(metric, metric_directions)
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        verdict = UNCHANGED
    else:
        effect = delta if direction == HIGHER_IS_BETTER else -delta
        verdict = IMPROVEMENT if effect > 0 else REGRESSION
    return MetricDelta(
        metric=metric,
        baseline=baseline,
        current=current,
        delta=delta,
        relative_delta=relative_delta,
        direction=direction,
        verdict=verdict,
    )


def _rank_changes(
    deltas: Iterable[MetricDelta],
    *,
    verdict: str,
    rank_limit: int | None,
) -> tuple[MetricDelta, ...]:
    ranked = sorted(
        (delta for delta in deltas if delta.verdict == verdict),
        key=lambda delta: (-abs(delta.delta), delta.metric),
    )
    if rank_limit is not None:
        ranked = ranked[:rank_limit]
    return tuple(ranked)


def _resolve_baseline_payload(
    current_payload: Mapping[str, Any],
    baseline_report: ReportLike | None,
    *,
    baseline_path: str | Path,
) -> dict[str, Any]:
    identity = _identity(current_payload)
    if baseline_report is None:
        return require_baseline(
            identity["family"],
            identity["tier"],
            identity["format"],
            path=baseline_path,
        )

    payload = _payload(baseline_report)
    if "entries" not in payload:
        return payload

    entry = get_baseline(
        identity["family"],
        identity["tier"],
        identity["format"],
        store=payload,
    )
    if entry is None:
        key = baseline_key(identity["family"], identity["tier"], identity["format"])
        raise BaselineMiss(f"No last-green baseline for key: {key}")
    return entry


def _payload(report: ReportLike) -> dict[str, Any]:
    if isinstance(report, BenchmarkReport):
        return report.to_dict()
    if isinstance(report, (str, Path)):
        path = Path(report)
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"benchmark history payload must be an object: {path}")
        return dict(loaded)
    if hasattr(report, "to_dict") and callable(report.to_dict):
        payload = report.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(report, Mapping):
        return dict(report)
    raise TypeError(f"unsupported benchmark history payload: {type(report).__name__}")


def _identity(payload: Mapping[str, Any]) -> dict[str, str | None]:
    metadata = _mapping(payload.get("metadata"))
    family = str(metadata.get("family") or payload.get("family") or "Unknown")
    tier = _optional_str(metadata.get("tier") or payload.get("tier"))
    format_name = str(
        metadata.get("format")
        or metadata.get("model_format")
        or payload.get("format")
        or payload.get("device")
        or "unknown"
    )
    return {"family": family, "tier": tier, "format": format_name}


def _metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return payload


def _numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, Mapping):
        rows: dict[str, float] = {}
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_numeric_metrics(value[key], child_prefix))
        return rows
    if _is_number(value) and prefix:
        return {prefix: float(value)}
    return {}


def _lookup_numeric_metric(metrics: Mapping[str, Any], metric: str) -> float:
    current: Any = metrics
    for part in metric.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"metric not found: {metric}")
        current = current[part]
    if not _is_number(current):
        raise ValueError(f"metric is not numeric: {metric}")
    return float(current)


def _direction(
    metric: str,
    metric_directions: Mapping[str, str] | None,
) -> str:
    if metric_directions and metric in metric_directions:
        direction = metric_directions[metric]
        if direction not in {HIGHER_IS_BETTER, LOWER_IS_BETTER}:
            raise ValueError(f"invalid metric direction for {metric}: {direction}")
        return direction

    normalized = metric.lower().replace("-", "_")
    if any(marker in normalized for marker in _LOWER_IS_BETTER_MARKERS):
        return LOWER_IS_BETTER
    return HIGHER_IS_BETTER


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


__all__ = [
    "BenchmarkHistoryDiff",
    "FLAKINESS_LEDGER_SCHEMA_VERSION",
    "FlakinessLedger",
    "FlakinessLedgerEntry",
    "HIGHER_IS_BETTER",
    "IMPROVEMENT",
    "LOWER_IS_BETTER",
    "MetricDelta",
    "MetricHistoryPoint",
    "REGRESSION",
    "UNCHANGED",
    "diff_against_baseline",
    "metric_history",
]
