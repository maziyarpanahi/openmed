"""Repeated-run variance checks for benchmark metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from typing import Any

from openmed.eval.history import FlakinessLedger

DEFAULT_FLAKY_TOLERANCE = 1e-12
DEFAULT_GATE_SEEDS = tuple(range(20))
DEFAULT_GATE_FLIP_RATE_TOLERANCE = 0.0
DEFAULT_GATE_METRIC_VARIANCE_TOLERANCE = DEFAULT_FLAKY_TOLERANCE
DEFAULT_GATE_CONFIDENCE_LEVEL = 0.95
DEFAULT_QUARANTINE_STABILITY_WINDOW = 3

RunCallable = Callable[[], Any]
GateRunCallable = Callable[[int], Any]
Tolerance = float | Mapping[str, float]


class NondeterministicGateRunError(RuntimeError):
    """Raised when same-seed gate probes produce different outcomes."""

    def __init__(
        self,
        *,
        seed: int,
        first_hash: str,
        second_hash: str,
        drift: tuple[str, ...],
    ) -> None:
        self.seed = seed
        self.first_hash = first_hash
        self.second_hash = second_hash
        self.drift = drift
        details = ", ".join(drift) if drift else "gate payload drift"
        super().__init__(
            f"same-seed release gate run drifted for seed {seed}: {details}"
        )


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
class NondeterminismProbeReport:
    """Same-seed release-gate rerun probe result."""

    seed: int
    stable: bool
    first_hash: str
    second_hash: str
    drift: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready nondeterminism probe report."""

        return {
            "drift": list(self.drift),
            "first_hash": self.first_hash,
            "second_hash": self.second_hash,
            "seed": self.seed,
            "stable": self.stable,
        }


@dataclass(frozen=True)
class GateInstabilityReport:
    """Per-gate decision and metric instability across a seed sweep."""

    gate: str
    samples: int
    passed: int
    failed: int
    flip_rate: float
    flip_rate_lower_bound: float
    flip_rate_upper_bound: float
    metric_variance: Mapping[str, float] = field(default_factory=dict)
    metric_values: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    unstable: bool = False
    quarantined: bool = False
    reason: str = "stable"
    ledger_state: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stable(self) -> bool:
        """Return whether this gate is stable and not quarantined."""

        return not self.unstable and not self.quarantined

    @property
    def verdict(self) -> str:
        """Return the stability verdict."""

        return "stable" if self.stable else "quarantined"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready per-gate instability report."""

        return {
            "failed": self.failed,
            "flip_rate": self.flip_rate,
            "flip_rate_bounds": {
                "lower": self.flip_rate_lower_bound,
                "upper": self.flip_rate_upper_bound,
            },
            "gate": self.gate,
            "ledger_state": _plain_json(self.ledger_state),
            "metric_values": {
                name: list(self.metric_values[name])
                for name in sorted(self.metric_values)
            },
            "metric_variance": {
                name: self.metric_variance[name]
                for name in sorted(self.metric_variance)
            },
            "passed": self.passed,
            "quarantined": self.quarantined,
            "reason": self.reason,
            "samples": self.samples,
            "stable": self.stable,
            "unstable": self.unstable,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class GateFlakinessReport:
    """Release-gate stability summary across a seed sweep."""

    seeds: tuple[int, ...]
    gates: Mapping[str, GateInstabilityReport] = field(default_factory=dict)
    nondeterminism_probe: NondeterminismProbeReport | None = None
    flip_rate_tolerance: float = DEFAULT_GATE_FLIP_RATE_TOLERANCE
    metric_variance_tolerance: Tolerance = DEFAULT_GATE_METRIC_VARIANCE_TOLERANCE
    confidence_level: float = DEFAULT_GATE_CONFIDENCE_LEVEL
    stability_window: int = DEFAULT_QUARANTINE_STABILITY_WINDOW
    ledger: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stable(self) -> bool:
        """Return whether every gate is stable and released from quarantine."""

        return not self.quarantined_gates

    @property
    def verdict(self) -> str:
        """Return the overall stability verdict."""

        return "stable" if self.stable else "quarantined"

    @property
    def unstable_gates(self) -> tuple[str, ...]:
        """Return gates whose current seed sweep exceeded tolerance."""

        return tuple(gate for gate in sorted(self.gates) if self.gates[gate].unstable)

    @property
    def quarantined_gates(self) -> tuple[str, ...]:
        """Return gates currently blocking release due to flakiness."""

        return tuple(
            gate for gate in sorted(self.gates) if self.gates[gate].quarantined
        )

    def gate(self, name: str) -> GateInstabilityReport:
        """Return one gate's instability report."""

        return self.gates[name]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload with stable ordering."""

        payload: dict[str, Any] = {
            "confidence_level": self.confidence_level,
            "flip_rate_tolerance": self.flip_rate_tolerance,
            "gates": {gate: self.gates[gate].to_dict() for gate in sorted(self.gates)},
            "ledger": _plain_json(self.ledger),
            "metric_variance_tolerance": _plain_json(self.metric_variance_tolerance),
            "quarantined_gates": list(self.quarantined_gates),
            "seeds": list(self.seeds),
            "stability_window": self.stability_window,
            "stable": self.stable,
            "unstable_gates": list(self.unstable_gates),
            "verdict": self.verdict,
        }
        if self.nondeterminism_probe is not None:
            payload["nondeterminism_probe"] = self.nondeterminism_probe.to_dict()
        return payload


@dataclass(frozen=True)
class _GateSample:
    gate: str
    passed: bool
    reason: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class _GateRun:
    seed: int
    gates: Mapping[str, _GateSample]
    fingerprint: str


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


def detect_gate_suite_flakiness(
    gate_runner: GateRunCallable,
    seeds: Sequence[int] = DEFAULT_GATE_SEEDS,
    *,
    flip_rate_tolerance: float = DEFAULT_GATE_FLIP_RATE_TOLERANCE,
    metric_variance_tolerance: Tolerance = DEFAULT_GATE_METRIC_VARIANCE_TOLERANCE,
    confidence_level: float = DEFAULT_GATE_CONFIDENCE_LEVEL,
    nondeterminism_seed: int | None = None,
    ledger_path: str | Path | None = None,
    stability_window: int = DEFAULT_QUARANTINE_STABILITY_WINDOW,
) -> GateFlakinessReport:
    """Run a release-gate suite across seeds and quarantine unstable gates.

    Args:
        gate_runner: Callable that accepts a seed and returns a ``GateReport`` or
            JSON-like gate report mapping with ``gate_results``.
        seeds: Configurable seed sweep. At least one seed is required.
        flip_rate_tolerance: Maximum tolerated per-gate verdict flip rate.
        metric_variance_tolerance: Scalar or per-metric population-variance
            tolerance. Mapping keys may be either metric names or
            ``"<gate>.<metric>"``.
        confidence_level: Wilson confidence level for flip-rate bounds.
        nondeterminism_seed: Fixed seed for the same-seed rerun probe. Defaults
            to the first seed in ``seeds``.
        ledger_path: Optional JSON ledger path used to persist quarantine state.
        stability_window: Consecutive stable sweeps needed to release a gate
            from a prior quarantine.

    Raises:
        NondeterministicGateRunError: If the same-seed probe detects drift.
        ValueError: If the seed sweep or tolerance configuration is invalid.
        TypeError: If a gate run cannot be interpreted as a gate report.
    """

    seed_values = _coerce_seeds(seeds)
    flip_tolerance = _non_negative_float(
        flip_rate_tolerance,
        name="flip_rate_tolerance",
    )
    if flip_tolerance > 1.0:
        raise ValueError("flip_rate_tolerance must be between 0 and 1")
    _validate_confidence_level(confidence_level)
    if stability_window < 1:
        raise ValueError("stability_window must be at least 1")

    probe_seed = seed_values[0] if nondeterminism_seed is None else nondeterminism_seed
    first_probe = _coerce_gate_run(gate_runner(probe_seed), seed=probe_seed)
    second_probe = _coerce_gate_run(gate_runner(probe_seed), seed=probe_seed)
    drift = _gate_run_drift(first_probe, second_probe)
    if first_probe.fingerprint != second_probe.fingerprint:
        raise NondeterministicGateRunError(
            seed=probe_seed,
            first_hash=first_probe.fingerprint,
            second_hash=second_probe.fingerprint,
            drift=drift or ("payload",),
        )
    probe = NondeterminismProbeReport(
        seed=probe_seed,
        stable=True,
        first_hash=first_probe.fingerprint,
        second_hash=second_probe.fingerprint,
    )

    runs = [_coerce_gate_run(gate_runner(seed), seed=seed) for seed in seed_values]
    gate_names = _stable_gate_names(runs)
    ledger = FlakinessLedger.load(ledger_path) if ledger_path is not None else None

    reports: dict[str, GateInstabilityReport] = {}
    for gate in gate_names:
        report = _gate_instability_report(
            gate,
            runs,
            flip_rate_tolerance=flip_tolerance,
            metric_variance_tolerance=metric_variance_tolerance,
            confidence_level=confidence_level,
        )
        ledger_state: Mapping[str, Any] = {}
        quarantined = report.unstable
        reason = report.reason
        if ledger is not None:
            ledger, entry = ledger.record_gate(
                gate,
                unstable=report.unstable,
                reason=report.reason,
                flip_rate=report.flip_rate,
                stability_window=stability_window,
            )
            ledger_state = entry.to_dict()
            quarantined = entry.quarantined
            if entry.quarantined and not report.unstable:
                reason = entry.reason
        reports[gate] = GateInstabilityReport(
            gate=report.gate,
            samples=report.samples,
            passed=report.passed,
            failed=report.failed,
            flip_rate=report.flip_rate,
            flip_rate_lower_bound=report.flip_rate_lower_bound,
            flip_rate_upper_bound=report.flip_rate_upper_bound,
            metric_variance=report.metric_variance,
            metric_values=report.metric_values,
            unstable=report.unstable,
            quarantined=quarantined,
            reason=reason,
            ledger_state=ledger_state,
        )

    ledger_payload: Mapping[str, Any] = {}
    if ledger is not None:
        ledger.save(ledger_path)
        ledger_payload = ledger.to_dict()

    return GateFlakinessReport(
        seeds=seed_values,
        gates=reports,
        nondeterminism_probe=probe,
        flip_rate_tolerance=flip_tolerance,
        metric_variance_tolerance=metric_variance_tolerance,
        confidence_level=confidence_level,
        stability_window=stability_window,
        ledger=ledger_payload,
    )


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


def _coerce_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes, bytearray)):
        raise TypeError("seeds must be a sequence of integers")
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise ValueError("seeds must contain at least one seed")
    return values


def _coerce_gate_run(result: Any, *, seed: int) -> _GateRun:
    payload = _gate_payload(result)
    checks = payload.get("gate_results") or []
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes, bytearray)):
        raise TypeError("gate report payload must contain a gate_results sequence")

    gates: dict[str, _GateSample] = {}
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, Mapping):
            raise TypeError(f"gate result {index} for seed {seed} must be an object")
        gate = str(check.get("gate") or "")
        if not gate:
            raise ValueError(f"gate result {index} for seed {seed} is missing a gate")
        metrics: dict[str, float] = {}
        details = check.get("details") or {}
        if isinstance(details, Mapping):
            _flatten_numeric_metrics(details, prefix="", output=metrics)
        gates[gate] = _GateSample(
            gate=gate,
            passed=bool(check.get("passed", False)),
            reason=str(check.get("reason", "")),
            metrics=metrics,
        )

    if not gates:
        raise ValueError("gate report payload did not include any gate results")
    return _GateRun(seed=seed, gates=gates, fingerprint=_fingerprint(payload))


def _gate_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("gate_runner must return a GateReport-like object or mapping")


def _stable_gate_names(runs: Sequence[_GateRun]) -> tuple[str, ...]:
    gate_names = tuple(sorted(runs[0].gates))
    expected = set(gate_names)
    for run in runs[1:]:
        actual = set(run.gates)
        if actual != expected:
            missing = sorted(expected - actual)
            added = sorted(actual - expected)
            raise ValueError(
                "gate suite changed across seed sweep: "
                f"seed {run.seed} missing={missing} added={added}"
            )
    return gate_names


def _gate_instability_report(
    gate: str,
    runs: Sequence[_GateRun],
    *,
    flip_rate_tolerance: float,
    metric_variance_tolerance: Tolerance,
    confidence_level: float,
) -> GateInstabilityReport:
    verdicts = tuple(run.gates[gate].passed for run in runs)
    passed = sum(1 for verdict in verdicts if verdict)
    failed = len(verdicts) - passed
    flips = min(passed, failed)
    flip_rate = flips / len(verdicts)
    lower, upper = _wilson_bounds(flips, len(verdicts), confidence_level)

    metric_values = _gate_metric_values(gate, runs)
    metric_variance = {
        metric: _population_variance(values)
        for metric, values in sorted(metric_values.items())
    }
    unstable_metrics = tuple(
        metric
        for metric, variance in metric_variance.items()
        if variance
        > _resolve_gate_metric_tolerance(metric_variance_tolerance, gate, metric)
    )

    reasons: list[str] = []
    if flip_rate > flip_rate_tolerance:
        reasons.append(
            f"flip_rate {flip_rate:.12g} exceeds tolerance {flip_rate_tolerance:.12g}"
        )
    if unstable_metrics:
        reasons.append(
            "metric variance exceeds tolerance: " + ", ".join(unstable_metrics)
        )
    reason = "; ".join(reasons) if reasons else "stable"
    unstable = bool(reasons)
    return GateInstabilityReport(
        gate=gate,
        samples=len(verdicts),
        passed=passed,
        failed=failed,
        flip_rate=flip_rate,
        flip_rate_lower_bound=lower,
        flip_rate_upper_bound=upper,
        metric_variance=metric_variance,
        metric_values=metric_values,
        unstable=unstable,
        quarantined=unstable,
        reason=reason,
    )


def _gate_metric_values(
    gate: str,
    runs: Sequence[_GateRun],
) -> dict[str, tuple[float, ...]]:
    metric_names = sorted(
        {
            metric
            for run in runs
            for metric in run.gates[gate].metrics
            if all(metric in item.gates[gate].metrics for item in runs)
        }
    )
    return {
        metric: tuple(run.gates[gate].metrics[metric] for run in runs)
        for metric in metric_names
    }


def _gate_run_drift(first: _GateRun, second: _GateRun) -> tuple[str, ...]:
    drift: list[str] = []
    for gate in sorted(set(first.gates) | set(second.gates)):
        left = first.gates.get(gate)
        right = second.gates.get(gate)
        if left is None or right is None:
            drift.append(f"{gate}.present")
            continue
        if left.passed != right.passed:
            drift.append(f"{gate}.passed")
        if left.reason != right.reason:
            drift.append(f"{gate}.reason")
        for metric in sorted(set(left.metrics) | set(right.metrics)):
            if left.metrics.get(metric) != right.metrics.get(metric):
                drift.append(f"{gate}.metrics.{metric}")
    return tuple(drift)


def _wilson_bounds(
    successes: int,
    samples: int,
    confidence_level: float,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    phat = successes / samples
    denominator = 1.0 + (z * z / samples)
    center = (phat + (z * z / (2.0 * samples))) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z / (4.0 * samples))) / samples)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if min(values) == max(values):
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _resolve_gate_metric_tolerance(
    tolerance: Tolerance,
    gate: str,
    metric_name: str,
) -> float:
    if isinstance(tolerance, Mapping):
        value = tolerance.get(
            f"{gate}.{metric_name}",
            tolerance.get(metric_name, DEFAULT_GATE_METRIC_VARIANCE_TOLERANCE),
        )
    else:
        value = tolerance
    return _non_negative_float(
        value,
        name=f"metric variance tolerance for {gate}.{metric_name}",
    )


def _non_negative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite non-negative value")
    return parsed


def _validate_confidence_level(confidence_level: float) -> None:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be between 0 and 1")


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _plain_json(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
