"""False-discovery classification of benchmark metric regressions.

Given aligned baseline and current nightly metric windows, this module runs one
one-sided regression test per metric, corrects the whole metric family with the
Benjamini-Hochberg false-discovery procedure, and classifies each surviving
regression as blocking (leakage-family metrics) or advisory (all other metrics,
including F1-only movements). Only metric-level aggregates are emitted; fixture
payloads, predictions, and record-level values are never retained.

The pipeline is deterministic: per-metric p-values come from a closed-form
Welch two-sample t-test with no sampling, and the Benjamini-Hochberg step-up is
a pure ordering over those p-values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, variance
from typing import Any

from openmed.eval.history import HIGHER_IS_BETTER, LOWER_IS_BETTER

REGRESSION_SIGNIFICANCE_SCHEMA_VERSION = "openmed.regression_significance.v1"

BLOCKING = "blocking"
ADVISORY = "advisory"
NONE = "none"
CLEAN = "clean"

DEFAULT_ALPHA = 0.05

LEAKAGE_FAMILY = "leakage"
OTHER_FAMILY = "other"

# Substrings (case-insensitive) that mark a metric as leakage-first, so a
# significant regression on it blocks the release rather than merely advising.
DEFAULT_BLOCKING_METRIC_MARKERS = (
    "critical_leakage",
    "exposure",
    "leak",
    "leaked",
    "leakage",
    "phi_leak",
    "reemission",
    "re_emission",
)

# Substrings (case-insensitive) that flip a metric to lower-is-better when no
# explicit direction override is supplied. Mirrors the history-diff markers and
# additionally covers every DEFAULT_BLOCKING_METRIC_MARKERS substring, so a
# leakage-first metric can never be auto-resolved to higher-is-better (which
# would treat a rise in leakage as an improvement and silently miss the
# blocking regression).
_LOWER_IS_BETTER_MARKERS = (
    "critical_leakage_count",
    "exposure",
    "false_negative",
    "false_positive",
    "leak",
    "leaked",
    "leakage",
    "loss",
    "memory",
    "over_redaction",
    "p50",
    "p95",
    "p99",
    "peak_rss",
    "quant_recall_delta",
    "ram",
    "reemission",
    "re_emission",
    "rss",
    "_ms",
)


@dataclass(frozen=True)
class FalseDiscoveryDecision:
    """Benjamini-Hochberg outcome for one hypothesis in a family."""

    key: str
    p_value: float
    rank: int
    critical_value: float
    adjusted_p_value: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "adjusted_p_value": self.adjusted_p_value,
            "critical_value": self.critical_value,
            "key": self.key,
            "p_value": self.p_value,
            "rank": self.rank,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class FalseDiscoveryResult:
    """Benjamini-Hochberg step-up decision over a family of p-values."""

    alpha: float
    decisions: tuple[FalseDiscoveryDecision, ...] = field(default_factory=tuple)
    schema_version: str = REGRESSION_SIGNIFICANCE_SCHEMA_VERSION

    @property
    def family_size(self) -> int:
        """Return the number of hypotheses in the corrected family."""

        return len(self.decisions)

    @property
    def significant(self) -> tuple[str, ...]:
        """Return the rejected hypothesis keys in stable key order."""

        return tuple(
            sorted(decision.key for decision in self.decisions if decision.significant)
        )

    def adjusted_p_values(self) -> dict[str, float]:
        """Return each hypothesis key mapped to its BH-adjusted p-value."""

        return {decision.key: decision.adjusted_p_value for decision in self.decisions}

    def decision(self, key: str) -> FalseDiscoveryDecision:
        """Return one hypothesis decision by key."""

        for decision in self.decisions:
            if decision.key == key:
                return decision
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "alpha": self.alpha,
            "decisions": [
                decision.to_dict()
                for decision in sorted(self.decisions, key=lambda item: item.key)
            ],
            "family_size": self.family_size,
            "schema_version": self.schema_version,
            "significant": list(self.significant),
        }


@dataclass(frozen=True)
class MetricRegressionSignal:
    """One metric's regression test outcome after family-wide correction."""

    metric: str
    family: str
    direction: str
    baseline_mean: float
    current_mean: float
    delta: float
    p_value: float
    adjusted_p_value: float
    significant: bool
    severity: str

    @property
    def blocking(self) -> bool:
        """Return whether this signal blocks the release."""

        return self.severity == BLOCKING

    @property
    def advisory(self) -> bool:
        """Return whether this signal is advisory only."""

        return self.severity == ADVISORY

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "adjusted_p_value": self.adjusted_p_value,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "delta": self.delta,
            "direction": self.direction,
            "family": self.family,
            "metric": self.metric,
            "p_value": self.p_value,
            "severity": self.severity,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class RegressionClassification:
    """Deterministic verdict payload for downstream gate consumers."""

    alpha: float
    signals: tuple[MetricRegressionSignal, ...] = field(default_factory=tuple)
    schema_version: str = REGRESSION_SIGNIFICANCE_SCHEMA_VERSION

    @property
    def significant_metrics(self) -> tuple[str, ...]:
        """Return every metric flagged as a significant regression."""

        return tuple(signal.metric for signal in self.signals if signal.significant)

    @property
    def blocking_metrics(self) -> tuple[str, ...]:
        """Return leakage-family metrics that block the release."""

        return tuple(signal.metric for signal in self.signals if signal.blocking)

    @property
    def advisory_metrics(self) -> tuple[str, ...]:
        """Return non-leakage metrics whose regression is advisory only."""

        return tuple(signal.metric for signal in self.signals if signal.advisory)

    @property
    def blocking(self) -> bool:
        """Return whether any leakage-family regression blocks the release."""

        return bool(self.blocking_metrics)

    @property
    def verdict(self) -> str:
        """Return the release verdict for gate consumers."""

        if self.blocking_metrics:
            return BLOCKING
        if self.advisory_metrics:
            return ADVISORY
        return CLEAN

    def signal(self, metric: str) -> MetricRegressionSignal:
        """Return one metric's regression signal."""

        for signal in self.signals:
            if signal.metric == metric:
                return signal
        raise KeyError(metric)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON summary keyed for gate consumers."""

        return {
            "advisory_metrics": list(self.advisory_metrics),
            "alpha": self.alpha,
            "blocking": self.blocking,
            "blocking_metrics": list(self.blocking_metrics),
            "schema_version": self.schema_version,
            "signals": [
                signal.to_dict()
                for signal in sorted(self.signals, key=lambda item: item.metric)
            ],
            "significant_metrics": list(self.significant_metrics),
            "verdict": self.verdict,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the verdict to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True)
class NightlyRegressionWindow:
    """One classified nightly window from a rolling scan.

    ``blocking`` is the debounced release signal: it is set only when a
    leakage-family metric has been significant for ``confirmations`` consecutive
    windows ending at this one, which suppresses the isolated single-window
    false positives that any per-window test scatters across a long nightly
    history. ``confirmed_metrics`` names the leakage metrics that met that
    persistence bar.
    """

    night_index: int
    current_start: int
    classification: RegressionClassification
    blocking: bool = False
    confirmed_metrics: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "blocking": self.blocking,
            "classification": self.classification.to_dict(),
            "confirmed_metrics": list(self.confirmed_metrics),
            "current_start": self.current_start,
            "night_index": self.night_index,
        }


def benjamini_hochberg(
    p_values: Mapping[str, float],
    alpha: float = DEFAULT_ALPHA,
) -> FalseDiscoveryResult:
    """Apply the Benjamini-Hochberg step-up procedure to a p-value family.

    ``p_values`` maps a hypothesis key (typically a metric name) to its raw
    p-value. The procedure sorts the ``m`` p-values ascending, rejects every
    hypothesis at or below the largest rank ``i`` whose p-value satisfies
    ``p_(i) <= (i / m) * alpha``, and reports monotone BH-adjusted p-values.
    Ties are broken by key so the outcome is fully deterministic.

    Args:
        p_values: Hypothesis key to raw p-value in ``[0, 1]``.
        alpha: Target false-discovery rate in ``(0, 1]``.

    Returns:
        A :class:`FalseDiscoveryResult` with one decision per hypothesis.

    Raises:
        ValueError: If ``alpha`` is out of range or a p-value is not in
            ``[0, 1]`` or is non-finite.
    """

    _validate_alpha(alpha)
    ordered_keys = sorted(p_values)
    validated = [(key, _validate_p_value(key, p_values[key])) for key in ordered_keys]
    total = len(validated)
    if total == 0:
        return FalseDiscoveryResult(alpha=alpha, decisions=())

    ranked = sorted(validated, key=lambda item: (item[1], item[0]))

    # Largest rank whose p-value clears its BH critical value.
    max_significant_rank = 0
    for rank, (_key, p_value) in enumerate(ranked, start=1):
        if p_value <= (rank / total) * alpha:
            max_significant_rank = rank

    # Monotone BH-adjusted p-values computed from the largest rank downward.
    adjusted_by_key: dict[str, float] = {}
    running_min = 1.0
    for rank in range(total, 0, -1):
        key, p_value = ranked[rank - 1]
        running_min = min(running_min, min(1.0, p_value * total / rank))
        adjusted_by_key[key] = running_min

    decisions = []
    for rank, (key, p_value) in enumerate(ranked, start=1):
        decisions.append(
            FalseDiscoveryDecision(
                key=key,
                p_value=p_value,
                rank=rank,
                critical_value=(rank / total) * alpha,
                adjusted_p_value=adjusted_by_key[key],
                significant=rank <= max_significant_rank,
            )
        )

    return FalseDiscoveryResult(
        alpha=alpha,
        decisions=tuple(sorted(decisions, key=lambda item: item.key)),
    )


def regression_p_value(
    baseline_values: Sequence[float],
    current_values: Sequence[float],
    *,
    direction: str,
) -> float:
    """Return a one-sided regression p-value between two metric windows.

    The alternative hypothesis is that the current window moved in the metric's
    *worse* direction relative to the baseline window. The test is a closed-form
    Welch two-sample t-test whose one-sided tail comes from the Student-t
    distribution, so it is deterministic and requires no resampling.

    Args:
        baseline_values: Earlier nightly metric values.
        current_values: Recent nightly metric values.
        direction: ``HIGHER_IS_BETTER`` or ``LOWER_IS_BETTER``.

    Returns:
        A p-value in ``[0, 1]``; large when the current window is no worse.
    """

    baseline = [float(value) for value in baseline_values]
    current = [float(value) for value in current_values]
    if not baseline or not current:
        raise ValueError("baseline and current windows must be non-empty")
    if direction not in {HIGHER_IS_BETTER, LOWER_IS_BETTER}:
        raise ValueError(f"unsupported metric direction: {direction}")

    baseline_mean = mean(baseline)
    current_mean = mean(current)
    worse_effect = _worse_effect(direction, baseline_mean, current_mean)

    baseline_var = variance(baseline) if len(baseline) > 1 else 0.0
    current_var = variance(current) if len(current) > 1 else 0.0
    standard_error = math.sqrt(
        baseline_var / len(baseline) + current_var / len(current)
    )
    if standard_error == 0.0:
        return 1.0 if worse_effect <= 0.0 else 0.0

    degrees_of_freedom = _welch_degrees_of_freedom(
        baseline_var,
        current_var,
        len(baseline),
        len(current),
    )
    t_statistic = worse_effect / standard_error
    return max(0.0, min(1.0, _student_t_upper_tail(t_statistic, degrees_of_freedom)))


def classify_regressions(
    baseline_windows: Mapping[str, Sequence[float]],
    current_windows: Mapping[str, Sequence[float]],
    *,
    metric_directions: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    blocking_markers: Sequence[str] = DEFAULT_BLOCKING_METRIC_MARKERS,
) -> RegressionClassification:
    """Classify metric regressions after family-wide false-discovery control.

    Every metric present in both window mappings is tested for a one-sided
    regression, the whole family of p-values is corrected with Benjamini-Hochberg
    at ``alpha``, and each surviving regression is labelled ``blocking`` when the
    metric is leakage-first or ``advisory`` otherwise (for example an F1-only
    movement).

    Args:
        baseline_windows: Metric name to its earlier nightly window.
        current_windows: Metric name to its recent nightly window.
        metric_directions: Optional exact direction overrides per metric.
        alpha: Target false-discovery rate in ``(0, 1]``.
        blocking_markers: Case-insensitive substrings marking leakage-first
            metrics whose regression blocks the release.

    Returns:
        A deterministic :class:`RegressionClassification` verdict payload.
    """

    _validate_alpha(alpha)
    markers = tuple(str(marker).lower() for marker in blocking_markers)
    metrics = sorted(set(baseline_windows) & set(current_windows))

    directions: dict[str, str] = {}
    means: dict[str, tuple[float, float, float]] = {}
    p_values: dict[str, float] = {}
    for metric in metrics:
        direction = _resolve_direction(metric, metric_directions)
        baseline = [float(value) for value in baseline_windows[metric]]
        current = [float(value) for value in current_windows[metric]]
        baseline_mean = mean(baseline)
        current_mean = mean(current)
        directions[metric] = direction
        means[metric] = (baseline_mean, current_mean, current_mean - baseline_mean)
        p_values[metric] = regression_p_value(
            baseline,
            current,
            direction=direction,
        )

    correction = benjamini_hochberg(p_values, alpha) if p_values else None

    signals = []
    for metric in metrics:
        baseline_mean, current_mean, delta = means[metric]
        direction = directions[metric]
        decision = correction.decision(metric) if correction is not None else None
        adjusted = decision.adjusted_p_value if decision is not None else 1.0
        worse = _worse_effect(direction, baseline_mean, current_mean) > 0.0
        significant = bool(decision.significant) if decision is not None else False
        significant = significant and worse
        family = (
            LEAKAGE_FAMILY if _is_blocking_metric(metric, markers) else OTHER_FAMILY
        )
        if not significant:
            severity = NONE
        elif family == LEAKAGE_FAMILY:
            severity = BLOCKING
        else:
            severity = ADVISORY
        signals.append(
            MetricRegressionSignal(
                metric=metric,
                family=family,
                direction=direction,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                delta=delta,
                p_value=p_values[metric],
                adjusted_p_value=adjusted,
                significant=significant,
                severity=severity,
            )
        )

    return RegressionClassification(alpha=alpha, signals=tuple(signals))


def scan_nightly_regressions(
    series: Mapping[str, Sequence[float]],
    *,
    window: int,
    baseline_window: int,
    step: int | None = None,
    confirmations: int = 3,
    metric_directions: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    blocking_markers: Sequence[str] = DEFAULT_BLOCKING_METRIC_MARKERS,
) -> tuple[NightlyRegressionWindow, ...]:
    """Monitor nightly metric series against a fixed leading reference baseline.

    The first ``baseline_window`` nights form the established green baseline. A
    ``window``-night current window then slides across the remaining nights, and
    each position is classified with :func:`classify_regressions` against that
    fixed baseline. Using a fixed reference (rather than a trailing one) keeps a
    sustained step-change visible for every post-change window instead of only
    the single window that straddles the change.

    A window reports ``blocking`` only once a leakage-family metric has been
    significant for ``confirmations`` consecutive windows. Consecutive current
    windows are non-overlapping by default (``step`` defaults to ``window``), so
    those confirmations are independent evidence and a pure-noise history clears
    the gate while a real step-change is confirmed within a few windows.

    Args:
        series: Metric name to its full, equal-length nightly value sequence.
        window: Number of nights in each sliding current window.
        baseline_window: Number of leading nights forming the fixed baseline.
        step: Stride between current-window starts (defaults to ``window``).
        confirmations: Consecutive significant windows a leakage metric needs
            before the release is blocked.
        metric_directions: Optional exact direction overrides per metric.
        alpha: Target false-discovery rate in ``(0, 1]``.
        blocking_markers: Case-insensitive leakage-first substrings.

    Returns:
        One :class:`NightlyRegressionWindow` per evaluated window, in order.
    """

    if window < 1:
        raise ValueError("window must be at least 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be at least 1")
    if confirmations < 1:
        raise ValueError("confirmations must be at least 1")
    stride = window if step is None else step
    if stride < 1:
        raise ValueError("step must be at least 1")
    if not series:
        return ()

    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        raise ValueError("all metric series must share the same length")
    length = lengths.pop()

    baseline_windows = {
        metric: values[:baseline_window] for metric, values in series.items()
    }

    windows: list[NightlyRegressionWindow] = []
    streaks: dict[str, int] = {}
    for current_start in range(baseline_window, length - window + 1, stride):
        current_windows = {
            metric: values[current_start : current_start + window]
            for metric, values in series.items()
        }
        classification = classify_regressions(
            baseline_windows,
            current_windows,
            metric_directions=metric_directions,
            alpha=alpha,
            blocking_markers=blocking_markers,
        )
        blocking_now = set(classification.blocking_metrics)
        confirmed = []
        for metric in classification.blocking_metrics:
            streaks[metric] = streaks.get(metric, 0) + 1
            if streaks[metric] >= confirmations:
                confirmed.append(metric)
        for metric in list(streaks):
            if metric not in blocking_now:
                streaks[metric] = 0
        windows.append(
            NightlyRegressionWindow(
                night_index=current_start + window - 1,
                current_start=current_start,
                classification=classification,
                blocking=bool(confirmed),
                confirmed_metrics=tuple(sorted(confirmed)),
            )
        )
    return tuple(windows)


def _worse_effect(
    direction: str,
    baseline_mean: float,
    current_mean: float,
) -> float:
    if direction == HIGHER_IS_BETTER:
        return baseline_mean - current_mean
    return current_mean - baseline_mean


def _welch_degrees_of_freedom(
    baseline_var: float,
    current_var: float,
    baseline_n: int,
    current_n: int,
) -> float:
    baseline_term = baseline_var / baseline_n
    current_term = current_var / current_n
    numerator = (baseline_term + current_term) ** 2
    denominator = 0.0
    if baseline_n > 1:
        denominator += baseline_term**2 / (baseline_n - 1)
    if current_n > 1:
        denominator += current_term**2 / (current_n - 1)
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def _student_t_upper_tail(t_statistic: float, degrees_of_freedom: float) -> float:
    """Return the one-sided upper-tail probability ``P(T >= t_statistic)``.

    Uses the regularized incomplete beta identity for the Student-t CDF, so the
    small-sample tails are correct rather than the anti-conservative normal
    approximation.
    """

    if degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be positive")
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    half_beta = 0.5 * _regularized_incomplete_beta(
        degrees_of_freedom / 2.0,
        0.5,
        x,
    )
    if t_statistic > 0.0:
        return half_beta
    return 1.0 - half_beta


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return the regularized incomplete beta ``I_x(a, b)``.

    Implements the standard continued-fraction expansion with the symmetry
    reflection for numerical stability, matching the classic ``betai`` routine.
    """

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    max_iterations = 300
    epsilon = 3e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m = float(iteration)
        numerator = m * (b - m) * x / ((qam + 2.0 * m) * (a + 2.0 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + 2.0 * m) * (qap + 2.0 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def _resolve_direction(
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


def _is_blocking_metric(metric: str, markers: Sequence[str]) -> bool:
    normalized = metric.lower()
    return any(marker in normalized for marker in markers)


def _validate_alpha(alpha: float) -> None:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise ValueError("alpha must be a finite value in (0, 1]")


def _validate_p_value(key: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"p-value for {key!r} must be a finite value in [0, 1]")
    return float(value)


__all__ = [
    "ADVISORY",
    "BLOCKING",
    "CLEAN",
    "DEFAULT_ALPHA",
    "DEFAULT_BLOCKING_METRIC_MARKERS",
    "LEAKAGE_FAMILY",
    "NONE",
    "OTHER_FAMILY",
    "REGRESSION_SIGNIFICANCE_SCHEMA_VERSION",
    "FalseDiscoveryDecision",
    "FalseDiscoveryResult",
    "MetricRegressionSignal",
    "NightlyRegressionWindow",
    "RegressionClassification",
    "benjamini_hochberg",
    "classify_regressions",
    "regression_p_value",
    "scan_nightly_regressions",
]
