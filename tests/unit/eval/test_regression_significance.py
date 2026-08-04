"""Unit tests for false-discovery classification of benchmark regressions."""

from __future__ import annotations

import random

import pytest

from openmed.eval.regression_significance import (
    ADVISORY,
    BLOCKING,
    CLEAN,
    LEAKAGE_FAMILY,
    OTHER_FAMILY,
    REGRESSION_SIGNIFICANCE_SCHEMA_VERSION,
    FalseDiscoveryResult,
    RegressionClassification,
    benjamini_hochberg,
    classify_regressions,
    regression_p_value,
    scan_nightly_regressions,
)


def _gaussian_series(rng: random.Random, mean: float, sigma: float, count: int):
    """Return ``count`` synthetic nightly values around ``mean``."""

    return [rng.gauss(mean, sigma) for _ in range(count)]


def test_benjamini_hochberg_matches_hand_verified_step_up():
    # Critical values (i/m) * 0.05 for m = 5 are 0.01, 0.02, 0.03, 0.04, 0.05.
    # Sorted p-values 0.001, 0.008, 0.039, 0.041, 0.9: the largest rank that
    # clears its critical value is rank 2, so only "a" and "b" are rejected --
    # "c" survives at raw 0.039 even though it sits under the naive 0.05 line.
    result = benjamini_hochberg(
        {"a": 0.001, "b": 0.008, "c": 0.039, "d": 0.041, "e": 0.9},
        alpha=0.05,
    )

    assert isinstance(result, FalseDiscoveryResult)
    assert result.significant == ("a", "b")
    adjusted = result.adjusted_p_values()
    assert adjusted["a"] == pytest.approx(0.005)
    assert adjusted["b"] == pytest.approx(0.02)
    assert adjusted["c"] == pytest.approx(0.05125)
    assert adjusted["d"] == pytest.approx(0.05125)
    assert adjusted["e"] == pytest.approx(0.9)
    assert result.decision("c").significant is False
    assert result.schema_version == REGRESSION_SIGNIFICANCE_SCHEMA_VERSION


def test_benjamini_hochberg_rejects_all_when_every_p_is_tiny():
    result = benjamini_hochberg({"m1": 0.0001, "m2": 0.0002, "m3": 0.0003})

    assert result.significant == ("m1", "m2", "m3")


def test_benjamini_hochberg_is_more_conservative_than_uncorrected():
    # A single p-value just under alpha within a large family is not rejected,
    # because the smallest critical value is alpha / m.
    result = benjamini_hochberg(
        {f"metric_{index}": 0.9 for index in range(19)} | {"metric_hit": 0.049},
        alpha=0.05,
    )

    assert result.significant == ()
    assert result.decision("metric_hit").significant is False


def test_benjamini_hochberg_validates_inputs():
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg({"a": 0.1}, alpha=0.0)
    with pytest.raises(ValueError, match="p-value"):
        benjamini_hochberg({"a": 1.5})


def test_regression_p_value_is_directional_and_deterministic():
    baseline = [0.02, 0.021, 0.019, 0.02, 0.022]
    worse_leakage = [0.09, 0.088, 0.091, 0.089, 0.092]

    regressed = regression_p_value(
        baseline,
        worse_leakage,
        direction="lower_is_better",
    )
    improved = regression_p_value(
        worse_leakage,
        baseline,
        direction="lower_is_better",
    )

    assert regressed < 0.01
    # The reverse comparison is an improvement, so it is never significant.
    assert improved > 0.5
    # No randomness: identical inputs reproduce byte-for-byte.
    assert (
        regression_p_value(baseline, worse_leakage, direction="lower_is_better")
        == regressed
    )


def test_leakage_regression_blocks_while_f1_movement_is_advisory():
    baseline = {
        "leakage_rate": [0.02, 0.021, 0.019, 0.02, 0.022, 0.02],
        "exact_span_f1": [0.90, 0.905, 0.898, 0.902, 0.9, 0.901],
    }
    current = {
        # Clear leakage step-change -> blocking.
        "leakage_rate": [0.12, 0.118, 0.121, 0.119, 0.122, 0.12],
        # Clear F1 drop -> significant but advisory only.
        "exact_span_f1": [0.60, 0.605, 0.598, 0.602, 0.6, 0.601],
    }

    result = classify_regressions(baseline, current, alpha=0.05)

    assert isinstance(result, RegressionClassification)
    assert result.blocking is True
    assert result.verdict == BLOCKING
    assert result.blocking_metrics == ("leakage_rate",)
    assert result.advisory_metrics == ("exact_span_f1",)

    leakage_signal = result.signal("leakage_rate")
    assert leakage_signal.family == LEAKAGE_FAMILY
    assert leakage_signal.severity == BLOCKING

    f1_signal = result.signal("exact_span_f1")
    assert f1_signal.family == OTHER_FAMILY
    assert f1_signal.severity == ADVISORY


def test_verdict_is_advisory_when_only_f1_regresses():
    baseline = {
        "leakage_rate": [0.02, 0.021, 0.019, 0.02, 0.022, 0.02],
        "exact_span_f1": [0.90, 0.905, 0.898, 0.902, 0.9, 0.901],
    }
    current = {
        "leakage_rate": [0.0201, 0.0205, 0.0198, 0.02, 0.021, 0.0202],
        "exact_span_f1": [0.55, 0.552, 0.548, 0.551, 0.55, 0.549],
    }

    result = classify_regressions(baseline, current, alpha=0.05)

    assert result.blocking is False
    assert result.verdict == ADVISORY
    assert result.blocking_metrics == ()
    assert result.advisory_metrics == ("exact_span_f1",)


def test_injected_leakage_step_change_is_flagged_within_three_windows():
    # Acceptance criterion 1: a real leakage step-change surfaces quickly. A
    # fixed leading baseline keeps every post-change window significant, and the
    # non-overlapping (default step == window) confirmation windows are
    # independent evidence, so a three-window debounce still confirms fast.
    window = 5
    baseline_window = 10
    change_night = 15
    detection_window_lags: list[int] = []
    for trial in range(40):
        rng = random.Random(1000 + trial)
        nights = change_night + 4 * window
        leakage = [
            abs(rng.gauss(0.02 if night < change_night else 0.11, 0.004))
            for night in range(nights)
        ]
        f1 = _gaussian_series(rng, 0.9, 0.01, nights)
        series = {"leakage_rate": leakage, "exact_span_f1": f1}

        scan = scan_nightly_regressions(
            series,
            window=window,
            baseline_window=baseline_window,
            confirmations=3,
        )
        post_change = [item for item in scan if item.current_start >= change_night]
        blocking_positions = [
            index for index, item in enumerate(post_change) if item.blocking
        ]
        assert blocking_positions, f"trial {trial} never flagged the step-change"
        assert "leakage_rate" in post_change[blocking_positions[0]].confirmed_metrics
        detection_window_lags.append(blocking_positions[0])

    # Confirmed within three post-change nightly windows (0-indexed <= 2).
    assert max(detection_window_lags) <= 2


def test_pure_noise_false_positive_rate_stays_at_or_below_five_percent():
    # Acceptance criterion 1 (control side): matched null trials rarely block.
    window = 5
    blocking_trials = 0
    trials = 200
    for trial in range(trials):
        rng = random.Random(7000 + trial)
        length = 2 * window
        series = {
            "leakage_rate": [abs(rng.gauss(0.02, 0.004)) for _ in range(length)],
            "critical_leakage_count": [abs(rng.gauss(1.0, 0.2)) for _ in range(length)],
            "exact_span_f1": _gaussian_series(rng, 0.9, 0.01, length),
            "character_recall": _gaussian_series(rng, 0.95, 0.01, length),
        }
        baseline = {name: values[:window] for name, values in series.items()}
        current = {name: values[window:] for name, values in series.items()}
        result = classify_regressions(baseline, current, alpha=0.05)
        if result.blocking:
            blocking_trials += 1

    assert blocking_trials / trials <= 0.05


def test_fifty_nights_of_noise_produce_zero_blocking_flags():
    # Acceptance criterion 2: a pure within-noise history blocks nothing.
    rng = random.Random(4242)
    nights = 50
    series = {
        "leakage_rate": [abs(rng.gauss(0.02, 0.004)) for _ in range(nights)],
        "critical_leakage_count": [abs(rng.gauss(1.0, 0.2)) for _ in range(nights)],
        "exact_span_f1": _gaussian_series(rng, 0.9, 0.01, nights),
        "character_recall": _gaussian_series(rng, 0.95, 0.01, nights),
    }

    scan = scan_nightly_regressions(
        series, window=5, baseline_window=10, confirmations=3
    )

    assert scan  # the scan actually evaluated windows
    assert all(not item.blocking for item in scan)
    assert sum(item.blocking for item in scan) == 0


def test_classification_payload_is_deterministic_and_json_ready():
    baseline = {
        "leakage_rate": [0.02, 0.021, 0.019, 0.02, 0.022, 0.02],
        "exact_span_f1": [0.90, 0.905, 0.898, 0.902, 0.9, 0.901],
    }
    current = {
        "leakage_rate": [0.12, 0.118, 0.121, 0.119, 0.122, 0.12],
        "exact_span_f1": [0.90, 0.905, 0.898, 0.902, 0.9, 0.901],
    }

    first = classify_regressions(baseline, current, alpha=0.05)
    second = classify_regressions(baseline, current, alpha=0.05)

    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()
    payload = first.to_dict()
    assert payload["verdict"] == BLOCKING
    assert payload["blocking_metrics"] == ["leakage_rate"]
    assert [signal["metric"] for signal in payload["signals"]] == [
        "exact_span_f1",
        "leakage_rate",
    ]


def test_clean_verdict_when_nothing_regresses():
    baseline = {
        "leakage_rate": [0.02, 0.021, 0.019, 0.02, 0.022, 0.02],
        "exact_span_f1": [0.90, 0.905, 0.898, 0.902, 0.9, 0.901],
    }

    result = classify_regressions(baseline, baseline, alpha=0.05)

    assert result.verdict == CLEAN
    assert result.blocking is False
    assert result.significant_metrics == ()
