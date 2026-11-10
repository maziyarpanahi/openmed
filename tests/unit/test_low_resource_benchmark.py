"""Unit tests for the low-resource benchmark gate."""

from __future__ import annotations

import json

import pytest

from scripts.benchmarks import low_resource_deid


def test_synthetic_notes_are_deterministic_and_sized():
    first = low_resource_deid.synthetic_notes(3)
    second = low_resource_deid.synthetic_notes(3)

    assert first == second
    assert len(first) == 3
    assert all("no real patient information" in note for note in first)


def test_load_baseline_requires_positive_peak(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"peak_rss_mib": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="positive peak_rss_mib"):
        low_resource_deid.load_baseline(baseline)


def test_load_baseline_requires_model_revision(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"peak_rss_mib": 100}), encoding="utf-8")

    with pytest.raises(ValueError, match="model_revision"):
        low_resource_deid.load_baseline(baseline)


def test_regression_gate_allows_ten_percent_growth():
    low_resource_deid.enforce_limits(
        {"peak_rss_mib": 110.0},
        max_peak_rss_mib=2560.0,
        baseline={"peak_rss_mib": 100.0},
        max_regression_percent=10.0,
    )


def test_regression_gate_rejects_more_than_ten_percent_growth():
    with pytest.raises(RuntimeError, match="regression limit"):
        low_resource_deid.enforce_limits(
            {"peak_rss_mib": 110.1},
            max_peak_rss_mib=2560.0,
            baseline={"peak_rss_mib": 100.0},
            max_regression_percent=10.0,
        )


def test_regression_gate_rejects_model_revision_mismatch():
    with pytest.raises(RuntimeError, match="model revision"):
        low_resource_deid.enforce_limits(
            {"peak_rss_mib": 100.0, "model_revision": "new"},
            max_peak_rss_mib=2560.0,
            baseline={"peak_rss_mib": 100.0, "model_revision": "baseline"},
            max_regression_percent=10.0,
        )


def test_absolute_peak_limit_is_strict():
    with pytest.raises(RuntimeError, match="must stay below"):
        low_resource_deid.enforce_limits(
            {"peak_rss_mib": 2560.0},
            max_peak_rss_mib=2560.0,
            baseline=None,
            max_regression_percent=10.0,
        )


def test_cgroup_limit_requires_four_gib_window():
    low_resource_deid.validate_cgroup_limit(4 * low_resource_deid.GIB, 4.0)

    with pytest.raises(RuntimeError, match="Expected a 4.00 GiB"):
        low_resource_deid.validate_cgroup_limit(None, 4.0)
