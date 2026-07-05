"""Unit tests for the throughput-versus-accuracy Pareto frontier report."""

from __future__ import annotations

import json

import pytest

from openmed.eval import (
    FrontierPoint,
    FrontierReport,
    frontier_point_from_reports,
    frontier_report,
)
from openmed.eval.frontier import FrontierEntry


def _entry_by_label(report: FrontierReport, label: str) -> FrontierEntry:
    for entry in report.entries:
        if entry.point.label == label:
            return entry
    raise AssertionError(f"no entry with label {label!r}")


def _frontier_labels(report: FrontierReport) -> set[str]:
    return {entry.point.label for entry in report.frontier}


def test_planted_dominated_point_is_flagged_with_its_dominator() -> None:
    # "slow-bad" is worse than "fast-good" on both throughput and accuracy,
    # so it must be dominated and record the dominator's label.
    points = [
        FrontierPoint(label="fast-good", throughput=100.0, accuracy=0.90),
        FrontierPoint(label="slow-bad", throughput=40.0, accuracy=0.70),
    ]

    report = frontier_report(points)

    dominated = _entry_by_label(report, "slow-bad")
    assert dominated.on_frontier is False
    assert dominated.dominated_by == "fast-good"

    survivor = _entry_by_label(report, "fast-good")
    assert survivor.on_frontier is True
    assert survivor.dominated_by is None


def test_non_dominated_points_form_the_frontier() -> None:
    # A speed/accuracy trade-off: each trades one axis for the other, so all
    # three are non-dominated. The fourth point is strictly worse than the
    # fast, low-accuracy corner and must fall off.
    points = [
        FrontierPoint(label="fast", throughput=200.0, accuracy=0.80),
        FrontierPoint(label="balanced", throughput=120.0, accuracy=0.90),
        FrontierPoint(label="accurate", throughput=60.0, accuracy=0.97),
        FrontierPoint(label="dominated", throughput=100.0, accuracy=0.75),
    ]

    report = frontier_report(points)

    assert _frontier_labels(report) == {"fast", "balanced", "accurate"}
    assert report.frontier_count == 3
    assert report.dominated_count == 1
    # "dominated" (100 docs/s, 0.75) is beaten by both "fast" and "balanced";
    # the strongest dominator (highest throughput first) is deterministically
    # recorded.
    assert _entry_by_label(report, "dominated").dominated_by == "fast"


def test_single_point_input_is_always_on_the_frontier() -> None:
    report = frontier_report([FrontierPoint("solo", throughput=10.0, accuracy=0.5)])

    assert report.point_count == 1
    assert report.frontier_count == 1
    assert report.dominated_count == 0
    entry = report.entries[0]
    assert entry.on_frontier is True
    assert entry.dominated_by is None


def test_empty_input_yields_an_empty_frontier() -> None:
    report = frontier_report([])

    assert report.entries == []
    assert report.frontier_count == 0
    assert report.dominated_count == 0


def test_identical_points_tie_and_both_stay_on_the_frontier() -> None:
    # Equal objectives => neither dominates the other; both remain.
    points = [
        FrontierPoint(label="twin-a", throughput=50.0, accuracy=0.8),
        FrontierPoint(label="twin-b", throughput=50.0, accuracy=0.8),
    ]

    report = frontier_report(points)

    assert _frontier_labels(report) == {"twin-a", "twin-b"}
    assert report.dominated_count == 0


def test_equal_throughput_higher_accuracy_dominates_the_tie_break_axis() -> None:
    # Same speed, but one is strictly more accurate -> it dominates.
    points = [
        FrontierPoint(label="same-speed-better", throughput=80.0, accuracy=0.95),
        FrontierPoint(label="same-speed-worse", throughput=80.0, accuracy=0.85),
    ]

    report = frontier_report(points)

    assert _frontier_labels(report) == {"same-speed-better"}
    assert _entry_by_label(report, "same-speed-worse").dominated_by == (
        "same-speed-better"
    )


def test_leakage_is_a_lower_is_better_objective() -> None:
    # Same throughput and accuracy; the higher-leakage variant is dominated by
    # the lower-leakage one.
    points = [
        FrontierPoint(label="clean", throughput=100.0, accuracy=0.9, leakage=0.001),
        FrontierPoint(label="leaky", throughput=100.0, accuracy=0.9, leakage=0.05),
    ]

    report = frontier_report(points)

    assert _frontier_labels(report) == {"clean"}
    assert _entry_by_label(report, "leaky").dominated_by == "clean"


def test_lower_leakage_keeps_a_slower_variant_on_the_frontier() -> None:
    # "safe" is slower but leaks far less, so it is not dominated by "quick".
    points = [
        FrontierPoint(label="quick", throughput=150.0, accuracy=0.9, leakage=0.02),
        FrontierPoint(label="safe", throughput=90.0, accuracy=0.9, leakage=0.0),
    ]

    report = frontier_report(points)

    assert _frontier_labels(report) == {"quick", "safe"}
    assert report.dominated_count == 0


def test_report_order_matches_input_order() -> None:
    labels = ["c", "a", "b", "d"]
    points = [
        FrontierPoint(label=label, throughput=float(index + 1), accuracy=0.5)
        for index, label in enumerate(labels)
    ]

    report = frontier_report(points)

    assert [entry.point.label for entry in report.entries] == labels


def test_dominator_selection_is_deterministic_regardless_of_input_order() -> None:
    weak = FrontierPoint(label="weak", throughput=10.0, accuracy=0.10)
    strong_a = FrontierPoint(label="strong-a", throughput=100.0, accuracy=0.90)
    strong_b = FrontierPoint(label="strong-b", throughput=100.0, accuracy=0.90)

    forward = frontier_report([weak, strong_a, strong_b])
    reverse = frontier_report([strong_b, strong_a, weak])

    # Both strong points are equally dominant ties, so the label tie-break must
    # pick the same dominator no matter the ordering.
    assert _entry_by_label(forward, "weak").dominated_by == "strong-a"
    assert _entry_by_label(reverse, "weak").dominated_by == "strong-a"


def test_mapping_inputs_are_accepted() -> None:
    report = frontier_report(
        [
            {"label": "m1", "throughput": 100.0, "accuracy": 0.9},
            {"variant": "m2", "docs_per_second": 40.0, "accuracy": 0.7},
        ]
    )

    assert report.point_count == 2
    assert _frontier_labels(report) == {"m1"}
    assert _entry_by_label(report, "m2").dominated_by == "m1"


def test_json_serialization_is_deterministic_and_round_trips_shape() -> None:
    points = [
        FrontierPoint(label="fast", throughput=200.0, accuracy=0.80, leakage=0.01),
        FrontierPoint(label="accurate", throughput=60.0, accuracy=0.97, leakage=0.0),
        FrontierPoint(label="bad", throughput=50.0, accuracy=0.70, leakage=0.02),
    ]

    report = frontier_report(points, accuracy_metric="exact_span_f1")

    first = report.to_json()
    second = report.to_json()
    assert first == second  # deterministic

    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["accuracy_metric"] == "exact_span_f1"
    assert payload["point_count"] == 3
    assert payload["frontier_count"] == 2
    assert payload["dominated_count"] == 1
    assert set(payload["frontier_labels"]) == {"fast", "accurate"}

    # Every entry carries the expected keys.
    for entry in payload["entries"]:
        assert set(entry) == {"dominated_by", "on_frontier", "point"}
        assert set(entry["point"]) == {
            "accuracy",
            "label",
            "leakage",
            "metadata",
            "throughput",
        }

    # sort_keys makes the top-level ordering stable and canonical.
    assert list(payload.keys()) == sorted(payload.keys())


def test_markdown_lists_every_configuration_with_status() -> None:
    points = [
        FrontierPoint(label="fast", throughput=200.0, accuracy=0.80),
        FrontierPoint(label="dominated", throughput=100.0, accuracy=0.70),
    ]

    report = frontier_report(points, generated_at="2026-07-05T00:00:00Z")
    markdown = report.to_markdown()

    assert "# Throughput vs Accuracy Frontier" in markdown
    assert "`fast`" in markdown
    assert "`dominated`" in markdown
    # Dominated row records its dominator.
    assert "`fast`" in markdown.splitlines()[-1] or any(
        "`dominated`" in line and "`fast`" in line for line in markdown.splitlines()
    )
    assert "2026-07-05T00:00:00Z" in markdown


def test_chart_data_splits_frontier_and_dominated_series() -> None:
    points = [
        FrontierPoint(label="fast", throughput=200.0, accuracy=0.80),
        FrontierPoint(label="accurate", throughput=60.0, accuracy=0.97),
        FrontierPoint(label="dominated", throughput=50.0, accuracy=0.70),
    ]

    chart = frontier_report(points).chart_data()

    assert chart["x_axis"]["key"] == "throughput"
    frontier_labels = [item["label"] for item in chart["frontier"]]
    # Frontier series is sorted ascending by throughput.
    throughputs = [item["throughput"] for item in chart["frontier"]]
    assert throughputs == sorted(throughputs)
    assert set(frontier_labels) == {"fast", "accurate"}
    assert [item["label"] for item in chart["dominated"]] == ["dominated"]


def test_invalid_point_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        FrontierPoint.from_mapping({"label": "x", "throughput": "fast", "accuracy": 1})
    with pytest.raises(ValueError):
        FrontierPoint.from_mapping({"label": "x", "throughput": 1.0, "accuracy": None})


# --- Assembling points from existing PerfReport / BenchmarkReport outputs -----


class _FakePerfReport:
    """Minimal stand-in exposing the PerfReport surface the frontier reads."""

    def __init__(self, model_name: str, docs_per_second: float) -> None:
        self.model_name = model_name
        self.docs_per_second = docs_per_second
        self.device = "cpu"
        self.tier = "base"
        self.canonical_tier = "Base"


class _FakeBenchmarkReport:
    """Minimal stand-in exposing the BenchmarkReport surface the frontier reads."""

    def __init__(self, model_name: str, f1: float, leakage: float) -> None:
        self.model_name = model_name
        self.suite = "phi-en"
        self.device = "cpu"
        self.metrics = {
            "exact_span_f1": {"f1": f1, "precision": f1, "recall": f1},
            "leakage": {"overall": leakage},
        }


def test_frontier_point_from_reports_reuses_measured_numbers() -> None:
    perf = _FakePerfReport("clinical-e5-small@int8", docs_per_second=180.0)
    benchmark = _FakeBenchmarkReport("clinical-e5-small@int8", f1=0.93, leakage=0.004)

    point = frontier_point_from_reports(perf, benchmark)

    assert point.label == "clinical-e5-small@int8"
    assert point.throughput == 180.0
    assert point.accuracy == 0.93
    assert point.leakage == 0.004
    assert point.metadata["accuracy_key"] == "exact_span_f1.f1"
    assert point.metadata["leakage_key"] == "leakage.overall"


def test_frontier_from_assembled_report_points_matches_direct_computation() -> None:
    variants = [
        ("fp32", 60.0, 0.97, 0.0),
        ("int8", 150.0, 0.94, 0.002),
        ("int4", 90.0, 0.80, 0.03),
    ]
    assembled = [
        frontier_point_from_reports(
            _FakePerfReport(name, docs_per_second=tput),
            _FakeBenchmarkReport(name, f1=f1, leakage=leak),
        )
        for name, tput, f1, leak in variants
    ]

    report = frontier_report(assembled, accuracy_metric="exact_span_f1.f1")

    # int4 is slower AND less accurate AND leakier than int8 -> dominated.
    assert _entry_by_label(report, "int4").on_frontier is False
    assert _entry_by_label(report, "int4").dominated_by == "int8"
    # fp32 (most accurate) and int8 (fastest) are the trade-off frontier.
    assert _frontier_labels(report) == {"fp32", "int8"}


def test_frontier_point_from_reports_requires_an_accuracy_metric() -> None:
    perf = _FakePerfReport("m", docs_per_second=10.0)
    benchmark = _FakeBenchmarkReport("m", f1=0.5, leakage=0.0)
    benchmark.metrics = {"latency": {"p50_ms": 1.0}}  # no accuracy key present

    with pytest.raises(ValueError, match="no accuracy metric"):
        frontier_point_from_reports(perf, benchmark)
