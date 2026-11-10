"""Tests for regression-escape tracking over benchmark history."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

from openmed.eval.regression_tracker import (
    ESCAPED,
    GATED,
    RECOVERED,
    REGRESSION_TRACKER_SCHEMA_VERSION,
    render_regression_dashboard,
    track_regression_escapes,
    write_regression_dashboard,
)


class _BalancedHTMLParser(HTMLParser):
    _VOID_TAGS = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self._VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        assert self.stack, f"unexpected closing tag: {tag}"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


def _assert_balanced_html(document: str) -> None:
    parser = _BalancedHTMLParser()
    parser.feed(document)
    parser.close()
    assert parser.stack == []


def _report(
    release: str,
    recall: float,
    *,
    suite: str = "clinical_phi",
    gate_passed: bool = True,
    released_at: str | None = None,
    extra_metrics: dict | None = None,
) -> dict:
    metrics = {"per_label_recall": {"PERSON": recall}}
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "suite": suite,
        "model_name": "synthetic-model",
        "device": "cpu",
        "fixture_count": 12,
        "metrics": metrics,
        "metadata": {
            "release": release,
            "released_at": released_at,
            "gate_passed": gate_passed,
        },
    }


def test_tracks_gated_escape_and_recovery_against_last_shipped_release() -> None:
    summary = track_regression_escapes(
        [
            _report("v1.0", 0.99, released_at="2026-01-01"),
            _report(
                "v1.1-rc1",
                0.90,
                gate_passed=False,
                released_at="2026-01-05",
            ),
            _report("v1.1", 0.93, released_at="2026-01-08"),
            _report("v1.2", 0.97, released_at="2026-01-15"),
            _report("v1.3", 0.99, released_at="2026-01-22"),
        ]
    )

    assert summary.release_count == 5
    assert [event.classification for event in summary.events] == [GATED, RECOVERED]

    gated, recovered = summary.events
    assert gated.baseline_release == "v1.0"
    assert gated.detected_release == "v1.1-rc1"
    assert gated.recovery_release is None

    assert recovered.baseline_release == "v1.0"
    assert recovered.detected_release == "v1.1"
    assert recovered.recovery_release == "v1.3"
    assert recovered.time_to_recovery_releases == 2
    assert recovered.time_to_recovery_days == 14
    assert summary.gated_count == 1
    assert summary.escape_count == 1
    assert summary.recovered_count == 1
    assert summary.open_escape_count == 0
    assert summary.release_ready is True
    assert summary.mean_time_to_recovery_releases == 2.0
    assert summary.mean_time_to_recovery_days == 14.0


def test_unrecovered_recurring_escape_blocks_release_readiness() -> None:
    summary = track_regression_escapes(
        [
            _report("v1", 0.99),
            _report("v2", 0.95),
            _report("v3", 0.99),
            _report("v4", 0.96),
        ]
    )

    assert [event.classification for event in summary.events] == [
        RECOVERED,
        ESCAPED,
    ]
    assert summary.escape_count == 2
    assert summary.open_escape_count == 1
    assert summary.release_ready is False

    [hotspot] = summary.hotspots
    assert hotspot.metric == "per_label_recall.PERSON"
    assert hotspot.regression_count == 2
    assert hotspot.escape_count == 2
    assert hotspot.recovered_count == 1
    assert hotspot.open_escape_count == 1


def test_lower_is_better_metric_recovers_at_or_below_pre_escape_value() -> None:
    summary = track_regression_escapes(
        [
            _report("v1", 0.99, extra_metrics={"leakage": {"overall": 0.01}}),
            _report("v2", 0.99, extra_metrics={"leakage": {"overall": 0.03}}),
            _report("v3", 0.99, extra_metrics={"leakage": {"overall": 0.01}}),
        ]
    )

    [event] = summary.events
    assert event.metric == "leakage.overall"
    assert event.classification == RECOVERED
    assert event.recovery_release == "v3"
    assert event.time_to_recovery_releases == 1


def test_hotspot_ranking_is_deterministic_for_metric_mapping_order() -> None:
    first = [
        _report("v1", 0.99, extra_metrics={"z_score": 0.99, "a_score": 0.99}),
        _report("v2", 0.98, extra_metrics={"z_score": 0.95, "a_score": 0.95}),
    ]
    second = [
        _report("v1", 0.99, extra_metrics={"a_score": 0.99, "z_score": 0.99}),
        _report("v2", 0.98, extra_metrics={"a_score": 0.95, "z_score": 0.95}),
    ]

    first_summary = track_regression_escapes(first)
    second_summary = track_regression_escapes(second)

    assert first_summary.to_json() == second_summary.to_json()
    assert [row.metric for row in first_summary.hotspots] == [
        "a_score",
        "per_label_recall.PERSON",
        "z_score",
    ]


def test_multiple_suite_reports_share_one_release_index() -> None:
    summary = track_regression_escapes(
        [
            _report("v1", 0.99, suite="suite-a"),
            _report("v1", 0.98, suite="suite-b"),
            _report("v2", 0.95, suite="suite-a"),
            _report("v2", 0.94, suite="suite-b"),
        ]
    )

    assert summary.release_count == 2
    assert [event.suite for event in summary.events] == ["suite-a", "suite-b"]
    assert all(event.classification == ESCAPED for event in summary.events)


def test_noncontiguous_release_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        track_regression_escapes(
            [
                _report("v1", 0.99, suite="suite-a"),
                _report("v2", 0.99, suite="suite-a"),
                _report("v1", 0.99, suite="suite-b"),
            ]
        )


def test_regression_threshold_ignores_small_metric_deltas() -> None:
    summary = track_regression_escapes(
        [_report("v1", 0.99), _report("v2", 0.985)],
        regression_thresholds={"per_label_recall.PERSON": 0.01},
    )

    assert summary.events == ()
    assert summary.release_ready is True


def test_json_and_dashboard_never_copy_raw_phi_or_source_metadata(tmp_path) -> None:
    reports = [
        _report("v1", 0.99, suite="Jane Doe private suite"),
        _report("v2", 0.95, suite="Jane Doe private suite"),
    ]
    for report in reports:
        report["metrics"] = {
            "per_label_recall": {
                "Jane Doe": report["metrics"]["per_label_recall"]["PERSON"]
            }
        }
        report["metadata"]["raw_note"] = "Jane Doe, MRN 12345678"
        report["predictions"] = ["Jane Doe"]

    summary = track_regression_escapes(reports)
    rendered = summary.to_json() + render_regression_dashboard(summary)

    assert "Jane Doe" not in rendered
    assert "12345678" not in rendered
    assert "predictions" not in rendered
    assert summary.events[0].suite.startswith("suite-")
    assert summary.events[0].metric.startswith("metric-")

    json_path = tmp_path / "readiness.json"
    dashboard_path = tmp_path / "dashboard.html"
    assert summary.write_json(json_path) == json_path
    assert write_regression_dashboard(summary, dashboard_path) == dashboard_path
    assert json.loads(json_path.read_text(encoding="utf-8"))["release_ready"] is False
    assert "Jane Doe" not in dashboard_path.read_text(encoding="utf-8")


def test_dashboard_is_self_contained_balanced_and_deterministic() -> None:
    summary = track_regression_escapes([_report("v1", 0.99), _report("v2", 0.95)])

    first = render_regression_dashboard(summary)
    second = render_regression_dashboard(summary)

    assert first == second
    assert first.startswith("<!doctype html>\n<html")
    assert first.rstrip().endswith("</html>")
    assert "Recurring-regression hotspots" in first
    assert "Regression history" in first
    assert "Mean recovery (days)" in first
    assert not re.search(r"\b(?:src|href)=[\"']https?://", first)
    _assert_balanced_html(first)


def test_summary_schema_is_stable_and_json_is_deterministic() -> None:
    summary = track_regression_escapes([_report("v1", 0.99), _report("v2", 0.95)])
    payload = json.loads(summary.to_json())

    assert payload["schema_version"] == REGRESSION_TRACKER_SCHEMA_VERSION
    assert payload["regression_count"] == 1
    assert payload["escape_count"] == 1
    assert payload["open_escape_count"] == 1
    assert payload["release_ready"] is False
    assert summary.to_json() == summary.to_json()
