"""Tests for deterministic per-language leakage dashboard artifacts."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.leakage_dashboard import (
    LEAKAGE_DASHBOARD_ARTIFACT_TYPE,
    LeakageDashboard,
    build_leakage_dashboard,
    render_leakage_dashboard_html,
    write_leakage_dashboard,
)
from openmed.eval.leakage_heatmap import compute_leakage_heatmap
from openmed.eval.report import BenchmarkReport


class _BalancedHTMLParser(HTMLParser):
    _VOID_TAGS = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag not in self._VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        assert self.stack, f"unexpected closing tag: {tag}"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


def _span(start: int, end: int, label: str, language: str) -> dict[str, Any]:
    return {"start": start, "end": end, "label": label, "language": language}


def _reports() -> tuple[BenchmarkReport, BenchmarkReport]:
    first = compute_leakage_heatmap(
        [
            _span(0, 10, "PERSON", "en"),
            _span(20, 25, "DATE", "en"),
            _span(30, 38, "ID_NUM", "fr"),
        ],
        [
            _span(0, 5, "PERSON", "en"),
            _span(20, 25, "DATE", "en"),
        ],
    )
    second = compute_leakage_heatmap(
        [
            _span(0, 10, "PERSON", "en"),
            _span(20, 28, "ID_NUM", "fr"),
            _span(30, 34, "PERSON", "fr"),
        ],
        [
            _span(0, 10, "PERSON", "en"),
            _span(20, 24, "ID_NUM", "fr"),
        ],
    )
    return (
        BenchmarkReport(
            suite="synthetic-multilingual",
            model_name="Patient Alice 123-45-6789",
            device="cpu",
            fixture_count=3,
            metrics={"leakage_heatmap": first.to_dict()},
            metadata={"raw_note": "Patient Alice 123-45-6789"},
        ),
        BenchmarkReport(
            suite="synthetic-multilingual",
            model_name="Patient Bob FR-ID-44",
            device="cpu",
            fixture_count=3,
            metrics={"leakage_heatmap": second.to_dict()},
            metadata={"raw_note": "Patient Bob FR-ID-44"},
        ),
    )


def test_dashboard_aggregates_trends_categories_and_worst_labels() -> None:
    dashboard = build_leakage_dashboard(_reports(), worst_n=2, thresholds=0.3)
    summaries = {item.language: item for item in dashboard.languages}

    assert dashboard.run_count == 2
    assert set(summaries) == SUPPORTED_LANGUAGES

    english = summaries["en"]
    assert english.leaked_chars == 5
    assert english.total_chars == 25
    assert english.rate == pytest.approx(0.2)
    assert [point.rate for point in english.trend] == pytest.approx([1 / 3, 0.0])
    assert [item.label for item in english.worst_labels] == ["PERSON"]
    assert english.threshold_passed is True

    french = summaries["fr"]
    assert french.leaked_chars == 16
    assert french.total_chars == 20
    assert french.rate == pytest.approx(0.8)
    assert [point.rate for point in french.trend] == pytest.approx([1.0, 2 / 3])
    assert [item.label for item in french.worst_labels] == ["PERSON", "ID_NUM"]
    assert {item.label: item.leaked_chars for item in french.residual_by_label} == {
        "ID_NUM": 12,
        "PERSON": 4,
    }
    assert french.threshold_passed is False

    german = summaries["de"]
    assert german.leaked_chars == 0
    assert german.total_chars == 0
    assert [point.rate for point in german.trend] == [0.0, 0.0]


def test_companion_json_is_deterministic_phi_free_and_gate_ready() -> None:
    dashboard = LeakageDashboard.from_runs(
        _reports(),
        thresholds={"*": 0.5, "fr": 0.75},
    )

    first = dashboard.to_json()
    second = dashboard.to_json()
    payload = json.loads(first)

    assert first == second
    assert payload["artifact_type"] == LEAKAGE_DASHBOARD_ARTIFACT_TYPE
    assert payload["languages"] == sorted(SUPPORTED_LANGUAGES)
    assert payload["gate"]["passed"] is False
    assert payload["gate"]["violations"] == [
        {"language": "fr", "observed": 0.8, "threshold": 0.75}
    ]
    assert payload["by_language"]["fr"]["residual_counts_by_category"] == {
        "ID_NUM": 12,
        "PERSON": 4,
    }
    assert "Patient Alice" not in first
    assert "123-45-6789" not in first
    assert "Patient Bob" not in first
    assert "FR-ID-44" not in first
    assert {"text", "start", "end", "offsets", "fixture_ids"}.isdisjoint(
        _walk_keys(payload)
    )


def test_html_is_self_contained_sortable_phi_free_and_has_every_panel() -> None:
    dashboard = build_leakage_dashboard(_reports(), thresholds={"fr": 0.7})

    first = render_leakage_dashboard_html(dashboard)
    second = render_leakage_dashboard_html(dashboard)
    parser = _BalancedHTMLParser()
    parser.feed(first)
    parser.close()

    assert first == second
    assert first.startswith("<!doctype html>\n<html")
    assert first.rstrip().endswith("</html>")
    assert first.count('class="language-panel"') == len(SUPPORTED_LANGUAGES)
    assert 'id="language-en"' in first
    assert 'id="language-fr"' in first
    assert 'button type="button" class="sort" data-sort="rate"' in first
    assert "addEventListener" in first
    assert not re.search(r"\\b(?:src|href)=[\"']https?://", first)
    assert "Patient Alice" not in first
    assert "123-45-6789" not in first
    assert "Patient Bob" not in first
    assert "FR-ID-44" not in first
    assert parser.stack == []


def test_write_dashboard_emits_html_and_companion_json(tmp_path: Path) -> None:
    dashboard = build_leakage_dashboard(_reports())
    html_path = tmp_path / "reports" / "leakage.html"

    paths = write_leakage_dashboard(dashboard, html_path)

    assert paths.html == html_path
    assert paths.json == html_path.with_suffix(".json")
    assert paths.html.read_text(encoding="utf-8") == dashboard.to_html()
    assert json.loads(paths.json.read_text(encoding="utf-8")) == dashboard.to_dict()


def test_single_language_benchmark_leakage_metrics_are_supported() -> None:
    report = BenchmarkReport(
        suite="synthetic-en",
        model_name="safe-model",
        device="cpu",
        fixture_count=1,
        metrics={
            "leakage": {
                "leaked_chars_by_language": {"en": 2},
                "total_chars_by_language": {"en": 10},
                "leaked_chars_by_label": {"PERSON": 2},
                "total_chars_by_label": {"PERSON": 10},
            }
        },
    )

    dashboard = build_leakage_dashboard([report])
    english = next(item for item in dashboard.languages if item.language == "en")

    assert english.rate == pytest.approx(0.2)
    assert english.worst_labels[0].label == "PERSON"


def test_multi_language_metrics_require_heatmap_for_label_attribution() -> None:
    report = {
        "metrics": {
            "leakage": {
                "leaked_chars_by_language": {"en": 1, "fr": 1},
                "total_chars_by_language": {"en": 5, "fr": 5},
                "leaked_chars_by_label": {"PERSON": 2},
                "total_chars_by_label": {"PERSON": 10},
            }
        }
    }

    with pytest.raises(ValueError, match="require a leakage_heatmap"):
        build_leakage_dashboard([report])


def test_dashboard_rejects_noncanonical_labels_before_rendering() -> None:
    heatmap = {
        "cells": {"Alice 123-45-6789": {"en": {"leaked_chars": 1, "total_chars": 1}}}
    }

    with pytest.raises(ValueError, match="unsupported leakage label"):
        build_leakage_dashboard([heatmap])


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_walk_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_walk_keys(item))
        return keys
    return set()
