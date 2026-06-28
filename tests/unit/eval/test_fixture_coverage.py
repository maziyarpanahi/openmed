"""Tests for golden fixture coverage reporting."""

from __future__ import annotations

import json

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval import FixtureCoverageReport, fixture_coverage_report
from openmed.eval.golden import GOLDEN_CATEGORIES, load_golden_fixtures


def test_fixture_coverage_report_lists_committed_fixture_gaps() -> None:
    report = fixture_coverage_report()

    assert isinstance(report, FixtureCoverageReport)
    assert report.fixture_count == len(load_golden_fixtures())
    assert set(report.covered_labels) | set(report.missing_labels) == CANONICAL_LABELS
    assert not (set(report.covered_labels) & set(report.missing_labels))
    assert set(report.covered_languages) | set(report.missing_languages) == (
        SUPPORTED_LANGUAGES
    )
    assert not report.missing_languages
    assert set(report.covered_categories) == set(GOLDEN_CATEGORIES)
    assert not report.missing_categories
    assert report.category_counts == {
        "nested_overlapping": 1,
        "chunk_boundary": 1,
        "multilingual": len(SUPPORTED_LANGUAGES),
        "checksum_ids": 1,
        "date_arithmetic": 1,
    }

    assert "PERSON" in report.covered_labels
    assert "ID_NUM" in report.covered_labels
    assert "ACCOUNT_NUMBER" in report.missing_labels


def test_fixture_coverage_report_flags_absent_categories() -> None:
    fixtures = [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.category != "date_arithmetic"
    ]

    report = fixture_coverage_report(fixtures=fixtures)

    assert "date_arithmetic" in report.missing_categories
    assert "date_arithmetic" not in report.covered_categories
    assert report.category_counts["date_arithmetic"] == 0


def test_fixture_coverage_to_dict_is_stable_and_aggregate_only() -> None:
    report = fixture_coverage_report()

    payload = report.to_dict()
    assert payload == report.to_dict()
    assert payload["category_counts"]["multilingual"] == len(SUPPORTED_LANGUAGES)
    assert payload["labels"]["covered"] == list(report.covered_labels)
    assert payload["languages"]["missing"] == []
    assert payload["categories"]["missing"] == []

    serialized = json.dumps(payload, sort_keys=True)
    assert "golden-" not in serialized
    assert "Synthetic" not in serialized


def test_fixture_coverage_markdown_is_byte_stable_and_aggregate_only() -> None:
    report = fixture_coverage_report()

    markdown = report.to_markdown()

    assert markdown.encode("utf-8") == report.to_markdown().encode("utf-8")
    assert "| `PERSON` | covered |" in markdown
    assert "| `ACCOUNT_NUMBER` | missing |" in markdown
    assert "| `en` | covered |" in markdown
    assert "| `nested_overlapping` | 1 | covered |" in markdown
    assert f"| `multilingual` | {len(SUPPORTED_LANGUAGES)} | covered |" in markdown
    assert "golden-" not in markdown
    assert "Synthetic" not in markdown
