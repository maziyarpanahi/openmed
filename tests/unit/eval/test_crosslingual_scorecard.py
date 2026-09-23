"""Focused tests for the aggregate-only cross-lingual scorecard."""

from __future__ import annotations

import json

import pytest

from openmed.eval.crosslingual_scorecard import (
    CrossLingualScorecard,
    build_crosslingual_scorecard,
    render_crosslingual_scorecard_json,
    render_crosslingual_scorecard_markdown,
)
from openmed.eval.report import BenchmarkReport


def _report(
    *,
    language: str | None,
    family: str,
    fixture_count: int,
    covered: int,
    total: int,
    critical: int,
    abstained: int,
    p50_ms: float,
    p95_ms: float,
) -> BenchmarkReport:
    metadata: dict[str, object] = {
        "family": family,
        "raw_fixture": "Synthetic Patient Alice 123-45-6789",
    }
    if language is not None:
        metadata["language"] = language
    return BenchmarkReport(
        suite="synthetic-crosslingual",
        model_name="synthetic-model",
        device="cpu",
        fixture_count=fixture_count,
        metadata=metadata,
        metrics={
            "abstention": {"abstained": abstained, "total": total},
            "character_recall": {
                "denominator": total,
                "numerator": covered,
            },
            "leakage": {
                "critical_leakage_count": critical,
            },
            "latency": {"p50_ms": p50_ms, "p95_ms": p95_ms},
            "unsafe_examples": ["Synthetic Patient Alice", "123-45-6789"],
        },
    )


def test_scorecard_aggregates_language_and_family_metrics() -> None:
    reports = [
        _report(
            language="fr",
            family="encoder",
            fixture_count=10,
            covered=9,
            total=10,
            critical=0,
            abstained=1,
            p50_ms=10.0,
            p95_ms=18.0,
        ),
        _report(
            language="en",
            family="encoder",
            fixture_count=5,
            covered=4,
            total=5,
            critical=1,
            abstained=0,
            p50_ms=8.0,
            p95_ms=15.0,
        ),
    ]

    scorecard = build_crosslingual_scorecard(
        reports,
        expected_languages=("en", "fr", "de"),
    )

    assert isinstance(scorecard, CrossLingualScorecard)
    assert scorecard.languages == ("en", "fr")
    assert scorecard.missing_languages == ("de",)
    assert scorecard.per_language["en"]["recall"] == 0.8
    assert scorecard.per_language["fr"]["critical_leakage"] == 0
    assert scorecard.per_language["fr"]["abstention"] == 0.1
    assert scorecard.per_language["fr"]["latency_p50_ms"] == 10.0
    assert scorecard.per_family["encoder"]["recall"] == pytest.approx(13 / 15)
    assert scorecard.per_family["encoder"]["critical_leakage_count"] == 1
    assert scorecard.per_family["encoder"]["counts"]["reports"] == 2


def test_scorecard_reads_language_slices_and_flags_unlabeled_reports() -> None:
    reports = [
        {
            "family": "decoder",
            "fixture_count": 6,
            "metrics": {
                "per_language": {
                    "de": {
                        "fixture_count": 3,
                        "character_recall": {"rate": 0.75},
                        "leakage": {"critical_leakage_count": 0},
                    },
                    "es": {
                        "fixture_count": 3,
                        "character_recall": {"rate": 0.5},
                        "leakage": {"critical_leakage_count": 2},
                    },
                }
            },
        },
        {
            "family": "decoder",
            "fixture_count": 2,
            "metrics": {"character_recall": {"rate": 1.0}},
        },
    ]

    scorecard = build_crosslingual_scorecard(
        reports,
        required_languages=("de", "es", "it"),
    )

    assert scorecard.languages == ("de", "es")
    assert scorecard.missing_languages == ("it",)
    assert scorecard.unlabeled_report_count == 1
    assert scorecard.unlabeled_fixture_count == 2
    assert scorecard.per_language["de"]["recall"] == 0.75
    assert scorecard.per_language["es"]["critical_leakage"] == 2
    assert scorecard.per_family["decoder"]["recall"] == pytest.approx(23 / 32)


def test_renderers_are_deterministic_and_aggregate_only() -> None:
    reports = [
        _report(
            language="fr",
            family="encoder",
            fixture_count=1,
            covered=1,
            total=1,
            critical=0,
            abstained=0,
            p50_ms=4.0,
            p95_ms=5.0,
        ),
        _report(
            language="en",
            family="encoder",
            fixture_count=1,
            covered=1,
            total=1,
            critical=0,
            abstained=0,
            p50_ms=3.0,
            p95_ms=4.0,
        ),
    ]
    scorecard = CrossLingualScorecard.from_reports(reversed(reports))

    first_json = render_crosslingual_scorecard_json(scorecard)
    assert first_json == render_crosslingual_scorecard_json(scorecard)
    payload = json.loads(first_json)
    assert payload["aggregate_only"] is True
    assert payload["per_language"]["en"]["recall"] == 1.0
    assert "Patient Alice" not in first_json
    assert "123-45-6789" not in first_json
    assert "unsafe_examples" not in first_json

    markdown = render_crosslingual_scorecard_markdown(scorecard)
    assert markdown == scorecard.to_markdown()
    assert "Language evidence" in markdown
    assert "Family evidence" in markdown
    assert "Patient Alice" not in markdown
    assert "123-45-6789" not in markdown
