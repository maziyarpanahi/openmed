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


def test_untrusted_dimension_labels_are_not_copied_into_artifacts() -> None:
    marker = "SyntheticCaseSecret"
    scorecard = build_crosslingual_scorecard(
        [
            {
                "language": marker,
                "family": marker,
                "fixture_count": 1,
                "metrics": {"character_recall": {"rate": 1.0}},
            }
        ]
    )

    assert marker not in render_crosslingual_scorecard_json(scorecard)
    assert marker not in scorecard.to_markdown()
    assert scorecard.languages[0].startswith("label_sha256_")
    assert scorecard.families[0].startswith("label_sha256_")


def test_metric_summary_keys_are_not_language_evidence():
    card = build_crosslingual_scorecard(
        [
            {
                "fixture_count": 2,
                "metrics": {
                    "character_recall": {"overall": 0.8, "by_language": {"en": 0.8}}
                },
            }
        ],
        expected_languages=("en",),
    )
    assert card.languages == ("en",)
    assert card.per_language["en"]["recall"] == 0.8


def test_zero_support_language_buckets_remain_missing():
    card = build_crosslingual_scorecard(
        [
            {
                "fixture_count": 2,
                "metrics": {
                    "character_recall": {
                        "by_language": {"en": 0.8, "fr": 1.0},
                        "total_chars_by_language": {"en": 10, "fr": 0},
                        "covered_chars_by_language": {"en": 8, "fr": 0},
                    }
                },
            }
        ],
        expected_languages=("en", "fr"),
    )
    assert card.languages == ("en",)
    assert card.missing_languages == ("fr",)


def test_critical_leakage_rate_is_not_misreported_as_a_count():
    card = build_crosslingual_scorecard(
        [{"language": "en", "metrics": {"critical_leakage_rate": 0.5}}]
    )
    assert card.per_language["en"]["critical_leakage_count"] is None


@pytest.mark.parametrize(
    "evidence",
    [
        {"numerator": -1, "denominator": 0, "rate": 0.8},
        {"numerator": 1, "denominator": -1, "rate": 0.8},
        {"numerator": 0, "denominator": 0, "rate": 1.0},
    ],
)
def test_invalid_rate_denominators_do_not_invent_support(evidence):
    card = build_crosslingual_scorecard(
        [{"language": "en", "metrics": {"recall": evidence}}]
    )
    assert card.per_language["en"]["recall"] is None


def test_huge_numeric_values_are_sanitized_without_overflow():
    card = build_crosslingual_scorecard(
        [{"language": "en", "metrics": {"recall": 10**10000}}]
    )
    assert card.per_language["en"]["recall"] is None


def test_report_callback_failure_has_no_sensitive_error_context():
    class Broken:
        def to_dict(self):
            raise RuntimeError("SyntheticPatientSecret")

    with pytest.raises(ValueError) as caught:
        build_crosslingual_scorecard([Broken()])
    assert "SyntheticPatientSecret" not in str(caught.value)
    assert caught.value.__context__ is None


def test_scorecard_input_collection_is_bounded():
    with pytest.raises(ValueError):
        build_crosslingual_scorecard(({"language": "en"} for _ in range(8193)))


def test_direct_scorecard_construction_cannot_bypass_aggregate_boundary():
    card = CrossLingualScorecard(
        report_count=1,
        fixture_count=1,
        languages=("en",),
        families=(),
        per_language={"en": {"recall": 0.5, "raw_text": "SyntheticPatientSecret"}},
        per_family={},
    )
    assert "SyntheticPatientSecret" not in card.to_json()


def test_serialized_scorecard_is_detached_from_nested_state():
    card = build_crosslingual_scorecard(
        [{"language": "en", "metrics": {"recall": 0.5}}]
    )
    payload = card.to_dict()
    payload["per_language"]["en"]["metrics"]["recall"] = "SyntheticPatientSecret"
    assert card.per_language["en"]["metrics"]["recall"] == 0.5


def test_empty_language_slice_is_not_complete_evidence():
    card = build_crosslingual_scorecard(
        [{"metrics": {"per_language": {"en": {}}}}], expected_languages=("en",)
    )
    assert card.missing_languages == ("en",)


def test_alias_metric_order_is_deterministic():
    first = {"language": "en", "metrics": {"recall": 0.9, "character_recall": 0.5}}
    second = {"language": "en", "metrics": {"character_recall": 0.5, "recall": 0.9}}
    assert (
        build_crosslingual_scorecard([first]).to_json()
        == build_crosslingual_scorecard([second]).to_json()
    )


def test_sibling_leakage_support_excludes_global_empty_buckets():
    card = build_crosslingual_scorecard(
        [
            {
                "fixture_count": 1,
                "metrics": {
                    "character_recall": {"by_language": {"en": 0.8, "fr": 1.0}},
                    "leakage": {"total_chars_by_language": {"en": 10, "fr": 0}},
                },
            }
        ],
        expected_languages=("en", "fr"),
    )
    assert card.languages == ("en",)
    assert card.missing_languages == ("fr",)


def test_unknown_fixture_counts_are_not_character_support():
    card = build_crosslingual_scorecard(
        [
            {
                "metrics": {
                    "per_language": {
                        "en": {"recall": {"numerator": 9, "denominator": 10}},
                        "fr": {"recall": {"numerator": 15, "denominator": 20}},
                    }
                }
            }
        ]
    )
    assert card.fixture_count == 0
    assert card.per_language["en"]["counts"]["fixtures"] == 0
    assert card.per_language["fr"]["counts"]["fixtures"] == 0


def test_nested_scorecard_state_is_immutable():
    card = build_crosslingual_scorecard(
        [{"language": "en", "metrics": {"recall": 0.5}}]
    )
    with pytest.raises(TypeError):
        card.per_language["en"]["metrics"]["recall"] = "SyntheticPatientSecret"


def test_scorecard_recursive_metadata_is_rejected_without_context():
    value = {}
    value["metadata"] = value
    with pytest.raises(ValueError) as caught:
        build_crosslingual_scorecard([value])
    assert caught.value.__context__ is None


def test_oversized_latency_aggregation_is_rejected_without_sensitive_context():
    with pytest.raises(ValueError) as caught:
        build_crosslingual_scorecard(
            [{"language": "en", "fixture_count": 3, "metrics": {"latency_ms": 1e308}}]
        )
    assert caught.value.__context__ is None
