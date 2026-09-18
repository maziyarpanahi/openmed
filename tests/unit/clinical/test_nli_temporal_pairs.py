"""Offline regression tests for temporality-aware clinical NLI pairs."""

from __future__ import annotations

import json
import traceback
from datetime import date

import pytest

from openmed.clinical import (
    HISTORICAL,
    RECENT,
    TemporalCompatibility,
    TemporalInterval,
    TemporalMetadata,
    build_temporal_nli_pair,
    build_temporal_nli_pairs,
    classify_temporal_compatibility,
    classify_temporal_intervals,
    compare_temporal_intervals,
    compare_temporal_metadata,
    normalize_temporal,
)
from openmed.clinical.timeline import NormalizedInterval


def _metadata(
    value: str | None,
    status: str,
    *,
    lower_bound: str | None = None,
    upper_bound: str | None = None,
) -> dict[str, object]:
    interval: dict[str, object] = {"value": value, "resolved": value is not None}
    if lower_bound is not None:
        interval["lower_bound"] = lower_bound
    if upper_bound is not None:
        interval["upper_bound"] = upper_bound
    return {"interval": interval, "temporal_status": status}


def test_interval_normalization_supports_dates_ranges_and_relative_values() -> None:
    assert TemporalInterval.from_value("2024-02").to_dict() == {
        "start": "2024-02-01",
        "end": "2024-02-29",
        "value": "2024-02-01/2024-02-29",
        "lower_bound": "2024-02-01",
        "upper_bound": "2024-02-29",
        "precision": "month",
        "resolved": True,
    }
    relative = TemporalInterval.from_value("3 days ago", reference_time="2026-06-15")
    assert relative.value == "2026-06-12"
    assert TemporalInterval.from_value("3 days ago").is_resolved is False


def test_approximate_relative_intervals_remain_review_required() -> None:
    interval = TemporalInterval.from_value(
        "about 3 days ago",
        reference_time="2026-06-15",
    )

    assert interval.value == "2026-06-12"
    assert interval.is_resolved is False


def test_interval_mapping_can_supply_its_own_explicit_reference_date() -> None:
    interval = TemporalInterval.from_value(
        {"value": "3 days ago", "reference_time": "2026-06-15"}
    )

    assert interval.value == "2026-06-12"


def test_equivalent_reference_time_aliases_are_accepted() -> None:
    interval = TemporalInterval.from_value(
        "3 days ago",
        reference_time="2026-06-15",
        reference_date="2026-06-15",
    )

    assert interval.value == "2026-06-12"


def test_existing_timeline_interval_is_preserved_with_uncertainty_bounds() -> None:
    timeline_interval = NormalizedInterval(
        start=date(2024, 1, 10),
        end=date(2024, 1, 12),
        lower_bound=date(2024, 1, 8),
        upper_bound=date(2024, 1, 14),
        uncertainty_days=2,
    )

    interval = TemporalInterval.from_value(timeline_interval)

    assert interval.value == "2024-01-10/2024-01-12"
    assert interval.iso_value == interval.value
    assert interval.possible_start == date(2024, 1, 8)
    assert interval.possible_end == date(2024, 1, 14)
    assert interval.uncertainty_days == 2


def test_date_endpoint_mapping_is_not_treated_as_text_offsets() -> None:
    pair = build_temporal_nli_pair(
        {
            "text": "synthetic dated finding",
            "start": "2024-01-01",
            "end": "2024-01-03",
            "temporality": HISTORICAL,
        },
        {
            "text": "synthetic dated claim",
            "interval": "2024-01-02",
            "temporality": HISTORICAL,
        },
    )

    assert pair.premise_interval.value == "2024-01-01/2024-01-03"
    assert pair.premise_offset is None
    assert pair.temporal_compatibility == "compatible"


def test_interval_only_comparison_reports_overlap_without_status_assumptions() -> None:
    assert compare_temporal_intervals("2024-01-01", "2024-01-01") == "compatible"
    assert classify_temporal_intervals("2024-01-01", "2025-01-01") == "incompatible"
    assert compare_temporal_intervals("3 days ago", "3 days ago") == "unresolved"


def test_normalized_timex_source_offsets_are_not_mistaken_for_dates() -> None:
    timex = normalize_temporal(
        "synthetic 3 days ago",
        [(10, 20)],
        reference_time="2026-06-15",
    )[0]

    interval = TemporalInterval.from_value(timex)

    assert interval.value == "2026-06-12"
    assert interval.is_resolved is True


def test_disjoint_historical_and_recent_events_gate_entailment() -> None:
    pair = build_temporal_nli_pair(
        {
            "text": "synthetic historical finding",
            "start": 0,
            "end": 28,
            **_metadata("2024-01-01", HISTORICAL),
        },
        {
            "text": "synthetic current finding",
            "start": 29,
            "end": 54,
            **_metadata("2026-01-01", RECENT),
        },
        predicted_label="entailment",
    )

    assert pair.premise_temporal_status == HISTORICAL
    assert pair.hypothesis_temporal_status == RECENT
    assert pair.temporal_compatibility == "incompatible"
    assert pair.temporal_comparison.reason == "status_mismatch"
    assert pair.review_required is True
    assert pair.label == "review_required"
    assert pair.predicted_label == "entailment"


def test_same_status_overlapping_intervals_retain_entailment() -> None:
    pair = build_temporal_nli_pair(
        "synthetic historical event",
        "synthetic historical event",
        premise_interval="2024-01-01/2024-01-03",
        hypothesis_interval="2024-01-03/2024-01-04",
        premise_temporality=HISTORICAL,
        hypothesis_temporality=HISTORICAL,
        predicted_label="entailment",
    )

    assert pair.temporal_compatibility == "compatible"
    assert pair.review_required is False
    assert pair.label == "entailment"


@pytest.mark.parametrize(
    ("premise", "hypothesis", "expected", "reason"),
    [
        (
            _metadata("2024-01-01/2024-01-02", HISTORICAL),
            _metadata("2024-01-02", HISTORICAL),
            TemporalCompatibility.COMPATIBLE,
            "interval_overlap",
        ),
        (
            _metadata(None, HISTORICAL),
            _metadata("2024-01-02", HISTORICAL),
            TemporalCompatibility.UNRESOLVED,
            "interval_unresolved",
        ),
        (
            _metadata("2024-01-01", HISTORICAL),
            _metadata(
                "2024-01-02",
                HISTORICAL,
                lower_bound="2023-12-31",
                upper_bound="2024-01-03",
            ),
            TemporalCompatibility.UNRESOLVED,
            "interval_uncertainty_overlap",
        ),
        (
            _metadata("2024-01-01", HISTORICAL),
            _metadata("2025-01-01", HISTORICAL),
            TemporalCompatibility.INCOMPATIBLE,
            "intervals_disjoint",
        ),
    ],
)
def test_temporal_compatibility_is_conservative(
    premise: dict[str, object],
    hypothesis: dict[str, object],
    expected: TemporalCompatibility,
    reason: str,
) -> None:
    comparison = compare_temporal_metadata(premise, hypothesis)

    assert comparison.status is expected
    assert comparison.reason == reason
    assert classify_temporal_compatibility(premise, hypothesis) == expected.value


def test_unresolved_and_hypothetical_sides_require_review() -> None:
    unresolved = build_temporal_nli_pair(
        "synthetic finding",
        "synthetic finding",
        premise_temporality=HISTORICAL,
        hypothesis_temporality=HISTORICAL,
        predicted_label="entailment",
    )
    hypothetical = build_temporal_nli_pair(
        "synthetic conditional finding",
        "synthetic current finding",
        premise_interval="2024-01-01",
        hypothesis_interval="2024-01-01",
        premise_temporality="hypothetical",
        hypothesis_temporality=RECENT,
        predicted_label="entailment",
    )

    assert unresolved.temporal_compatibility == "unresolved"
    assert unresolved.label == "review_required"
    assert hypothetical.temporal_compatibility == "incompatible"
    assert hypothetical.label == "review_required"


def test_unanchored_relative_intervals_are_review_required() -> None:
    pair = build_temporal_nli_pair(
        "synthetic prior event",
        "synthetic prior event",
        premise_time="3 days ago",
        hypothesis_time="3 days ago",
        premise_temporality=HISTORICAL,
        hypothesis_temporality=HISTORICAL,
        predicted_label="entailment",
    )

    assert pair.premise_interval.is_resolved is False
    assert pair.hypothesis_interval.is_resolved is False
    assert pair.temporal_comparison.reason == "interval_unresolved"
    assert pair.label == "review_required"


def test_mapping_status_aliases_and_context_cues_are_normalized() -> None:
    pair = build_temporal_nli_pair(
        {"text": "synthetic history of pain", "interval": "2024-01-01"},
        {"text": "synthetic current pain", "interval": "2024-01-01"},
        predicted_label="entailment",
    )

    assert pair.premise_temporality == HISTORICAL
    assert pair.hypothesis_temporality == RECENT
    assert pair.temporal_compatibility == "incompatible"


def test_nested_context_temporality_is_attached_to_the_pair_side() -> None:
    pair = build_temporal_nli_pair(
        {
            "text": "synthetic history of pain",
            "context": {"temporality": HISTORICAL},
            "interval": "2024-01-01",
        },
        {
            "text": "synthetic current pain",
            "context": {"temporality": RECENT},
            "interval": "2024-01-01",
        },
    )

    assert pair.premise_temporal_status == HISTORICAL
    assert pair.hypothesis_temporal_status == RECENT
    assert pair.temporal_compatibility == "incompatible"


def test_explicit_unknown_status_is_not_replaced_by_interval_inference() -> None:
    metadata = TemporalMetadata.from_value(
        {
            "clinical_assertion": {"temporality": "unknown"},
            "interval": "2024-01-01",
        },
        reference_time="2026-01-01",
    )

    assert metadata.temporal_status == "unknown"
    assert metadata.interval.value == "2024-01-01"


def test_pair_serialization_is_deterministic_and_source_value_free() -> None:
    premise = "synthetic source value that must not be serialized"
    hypothesis = "synthetic target value that must not be serialized"
    pair = build_temporal_nli_pair(
        premise,
        hypothesis,
        premise_interval="2024-01-01",
        hypothesis_interval="2024-01-01",
        premise_temporality=HISTORICAL,
        hypothesis_temporality=HISTORICAL,
        premise_offset=(4, 12),
        hypothesis_offset=(20, 28),
        predicted_label="entailment",
    )

    first = pair.to_json()
    payload = json.loads(first)

    assert first == pair.to_json()
    assert payload["schema_version"] == 1
    assert payload["pair_id"] == pair.pair_id
    assert premise not in first
    assert hypothesis not in first
    assert pair.to_audit_dict() == pair.to_dict()
    assert "human review" in payload["advisory"]
    assert "interval" in payload["premise"]["temporal"]


def test_raw_claim_dates_are_not_reused_as_serialized_temporal_evidence() -> None:
    pair = build_temporal_nli_pair("2024-01-01", "2024-01-01")

    assert pair.premise_interval.is_resolved is False
    assert pair.hypothesis_interval.is_resolved is False
    assert "2024-01-01" not in pair.to_json()


def test_model_input_is_the_explicit_raw_text_boundary() -> None:
    pair = build_temporal_nli_pair(
        "synthetic premise",
        "synthetic hypothesis",
        premise_interval="2024-01-01",
        hypothesis_interval="2024-01-01",
        premise_temporality=RECENT,
        hypothesis_temporality=RECENT,
    )

    assert pair.to_text_pair() == ("synthetic premise", "synthetic hypothesis")
    assert pair.to_model_input()["premise"] == "synthetic premise"
    assert pair.to_model_input()["hypothesis_temporal"]["temporal_status"] == RECENT
    assert "synthetic premise" not in repr(pair)


def test_nli_label_aliases_are_canonicalized() -> None:
    pair = build_temporal_nli_pair(
        "synthetic premise",
        "synthetic hypothesis",
        premise_interval="2024-01-01",
        hypothesis_interval="2024-01-01",
        premise_temporality=RECENT,
        hypothesis_temporality=RECENT,
        predicted_label="ENTAIL",
    )

    assert pair.predicted_label == "entailment"
    assert pair.label == "entailment"


def test_batch_construction_preserves_order_and_is_repeatable() -> None:
    inputs = [
        {
            "premise": {
                "text": "synthetic first",
                "interval": "2024-01-01",
                "temporality": HISTORICAL,
            },
            "hypothesis": {
                "text": "synthetic first claim",
                "interval": "2024-01-01",
                "temporality": HISTORICAL,
            },
            "predicted_label": "entailment",
        },
        (
            {
                "text": "synthetic second",
                "interval": "2024-01-01",
                "temporality": RECENT,
            },
            {
                "text": "synthetic second claim",
                "interval": "2024-01-02",
                "temporality": RECENT,
            },
        ),
    ]

    pairs = build_temporal_nli_pairs(inputs)

    assert len(pairs) == 2
    assert pairs[0].label == "entailment"
    assert pairs[1].temporal_compatibility == "incompatible"
    assert pairs == build_temporal_nli_pairs(inputs)


def test_invalid_metadata_does_not_echo_sensitive_values() -> None:
    sensitive_value = "synthetic-private-value"

    with pytest.raises(ValueError) as exc_info:
        build_temporal_nli_pair(
            "synthetic premise",
            "synthetic hypothesis",
            premise_interval={"start": sensitive_value, "end": "2024-01-01"},
            premise_temporality=RECENT,
            hypothesis_interval="2024-01-01",
            hypothesis_temporality=RECENT,
        )

    assert sensitive_value not in str(exc_info.value)


def test_conflicting_redundant_temporal_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_temporal_nli_pair(
            {"text": "synthetic premise", "interval": "2024-01-01"},
            {"text": "synthetic hypothesis", "interval": "2024-01-01"},
            premise_interval="2025-01-01",
            premise_temporality=RECENT,
            hypothesis_temporality=RECENT,
        )


def test_conflicting_interval_aliases_fail_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        TemporalMetadata.from_value(
            {
                "interval": "2024-01-01",
                "normalized_time": "2025-01-01",
                "temporality": HISTORICAL,
            }
        )


def test_conflicting_status_aliases_fail_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        TemporalMetadata.from_value(
            {
                "interval": "2024-01-01",
                "temporality": HISTORICAL,
                "status": RECENT,
            }
        )


def test_temporal_metadata_accepts_clinical_assertion_objects() -> None:
    metadata = TemporalMetadata.from_value(
        {"clinical_assertion": {"temporality": HISTORICAL}},
    )

    assert metadata.temporal_status == HISTORICAL
    assert metadata.resolved is False


@pytest.mark.parametrize("score", [10**400, -(10**400)])
def test_oversized_scores_have_controlled_error(score: int) -> None:
    with pytest.raises(ValueError, match="NLI score is invalid"):
        build_temporal_nli_pair(
            "synthetic premise", "synthetic hypothesis", predicted_score=score
        )


def test_final_supported_calendar_month_normalizes() -> None:
    interval = TemporalInterval.from_value("9999-12")
    assert interval.start == date(9999, 12, 1)
    assert interval.end == date(9999, 12, 31)


def test_arbitrary_precision_is_not_retained_in_reports() -> None:
    sentinel = "SYNTHETIC_PRIVATE_PRECISION"
    with pytest.raises(ValueError, match="interval precision is invalid") as caught:
        TemporalInterval("2024-01-01", "2024-01-01", precision=sentinel)
    assert sentinel not in str(caught.value)


def test_single_uncertain_flag_remains_review_required() -> None:
    interval = TemporalInterval.from_value(
        {"value": "2024-01-01", "granularity_flags": "uncertain"}
    )
    assert not interval.is_resolved


@pytest.mark.parametrize("field", ["value", "granularity_flags"])
def test_temporal_conversion_callback_errors_hide_values(field: str) -> None:
    sentinel = "SYNTHETIC_PRIVATE_TIMEX"

    class InvalidValue:
        def __str__(self) -> str:
            raise ValueError(sentinel)

    record = {"value": "2024-01-01"}
    record[field] = [InvalidValue()] if field == "granularity_flags" else InvalidValue()
    with pytest.raises(ValueError, match="is invalid") as caught:
        TemporalInterval.from_value(record)
    assert sentinel not in "".join(traceback.format_exception(caught.value))
