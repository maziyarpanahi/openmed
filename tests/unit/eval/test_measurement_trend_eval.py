"""Offline eval gate for serial-measurement grouping and trend derivation.

Scores the deterministic extractor against the committed synthetic gold set and
enforces the acceptance thresholds: trend-direction classification accuracy
>= 0.90 and exact grouping/comparability reproduction. Runs fully offline with
no models or network.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import TREND_ADVISORY, extract_measurement_trends
from openmed.eval.metrics import trend_direction_accuracy, trend_grouping_accuracy

GOLD = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "measurement_trend.jsonl"
)

DIRECTION_FLOOR = 0.90
GROUPING_FLOOR = 0.90


def _load_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _predict(rows: list[dict]) -> list[list[dict]]:
    return [
        extract_measurement_trends(
            row["points"], reference_date=row.get("reference_date")
        )
        for row in rows
    ]


def test_gold_set_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= 10
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_direction_and_grouping_accuracy_meet_floor():
    rows = _load_gold()
    predicted = _predict(rows)
    gold = [row["gold"]["trends"] for row in rows]

    direction = trend_direction_accuracy(predicted, gold)
    grouping = trend_grouping_accuracy(predicted, gold)

    assert direction.rate >= DIRECTION_FLOOR, (
        f"trend-direction accuracy {direction.rate:.3f} < {DIRECTION_FLOOR}"
    )
    assert grouping.rate >= GROUPING_FLOOR, (
        f"trend-grouping accuracy {grouping.rate:.3f} < {GROUPING_FLOOR}"
    )


def test_gold_covers_every_direction_including_unknown():
    rows = _load_gold()
    directions = {trend["direction"] for row in rows for trend in row["gold"]["trends"]}
    assert directions == {"increasing", "decreasing", "stable", "mixed", "unknown"}


def test_gold_covers_unit_conversion_and_incomparable_flagging():
    rows = _load_gold()
    predicted = _predict(rows)

    # At least one series mixes commensurable units (cm/mm) and still compares.
    assert any(
        trend["canonical_unit"] == "m" and trend["comparable_count"] >= 2
        for trends in predicted
        for trend in trends
    )
    # At least one point is flagged incomparable rather than silently dropped.
    assert any(trend["incomparable_points"] for trends in predicted for trend in trends)


def test_series_lacking_orderable_timepoints_are_labeled_unknown():
    # Directly enforces the acceptance criterion "series lacking timepoints ->
    # unknown", independent of the aggregate accuracy floor: every gold trend
    # marked unknown for a timepoint reason must be produced as unknown and
    # explicitly not ordered.
    rows = _load_gold()
    predicted = _predict(rows)
    checked = 0
    for row, trends in zip(rows, predicted):
        by_entity = {trend["entity"].casefold(): trend for trend in trends}
        for gold in row["gold"]["trends"]:
            if gold.get("unknown_reason") == "timepoints":
                produced = by_entity[gold["entity"].casefold()]
                assert produced["direction"] == "unknown", gold["entity"]
                assert produced["ordered"] is False, gold["entity"]
                checked += 1
    # No-timepoint, partial-timepoint, and non-temporal-string cases are covered.
    assert checked >= 3


def test_metrics_penalize_wrong_direction_and_missing_or_misgrouped_trends():
    # The eval gate must be a real gate: prove the metrics score down when a
    # predicted trend has the wrong direction, is missing, or is grouped with
    # the wrong point partition -- not just when everything matches.
    gold = [
        [
            {
                "entity": "hemoglobin",
                "direction": "increasing",
                "comparable_count": 3,
                "incomparable_count": 0,
            }
        ]
    ]

    correct = [
        [
            {
                "entity": "hemoglobin",
                "direction": "increasing",
                "comparable_count": 3,
                "incomparable_points": [],
            }
        ]
    ]
    assert trend_direction_accuracy(correct, gold).rate == 1.0
    assert trend_grouping_accuracy(correct, gold).rate == 1.0

    wrong_direction = [
        [
            {
                "entity": "hemoglobin",
                "direction": "decreasing",
                "comparable_count": 3,
                "incomparable_points": [],
            }
        ]
    ]
    assert trend_direction_accuracy(wrong_direction, gold).rate == 0.0

    missing_group = [[]]
    assert trend_direction_accuracy(missing_group, gold).rate == 0.0
    assert trend_grouping_accuracy(missing_group, gold).rate == 0.0

    wrong_partition = [
        [
            {
                "entity": "hemoglobin",
                "direction": "increasing",
                "comparable_count": 2,
                "incomparable_points": [{"unit": "mmol/L"}],
            }
        ]
    ]
    assert trend_grouping_accuracy(wrong_partition, gold).rate == 0.0


def test_advisory_emitted_on_every_predicted_trend():
    rows = _load_gold()
    predicted = _predict(rows)
    trends = [trend for trends in predicted for trend in trends]
    assert trends
    assert all(trend["advisory"] == TREND_ADVISORY for trend in trends)
    assert "not a clinical judgment" in TREND_ADVISORY
