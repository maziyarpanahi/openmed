"""Offline tests for the differential-privacy budget summary surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any, get_args

import pytest

from openmed.risk.budget import CompositionRule
from openmed.training.dp_budget_summary import (
    COMPOSITION_METHODS,
    DP_BUDGET_SUMMARY_SCHEMA_VERSION,
    EPSILON_BUCKETS,
    EPSILON_OVERFLOW_BUCKET,
    MAX_DP_BUDGET_CONSUMPTIONS,
    MAX_DP_BUDGET_EPSILON,
    MIN_CELL_SIZE,
    OUTCOMES,
    SUPPRESSED_COUNT_BUCKET,
    DpBudgetConsumption,
    DpBudgetSummaryError,
    build_dp_budget_summary,
    build_dp_budget_summary_from_dicts,
)
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

_GOLDEN_MARKDOWN = """# Differential-privacy budget summary

- schema: openmed.training.dp_budget_summary.v1
- consumptions: 7
- cells: 2
- suppressed cells: 1
- minimum reported cell: 5

## `clinical-ner` / `advanced`

Suppressed: fewer than the minimum reported cell of 5 consumptions.

## `clinical-ner` / `basic`

| outcome | count |
| --- | --- |
| allowed | 5 |
| denied | suppressed (<min_cell_size) |
| exhausted | suppressed (<min_cell_size) |

| epsilon bucket | count |
| --- | --- |
| le_0.5 | 6 |
| le_1 | suppressed (<min_cell_size) |
| le_3 | suppressed (<min_cell_size) |
| le_8 | suppressed (<min_cell_size) |
| gt_8 | suppressed (<min_cell_size) |

| metric | value |
| --- | --- |
| epsilon total | 3 |
| epsilon max | 0.5 |
| delta total | 0.021 |
| delta max | 0.006 |"""

_GOLDEN_JSON_SHA256 = "e76c78f2956858381933e933f9c1c13675dd3d9ea0b03766220479578e90f1f5"
_GOLDEN_MARKDOWN_SHA256 = (
    "65e2980ad09605a2ff4a8a26da0d69cd996653e8f1d7db7320dba4013f00c3c9"
)

_CASE_IDS = [case.reason_code for case in FORBIDDEN_FIELD_CASES]


def _record(
    round_index: int,
    *,
    model_family: str = "clinical-ner",
    epsilon: float = 0.5,
    delta: float = 0.0,
    composition: str = "basic",
    outcome: str = "allowed",
) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "model_family": model_family,
        "epsilon": epsilon,
        "delta": delta,
        "composition": composition,
        "outcome": outcome,
    }


def _golden_records() -> list[dict[str, Any]]:
    records = [_record(index, delta=0.001 * index) for index in range(1, 7)]
    records[0]["outcome"] = "denied"
    records.append(
        _record(7, composition="advanced"),
    )
    return records


def _golden_summary():
    return build_dp_budget_summary_from_dicts(_golden_records())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_constants_form_a_closed_public_set() -> None:
    assert DP_BUDGET_SUMMARY_SCHEMA_VERSION == "openmed.training.dp_budget_summary.v1"
    assert MIN_CELL_SIZE == 5
    assert COMPOSITION_METHODS == ("basic", "advanced")
    assert OUTCOMES == ("allowed", "denied", "exhausted")
    assert [label for _, label in EPSILON_BUCKETS] == [
        "le_0.5",
        "le_1",
        "le_3",
        "le_8",
    ]
    assert EPSILON_OVERFLOW_BUCKET == "gt_8"
    assert SUPPRESSED_COUNT_BUCKET == "<min_cell_size"


def test_composition_methods_track_the_risk_literal() -> None:
    assert set(get_args(CompositionRule)) == set(COMPOSITION_METHODS)


def test_empty_input_reports_no_cells() -> None:
    summary = build_dp_budget_summary([])

    assert summary.cells == ()
    assert summary.consumption_count == 0
    assert summary.suppressed_cells == 0
    assert summary.to_dict() == {
        "cells": [],
        "consumption_count": 0,
        "min_cell_size": MIN_CELL_SIZE,
        "schema_version": DP_BUDGET_SUMMARY_SCHEMA_VERSION,
        "suppressed_cells": 0,
    }


def test_single_round_cell_is_suppressed() -> None:
    summary = build_dp_budget_summary([_record(1)])
    (cell,) = summary.cells

    assert cell.suppressed is True
    assert cell.consumption_count == 1
    assert cell.epsilon_total is None
    assert cell.epsilon_max is None
    assert cell.delta_total is None
    assert cell.delta_max is None
    assert cell.outcomes["allowed"] == {
        "count": None,
        "count_bucket": SUPPRESSED_COUNT_BUCKET,
        "suppressed": True,
    }
    assert cell.epsilon_buckets["le_0.5"]["suppressed"] is True


def test_reported_cell_counts_totals_and_buckets() -> None:
    summary = _golden_summary()
    reported = summary.cells[1]

    assert (reported.model_family, reported.composition) == ("clinical-ner", "basic")
    assert reported.suppressed is False
    assert reported.consumption_count == 6
    assert reported.outcomes["allowed"]["count"] == 5
    assert reported.outcomes["denied"]["suppressed"] is True
    assert reported.outcomes["exhausted"]["suppressed"] is True
    assert reported.epsilon_buckets["le_0.5"]["count"] == 6
    assert reported.epsilon_buckets["gt_8"]["suppressed"] is True
    assert reported.epsilon_total == 3.0
    assert reported.epsilon_max == 0.5
    assert reported.delta_total == 0.021
    assert reported.delta_max == 0.006


def test_golden_markdown_is_byte_stable() -> None:
    assert _golden_summary().render_markdown() == _GOLDEN_MARKDOWN


def test_golden_markdown_digest_is_pinned() -> None:
    assert _digest(_golden_summary().render_markdown()) == _GOLDEN_MARKDOWN_SHA256


def test_golden_json_digest_is_pinned() -> None:
    rendered = _golden_summary().render_json()

    assert rendered == json.dumps(
        _golden_summary().to_dict(), indent=2, sort_keys=True, ensure_ascii=False
    )
    assert _digest(rendered) == _GOLDEN_JSON_SHA256


def test_rendered_text_has_no_trailing_newline() -> None:
    summary = _golden_summary()

    assert not summary.render_json().endswith("\n")
    assert not summary.render_markdown().endswith("\n")


def test_record_order_never_changes_the_output() -> None:
    records = _golden_records()
    shuffled = list(reversed(records))
    rotated = records[3:] + records[:3]
    baseline = build_dp_budget_summary_from_dicts(records)

    for candidate in (shuffled, rotated):
        summary = build_dp_budget_summary_from_dicts(candidate)
        assert summary.render_json() == baseline.render_json()
        assert summary.render_markdown() == baseline.render_markdown()


def test_cells_are_sorted_by_labels() -> None:
    summary = build_dp_budget_summary_from_dicts(
        [
            _record(1, model_family="zeta-family"),
            _record(1, model_family="alpha-family", composition="advanced"),
            _record(2, model_family="alpha-family", composition="advanced"),
        ]
    )

    assert [(cell.model_family, cell.composition) for cell in summary.cells] == [
        ("alpha-family", "advanced"),
        ("zeta-family", "basic"),
    ]


@pytest.mark.parametrize("count", [1, 4])
def test_counts_below_the_floor_are_suppressed(count: int) -> None:
    summary = build_dp_budget_summary_from_dicts(
        [_record(index) for index in range(1, count + 1)]
    )

    assert summary.suppressed_cells == 1
    assert summary.cells[0].suppressed is True


def test_counts_at_the_floor_are_reported() -> None:
    summary = build_dp_budget_summary_from_dicts(
        [_record(index) for index in range(1, MIN_CELL_SIZE + 1)]
    )

    assert summary.suppressed_cells == 0
    assert summary.cells[0].suppressed is False
    assert summary.cells[0].epsilon_total == pytest.approx(2.5)


def test_configured_floor_changes_reported_and_suppressed_cells() -> None:
    records = [_record(index) for index in range(1, 3)]
    summary = build_dp_budget_summary_from_dicts(records, min_cell_size=2)

    assert summary.min_cell_size == 2
    assert summary.cells[0].suppressed is False
    assert summary.cells[0].consumption_count == 2
    assert summary.cells[0].epsilon_total == pytest.approx(1.0)


def test_epsilon_buckets_follow_inclusive_upper_bounds() -> None:
    epsilons = [0.5, 0.5, 0.5, 0.5, 0.75, 0.9, 3.0, 2.0, 8.0, 5.0, 8.5, 9.0]
    summary = build_dp_budget_summary(
        [
            DpBudgetConsumption(
                round_index=index,
                model_family="clinical-ner",
                epsilon=value,
                delta=0.0,
                composition="basic",
                outcome="allowed",
            )
            for index, value in enumerate(epsilons, start=1)
        ],
        min_cell_size=2,
    )
    buckets = summary.cells[0].epsilon_buckets

    assert buckets["le_0.5"]["count"] == 4
    assert buckets["le_1"]["count"] == 2
    assert buckets["le_3"]["count"] == 2
    assert buckets["le_8"]["count"] == 2
    assert buckets["gt_8"]["count"] == 2


def test_outcomes_are_counted_from_aggregate_labels_only() -> None:
    records = [_record(index) for index in range(1, 6)] + [
        _record(6, outcome="denied"),
        _record(7, outcome="exhausted"),
    ]
    summary = build_dp_budget_summary_from_dicts(records)
    outcomes = summary.cells[0].outcomes

    assert outcomes["allowed"]["count"] == 5
    assert outcomes["denied"]["suppressed"] is True
    assert outcomes["exhausted"]["suppressed"] is True


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
def test_forbidden_fields_are_rejected_without_echoing_values(case) -> None:
    payload = _record(1)
    payload[case.field] = case.value

    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary_from_dicts([payload])

    message = str(error.value)
    assert message == "invalid differential-privacy budget consumption fields"
    if case.marker is not None:
        assert case.marker not in message


def test_from_dict_rejects_missing_and_extra_fields() -> None:
    with pytest.raises(DpBudgetSummaryError) as missing:
        DpBudgetConsumption.from_dict(
            {key: value for key, value in _record(1).items() if key != "delta"}
        )
    assert str(missing.value) == "differential-privacy budget consumption is incomplete"

    with pytest.raises(DpBudgetSummaryError) as extra:
        DpBudgetConsumption.from_dict({**_record(1), "note": "placeholder"})
    assert str(extra.value) == "invalid differential-privacy budget consumption fields"

    with pytest.raises(DpBudgetSummaryError) as mapping:
        DpBudgetConsumption.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert str(mapping.value) == "consumption record must be a mapping"


def test_from_dict_round_trips_through_to_dict() -> None:
    payload = _record(
        3, epsilon=1.5, delta=0.25, composition="advanced", outcome="denied"
    )
    consumption = DpBudgetConsumption.from_dict(payload)

    assert consumption.to_dict() == payload
    assert DpBudgetConsumption.from_dict(consumption.to_dict()) == consumption


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("round_index", 0, "round_index must be a bounded positive integer"),
        (
            "round_index",
            1_000_001,
            "round_index must be a bounded positive integer",
        ),
        ("round_index", 1.0, "round_index must be a bounded positive integer"),
        ("round_index", True, "round_index must be a bounded positive integer"),
        ("model_family", "Site-01", "model_family must be a lowercase label"),
        ("model_family", "clinical ner", "model_family must be a lowercase label"),
        ("epsilon", True, "epsilon must be a finite number"),
        ("epsilon", "0.5", "epsilon must be a finite number"),
        ("epsilon", float("nan"), "epsilon must be a bounded finite number"),
        ("epsilon", -1.0, "epsilon must be a bounded finite number"),
        (
            "epsilon",
            MAX_DP_BUDGET_EPSILON + 1.0,
            "epsilon must be a bounded finite number",
        ),
        ("delta", 1.0, "delta must be less than 1"),
        ("delta", 1.5, "delta must be a bounded finite number"),
        ("delta", -0.5, "delta must be a bounded finite number"),
        ("composition", "renyi", "composition must be a supported method"),
        ("outcome", "failed", "outcome must be a supported outcome"),
    ],
)
def test_invalid_record_values_are_rejected(
    field: str, value: Any, message: str
) -> None:
    payload = _record(1)
    payload[field] = value

    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary_from_dicts([payload])

    assert str(error.value) == message


def test_duplicate_rounds_within_a_model_family_are_rejected() -> None:
    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary_from_dicts([_record(1), _record(1)])

    assert str(error.value) == "duplicate round index for a model family"


def test_same_round_across_model_families_is_allowed() -> None:
    summary = build_dp_budget_summary_from_dicts(
        [_record(1), _record(1, model_family="other-family")]
    )

    assert len(summary.cells) == 2


@pytest.mark.parametrize("payload", ["text", b"bytes", iter([]), 7, None])
def test_non_sequence_input_is_rejected(payload: Any) -> None:
    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary(payload)

    assert str(error.value) == "consumptions must be a sequence of records"


def test_record_limit_is_enforced() -> None:
    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary([{}] * (MAX_DP_BUDGET_CONSUMPTIONS + 1))

    assert str(error.value) == "too many consumption records to summarize"


@pytest.mark.parametrize("value", [1, 0, 1_001, "5", 5.0, True])
def test_min_cell_size_is_bounded(value: Any) -> None:
    with pytest.raises(DpBudgetSummaryError) as error:
        build_dp_budget_summary([], min_cell_size=value)

    assert str(error.value) == "min_cell_size must be a bounded integer >= 2"


def test_summary_payload_key_sets_are_stable() -> None:
    summary = _golden_summary()
    payload = summary.to_dict()
    cell = payload["cells"][1]

    assert sorted(payload) == [
        "cells",
        "consumption_count",
        "min_cell_size",
        "schema_version",
        "suppressed_cells",
    ]
    assert sorted(cell) == [
        "composition",
        "consumption_count",
        "delta_max",
        "delta_total",
        "epsilon_buckets",
        "epsilon_max",
        "epsilon_total",
        "model_family",
        "outcomes",
        "suppressed",
    ]
    assert sorted(cell["outcomes"]) == sorted(OUTCOMES)
    assert sorted(cell["epsilon_buckets"]) == [
        "gt_8",
        "le_0.5",
        "le_1",
        "le_3",
        "le_8",
    ]


def test_suppressed_cell_hides_numeric_totals() -> None:
    summary = build_dp_budget_summary_from_dicts([_record(1)])
    payload = summary.cells[0].to_dict()

    assert payload["suppressed"] is True
    assert payload["epsilon_total"] is None
    assert payload["epsilon_max"] is None
    assert payload["delta_total"] is None
    assert payload["delta_max"] is None
    assert payload["epsilon_buckets"]["le_0.5"] == {
        "count": None,
        "count_bucket": SUPPRESSED_COUNT_BUCKET,
        "suppressed": True,
    }


def test_dataclass_input_is_accepted() -> None:
    consumption = DpBudgetConsumption(
        round_index=1,
        model_family="clinical-ner",
        epsilon=0.5,
        delta=0.0,
        composition="basic",
        outcome="allowed",
    )
    summary = build_dp_budget_summary([consumption])

    assert summary.consumption_count == 1
    assert summary.cells[0].suppressed is True
