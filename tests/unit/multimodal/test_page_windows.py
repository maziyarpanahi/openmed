"""Synthetic unit tests for deterministic document page batching."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.page_windows import (
    MAX_DOCUMENT_PAGES,
    MAX_PAGE_BATCH_COUNT,
    MAX_PAGE_PIXELS,
    PAGE_BATCH_SCHEMA_VERSION,
    OversizePolicy,
    PageBatchError,
    PageBatchPlan,
    plan_page_batches,
)


def spans(plan: PageBatchPlan) -> list[tuple[int, int]]:
    return [(batch.start_page, batch.end_page) for batch in plan.batches]


def covered(plan: PageBatchPlan) -> list[int]:
    return [
        page
        for batch in plan.batches
        for page in range(batch.start_page, batch.end_page)
    ]


def test_empty_document_plans_no_batches() -> None:
    plan = plan_page_batches(0, max_pages_per_batch=4)
    assert plan.batches == ()
    assert (plan.batch_count, plan.page_count) == (0, 0)
    assert plan.schema_version == PAGE_BATCH_SCHEMA_VERSION


def test_exact_division_fills_every_batch() -> None:
    plan = plan_page_batches(8, max_pages_per_batch=4)
    assert spans(plan) == [(0, 4), (4, 8)]
    assert [batch.page_count for batch in plan.batches] == [4, 4]


def test_remainder_becomes_a_final_short_batch() -> None:
    plan = plan_page_batches(9, max_pages_per_batch=4)
    assert spans(plan) == [(0, 4), (4, 8), (8, 9)]
    assert plan.batches[-1].page_count == 1


def test_single_page_batches() -> None:
    plan = plan_page_batches(3, max_pages_per_batch=1)
    assert spans(plan) == [(0, 1), (1, 2), (2, 3)]


def test_batch_larger_than_the_document() -> None:
    plan = plan_page_batches(3, max_pages_per_batch=100)
    assert spans(plan) == [(0, 3)]


@pytest.mark.parametrize("pages", [1, 2, 7, 16, 31, 100])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 16])
def test_batches_partition_the_document_exactly(pages, size) -> None:
    plan = plan_page_batches(pages, max_pages_per_batch=size)
    assert covered(plan) == list(range(pages))
    assert all(batch.page_count <= size for batch in plan.batches)
    assert [batch.batch_index for batch in plan.batches] == list(
        range(plan.batch_count)
    )


def test_pixel_budget_closes_a_batch_early() -> None:
    plan = plan_page_batches(
        5,
        max_pages_per_batch=4,
        page_pixels=[10, 10, 10, 10, 10],
        max_pixels_per_batch=25,
    )
    assert spans(plan) == [(0, 2), (2, 4), (4, 5)]
    assert [batch.pixel_total for batch in plan.batches] == [20, 20, 10]


def test_variable_page_pixels_are_packed_greedily() -> None:
    plan = plan_page_batches(
        6,
        max_pages_per_batch=10,
        page_pixels=[5, 5, 20, 1, 1, 1],
        max_pixels_per_batch=25,
    )
    assert spans(plan) == [(0, 2), (2, 6)]
    assert [batch.pixel_total for batch in plan.batches] == [10, 23]
    assert covered(plan) == list(range(6))


def test_page_pixels_are_reported_without_a_budget() -> None:
    plan = plan_page_batches(3, max_pages_per_batch=3, page_pixels=[1, 2, 3])
    assert spans(plan) == [(0, 3)]
    assert plan.batches[0].pixel_total == 6
    assert plan.max_pixels_per_batch is None


def test_oversize_page_is_rejected_by_default() -> None:
    with pytest.raises(PageBatchError, match="^page_exceeds_pixel_budget$"):
        plan_page_batches(
            4,
            max_pages_per_batch=4,
            page_pixels=[10, 99, 10, 10],
            max_pixels_per_batch=25,
        )


def test_oversize_page_can_be_isolated() -> None:
    plan = plan_page_batches(
        4,
        max_pages_per_batch=4,
        page_pixels=[10, 99, 10, 10],
        max_pixels_per_batch=25,
        oversize_policy=OversizePolicy.ISOLATE,
    )
    assert spans(plan) == [(0, 1), (1, 2), (2, 4)]
    assert [batch.isolated for batch in plan.batches] == [False, True, False]
    assert covered(plan) == [0, 1, 2, 3]


def test_leading_and_consecutive_oversize_pages_isolate_cleanly() -> None:
    plan = plan_page_batches(
        4,
        max_pages_per_batch=4,
        page_pixels=[99, 99, 1, 1],
        max_pixels_per_batch=25,
        oversize_policy="isolate",
    )
    assert spans(plan) == [(0, 1), (1, 2), (2, 4)]
    assert [batch.isolated for batch in plan.batches] == [True, True, False]


def test_trailing_oversize_page_isolates_cleanly() -> None:
    plan = plan_page_batches(
        3,
        max_pages_per_batch=4,
        page_pixels=[1, 1, 99],
        max_pixels_per_batch=25,
        oversize_policy="isolate",
    )
    assert spans(plan) == [(0, 2), (2, 3)]
    assert plan.batches[-1].isolated is True


def test_zero_pixel_pages_do_not_split_batches() -> None:
    plan = plan_page_batches(
        4, max_pages_per_batch=4, page_pixels=[0, 0, 0, 0], max_pixels_per_batch=1
    )
    assert spans(plan) == [(0, 4)]


def test_batch_count_limit_fails_closed() -> None:
    with pytest.raises(PageBatchError, match="^page_batch_count_limit_exceeded$"):
        plan_page_batches(10, max_pages_per_batch=1, max_batches=4)
    assert plan_page_batches(4, max_pages_per_batch=1, max_batches=4).batch_count == 4
    assert MAX_PAGE_BATCH_COUNT > 0


def test_batch_identifiers_are_stable_and_indexed() -> None:
    plan = plan_page_batches(3, max_pages_per_batch=1)
    assert [batch.batch_id for batch in plan.batches] == [
        "b000000",
        "b000001",
        "b000002",
    ]


def test_plan_serialization_is_byte_stable_and_field_ordered() -> None:
    plan = plan_page_batches(
        3, max_pages_per_batch=2, page_pixels=[1, 2, 3], max_pixels_per_batch=10
    )
    assert list(plan.to_dict()) == [
        "schema_version",
        "page_count",
        "max_pages_per_batch",
        "max_pixels_per_batch",
        "oversize_policy",
        "batch_count",
        "batches",
    ]
    assert list(plan.to_dict()["batches"][0]) == [
        "batch_id",
        "batch_index",
        "start_page",
        "end_page",
        "page_count",
        "pixel_total",
        "isolated",
    ]
    assert json.loads(plan.to_json()) == plan.to_dict()
    assert (
        plan.to_json()
        == plan_page_batches(
            3, max_pages_per_batch=2, page_pixels=[1, 2, 3], max_pixels_per_batch=10
        ).to_json()
    )


def test_plans_carry_only_page_numbers_and_counts() -> None:
    payload = json.loads(plan_page_batches(2, max_pages_per_batch=1).to_json())
    for batch in payload["batches"]:
        assert set(batch) == {
            "batch_id",
            "batch_index",
            "start_page",
            "end_page",
            "page_count",
            "pixel_total",
            "isolated",
        }
        assert isinstance(batch["batch_id"], str)
        assert isinstance(batch["isolated"], bool)


def test_zero_batch_size_fails_closed() -> None:
    with pytest.raises(PageBatchError, match="^page_batch_size_invalid$"):
        plan_page_batches(3, max_pages_per_batch=0)


def test_zero_max_batches_fails_closed() -> None:
    with pytest.raises(PageBatchError, match="^page_batch_count_invalid$"):
        plan_page_batches(3, max_pages_per_batch=1, max_batches=0)


def test_zero_pixel_budget_fails_closed() -> None:
    with pytest.raises(PageBatchError, match="^page_pixel_budget_invalid$"):
        plan_page_batches(
            1, max_pages_per_batch=1, page_pixels=[0], max_pixels_per_batch=0
        )


@pytest.mark.parametrize("value", [True, False, 1.0, "4", None])
def test_boolean_and_non_integer_parameters_fail_closed(value) -> None:
    with pytest.raises(PageBatchError, match="^page_count_not_an_integer$"):
        plan_page_batches(value, max_pages_per_batch=4)
    with pytest.raises(PageBatchError, match="^page_batch_size_not_an_integer$"):
        plan_page_batches(4, max_pages_per_batch=value)
    with pytest.raises(PageBatchError, match="^page_batch_count_not_an_integer$"):
        plan_page_batches(4, max_pages_per_batch=4, max_batches=value)


@pytest.mark.parametrize("value", [True, False, 1.0, "4"])
def test_non_integer_pixel_budgets_fail_closed(value) -> None:
    with pytest.raises(PageBatchError, match="^page_pixel_budget_not_an_integer$"):
        plan_page_batches(
            1, max_pages_per_batch=1, page_pixels=[1], max_pixels_per_batch=value
        )


def test_an_unset_pixel_budget_is_not_a_validation_error() -> None:
    plan = plan_page_batches(
        2, max_pages_per_batch=2, page_pixels=[1, 2], max_pixels_per_batch=None
    )
    assert plan.max_pixels_per_batch is None
    assert plan.batches[0].pixel_total == 3


@pytest.mark.parametrize(
    "pages,category",
    [
        (-1, "page_count_out_of_range"),
        (MAX_DOCUMENT_PAGES + 1, "page_count_out_of_range"),
    ],
)
def test_out_of_range_page_counts_fail_closed(pages, category) -> None:
    with pytest.raises(PageBatchError) as excinfo:
        plan_page_batches(pages, max_pages_per_batch=4)
    assert excinfo.value.category == category


def test_overflowing_page_pixels_fail_closed() -> None:
    with pytest.raises(PageBatchError, match="^page_pixels_out_of_range$"):
        plan_page_batches(1, max_pages_per_batch=1, page_pixels=[MAX_PAGE_PIXELS + 1])
    with pytest.raises(PageBatchError, match="^page_pixels_out_of_range$"):
        plan_page_batches(1, max_pages_per_batch=1, page_pixels=[-1])
    with pytest.raises(PageBatchError, match="^page_pixels_out_of_range$"):
        plan_page_batches(1, max_pages_per_batch=1, page_pixels=[2**70])


@pytest.mark.parametrize("value", [True, 1.5, "1", None])
def test_non_integer_page_pixels_fail_closed(value) -> None:
    with pytest.raises(PageBatchError, match="^page_pixels_not_an_integer$"):
        plan_page_batches(1, max_pages_per_batch=1, page_pixels=[value])


@pytest.mark.parametrize("value", ["abc", b"abc", 7])
def test_invalid_page_pixel_containers_fail_closed(value) -> None:
    with pytest.raises(PageBatchError, match="^page_pixels_invalid$"):
        plan_page_batches(3, max_pages_per_batch=1, page_pixels=value)


def test_page_pixel_length_must_match_the_page_count() -> None:
    with pytest.raises(PageBatchError, match="^page_pixels_length_mismatch$"):
        plan_page_batches(3, max_pages_per_batch=1, page_pixels=[1, 2])
    with pytest.raises(PageBatchError, match="^page_pixels_length_mismatch$"):
        plan_page_batches(3, max_pages_per_batch=1, page_pixels=[1, 2, 3, 4])


def test_a_pixel_budget_requires_page_pixels() -> None:
    with pytest.raises(PageBatchError, match="^page_pixels_required$"):
        plan_page_batches(3, max_pages_per_batch=1, max_pixels_per_batch=10)


@pytest.mark.parametrize("value", ["", "Reject", "split", 0, None, True])
def test_unsupported_oversize_policies_fail_closed(value) -> None:
    with pytest.raises(PageBatchError, match="^page_oversize_policy_unsupported$"):
        plan_page_batches(1, max_pages_per_batch=1, oversize_policy=value)


def test_oversize_policy_values_are_a_closed_vocabulary() -> None:
    assert [policy.value for policy in OversizePolicy] == ["reject", "isolate"]
    plan = plan_page_batches(1, max_pages_per_batch=1, oversize_policy="reject")
    assert plan.oversize_policy is OversizePolicy.REJECT


def test_generators_are_accepted_for_page_pixels() -> None:
    plan = plan_page_batches(
        3, max_pages_per_batch=3, page_pixels=(index for index in [1, 2, 3])
    )
    assert plan.batches[0].pixel_total == 6
