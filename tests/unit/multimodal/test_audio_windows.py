"""Synthetic unit tests for deterministic streaming-audio window planning."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.audio_windows import (
    AUDIO_WINDOW_SCHEMA_VERSION,
    MAX_AUDIO_DURATION_MS,
    MAX_AUDIO_WINDOW_COUNT,
    MAX_AUDIO_WINDOW_MS,
    AudioWindowError,
    AudioWindowPlan,
    TailPolicy,
    plan_audio_windows,
)


def spans(plan: AudioWindowPlan) -> list[tuple[int, int]]:
    return [(window.start_ms, window.end_ms) for window in plan.windows]


def test_exact_division_produces_equal_windows() -> None:
    plan = plan_audio_windows(1500, window_ms=500)
    assert spans(plan) == [(0, 500), (500, 1000), (1000, 1500)]
    assert [window.overlap_ms for window in plan.windows] == [0, 0, 0]
    assert (plan.covered_ms, plan.stride_ms, plan.window_count) == (1500, 500, 3)
    assert plan.schema_version == AUDIO_WINDOW_SCHEMA_VERSION


def test_short_input_is_one_clipped_window() -> None:
    plan = plan_audio_windows(120, window_ms=500)
    assert spans(plan) == [(0, 120)]
    assert plan.windows[0].duration_ms == 120
    assert plan.covered_ms == 120


def test_zero_duration_plans_no_windows() -> None:
    plan = plan_audio_windows(0, window_ms=500)
    assert plan.windows == ()
    assert (plan.covered_ms, plan.window_count) == (0, 0)


def test_one_millisecond_tail_is_kept_by_default() -> None:
    plan = plan_audio_windows(1001, window_ms=500)
    assert spans(plan) == [(0, 500), (500, 1000), (1000, 1001)]
    assert plan.windows[-1].duration_ms == 1
    assert plan.covered_ms == 1001


def test_short_tail_merges_into_the_previous_window() -> None:
    plan = plan_audio_windows(
        1001, window_ms=500, min_tail_ms=100, tail_policy=TailPolicy.MERGE
    )
    assert spans(plan) == [(0, 500), (500, 1001)]
    assert plan.covered_ms == 1001
    assert plan.tail_policy is TailPolicy.MERGE


def test_short_tail_can_be_dropped() -> None:
    plan = plan_audio_windows(1001, window_ms=500, min_tail_ms=100, tail_policy="drop")
    assert spans(plan) == [(0, 500), (500, 1000)]
    assert plan.covered_ms == 1000
    assert plan.duration_ms == 1001


def test_acceptable_tail_is_untouched_by_every_policy() -> None:
    for policy in TailPolicy:
        plan = plan_audio_windows(
            1300, window_ms=500, min_tail_ms=100, tail_policy=policy
        )
        assert spans(plan) == [(0, 500), (500, 1000), (1000, 1300)]


def test_tail_policy_never_removes_the_only_window() -> None:
    for policy in TailPolicy:
        plan = plan_audio_windows(
            40, window_ms=500, min_tail_ms=100, tail_policy=policy
        )
        assert spans(plan) == [(0, 40)]
        assert plan.covered_ms == 40


def test_overlapping_windows_report_the_overlap() -> None:
    plan = plan_audio_windows(1000, window_ms=600, overlap_ms=300)
    assert spans(plan) == [(0, 600), (300, 900), (600, 1000)]
    assert [window.overlap_ms for window in plan.windows] == [0, 300, 300]
    assert plan.stride_ms == 300


def test_planning_stops_once_a_window_reaches_the_end() -> None:
    plan = plan_audio_windows(700, window_ms=600, overlap_ms=300)
    assert spans(plan) == [(0, 600), (300, 700)]


@pytest.mark.parametrize("duration", [1, 7, 999, 1000, 1001, 4321])
@pytest.mark.parametrize("window,overlap", [(500, 0), (600, 300), (250, 249), (1, 0)])
def test_windows_cover_the_duration_without_gaps(duration, window, overlap) -> None:
    plan = plan_audio_windows(duration, window_ms=window, overlap_ms=overlap)
    assert plan.windows[0].start_ms == 0
    assert plan.covered_ms == duration
    for previous, current in zip(plan.windows, plan.windows[1:]):
        assert current.start_ms <= previous.end_ms
        assert current.end_ms > previous.end_ms
        assert current.overlap_ms == previous.end_ms - current.start_ms


@pytest.mark.parametrize("policy", [TailPolicy.DROP, TailPolicy.MERGE])
def test_dropped_and_merged_plans_still_start_at_zero(policy) -> None:
    plan = plan_audio_windows(1001, window_ms=500, min_tail_ms=499, tail_policy=policy)
    assert plan.windows[0].start_ms == 0
    assert plan.covered_ms == (1000 if policy is TailPolicy.DROP else 1001)


def test_window_identifiers_are_stable_and_indexed() -> None:
    plan = plan_audio_windows(1500, window_ms=500)
    assert [window.window_id for window in plan.windows] == [
        "w000000",
        "w000001",
        "w000002",
    ]
    assert [window.window_index for window in plan.windows] == [0, 1, 2]


def test_plan_serialization_is_byte_stable_and_field_ordered() -> None:
    plan = plan_audio_windows(1001, window_ms=500, overlap_ms=100)
    assert list(plan.to_dict()) == [
        "schema_version",
        "duration_ms",
        "window_ms",
        "overlap_ms",
        "stride_ms",
        "min_tail_ms",
        "tail_policy",
        "covered_ms",
        "windows",
    ]
    assert list(plan.to_dict()["windows"][0]) == [
        "window_id",
        "window_index",
        "start_ms",
        "end_ms",
        "overlap_ms",
    ]
    assert json.loads(plan.to_json()) == plan.to_dict()
    assert (
        plan.to_json()
        == plan_audio_windows(1001, window_ms=500, overlap_ms=100).to_json()
    )
    assert plan.to_dict()["tail_policy"] == "keep"


def test_overlap_equal_to_or_greater_than_window_fails_closed() -> None:
    for overlap in (500, 501):
        with pytest.raises(
            AudioWindowError, match="^audio_overlap_not_less_than_window$"
        ):
            plan_audio_windows(1000, window_ms=500, overlap_ms=overlap)


def test_zero_window_size_fails_closed() -> None:
    with pytest.raises(AudioWindowError, match="^audio_window_size_invalid$"):
        plan_audio_windows(1000, window_ms=0)


def test_min_tail_longer_than_the_window_fails_closed() -> None:
    with pytest.raises(AudioWindowError, match="^audio_min_tail_exceeds_window$"):
        plan_audio_windows(1000, window_ms=500, min_tail_ms=501)


@pytest.mark.parametrize("value", [True, False, 1.0, "500", None, b"500"])
def test_boolean_and_non_integer_parameters_fail_closed(value) -> None:
    with pytest.raises(AudioWindowError, match="_not_an_integer$"):
        plan_audio_windows(value, window_ms=500)
    with pytest.raises(AudioWindowError, match="_not_an_integer$"):
        plan_audio_windows(1000, window_ms=value)
    with pytest.raises(AudioWindowError, match="_not_an_integer$"):
        plan_audio_windows(1000, window_ms=500, overlap_ms=value)
    with pytest.raises(AudioWindowError, match="_not_an_integer$"):
        plan_audio_windows(1000, window_ms=500, min_tail_ms=value)


@pytest.mark.parametrize(
    "kwargs,category",
    [
        ({"duration_ms": -1}, "audio_duration_out_of_range"),
        ({"duration_ms": MAX_AUDIO_DURATION_MS + 1}, "audio_duration_out_of_range"),
        ({"duration_ms": 2**70}, "audio_duration_out_of_range"),
        ({"window_ms": -1}, "audio_window_out_of_range"),
        ({"window_ms": MAX_AUDIO_WINDOW_MS + 1}, "audio_window_out_of_range"),
        ({"overlap_ms": -1}, "audio_overlap_out_of_range"),
        ({"min_tail_ms": -1}, "audio_min_tail_out_of_range"),
    ],
)
def test_out_of_range_parameters_fail_closed(kwargs, category) -> None:
    call = {"duration_ms": 1000, "window_ms": 500}
    call.update(kwargs)
    duration = call.pop("duration_ms")
    with pytest.raises(AudioWindowError) as excinfo:
        plan_audio_windows(duration, **call)
    assert excinfo.value.category == category


def test_excessive_window_counts_fail_closed_before_planning() -> None:
    duration = MAX_AUDIO_WINDOW_COUNT + 1
    with pytest.raises(AudioWindowError, match="^audio_window_count_limit_exceeded$"):
        plan_audio_windows(duration, window_ms=1)
    assert (
        plan_audio_windows(MAX_AUDIO_WINDOW_COUNT, window_ms=1).window_count
        == MAX_AUDIO_WINDOW_COUNT
    )


@pytest.mark.parametrize("value", ["", "Keep", "truncate", 0, None, True])
def test_unsupported_tail_policies_fail_closed(value) -> None:
    with pytest.raises(AudioWindowError, match="^audio_tail_policy_unsupported$"):
        plan_audio_windows(1000, window_ms=500, tail_policy=value)


def test_tail_policy_values_are_a_closed_vocabulary() -> None:
    assert [policy.value for policy in TailPolicy] == ["keep", "merge", "drop"]
    assert (
        plan_audio_windows(1000, window_ms=500, tail_policy="keep").tail_policy
        is TailPolicy.KEEP
    )
