"""Synthetic unit tests for deterministic audio resampling plans."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.audio_resample_plan import (
    AUDIO_RESAMPLE_PLAN_SCHEMA_VERSION,
    MAX_AUDIO_FRAMES,
    MAX_AUDIO_RATE_HZ,
    AudioResamplePlanError,
    ResampleStatus,
    RoundingPolicy,
    plan_audio_resampling,
)

_UINT32_MAX = (1 << 32) - 1


def test_identity_plan_is_exact() -> None:
    plan = plan_audio_resampling(16000, 16000, 48000)
    assert plan.target_frames == 48000
    assert (plan.rate_ratio_numerator, plan.rate_ratio_denominator) == (1, 1)
    assert plan.status is ResampleStatus.EXACT
    assert plan.duration_error_seconds == 0.0


def test_integer_ratio_matches_hand_calculation() -> None:
    plan = plan_audio_resampling(48000, 16000, 9000)
    assert plan.target_frames == 3000
    assert (plan.rate_ratio_numerator, plan.rate_ratio_denominator) == (3, 1)
    assert plan.status is ResampleStatus.EXACT
    assert plan.duration_error_seconds == 0.0


def test_non_integer_ratio_exact_conversion() -> None:
    plan = plan_audio_resampling(44100, 16000, 44100)
    assert plan.target_frames == 16000
    assert (plan.rate_ratio_numerator, plan.rate_ratio_denominator) == (441, 160)
    assert plan.status is ResampleStatus.EXACT
    assert plan.duration_error_seconds == 0.0


@pytest.mark.parametrize(
    "policy,frames_count,error",
    [
        (RoundingPolicy.FLOOR, 3, -0.0000625),
        (RoundingPolicy.CEILING, 4, 4000 / 192_000_000),
        (RoundingPolicy.NEAREST_EVEN, 4, 4000 / 192_000_000),
    ],
)
def test_rounded_policies_match_hand_calculations(policy, frames_count, error) -> None:
    plan = plan_audio_resampling(16000, 12000, 5, rounding=policy)
    assert plan.target_frames == frames_count
    assert plan.status is ResampleStatus.ROUNDED
    assert (plan.rate_ratio_numerator, plan.rate_ratio_denominator) == (4, 3)
    assert plan.duration_error_seconds == pytest.approx(error)
    assert plan.rounding is policy


@pytest.mark.parametrize(
    "frames_count,expected",
    [(5, 2), (7, 4), (6, 3)],
)
def test_nearest_even_resolves_ties_to_even(frames_count, expected) -> None:
    plan = plan_audio_resampling(
        2, 1, frames_count, rounding=RoundingPolicy.NEAREST_EVEN
    )
    assert plan.target_frames == expected


def test_one_frame_plans_match_hand_calculations() -> None:
    floor_plan = plan_audio_resampling(16000, 8000, 1, rounding=RoundingPolicy.FLOOR)
    assert floor_plan.target_frames == 0
    assert floor_plan.duration_error_seconds == pytest.approx(-1 / 16000)
    ceiling_plan = plan_audio_resampling(
        16000, 8000, 1, rounding=RoundingPolicy.CEILING
    )
    assert ceiling_plan.target_frames == 1
    assert ceiling_plan.duration_error_seconds == pytest.approx(1 / 16000)
    even_plan = plan_audio_resampling(16000, 8000, 1)
    assert even_plan.target_frames == 0


def test_zero_frames_is_a_valid_exact_plan() -> None:
    plan = plan_audio_resampling(16000, 8000, 0)
    assert plan.source_frames == 0
    assert plan.target_frames == 0
    assert plan.status is ResampleStatus.EXACT
    assert plan.duration_error_seconds == 0.0


def test_long_duration_matches_hand_calculation() -> None:
    plan = plan_audio_resampling(16000, 48000, _UINT32_MAX)
    assert plan.target_frames == 3 * _UINT32_MAX
    assert plan.status is ResampleStatus.EXACT
    assert plan.duration_error_seconds == 0.0


def test_product_overflow_fails_categorically() -> None:
    with pytest.raises(
        AudioResamplePlanError, match="^audio_resample_product_overflow$"
    ):
        plan_audio_resampling(1, _UINT32_MAX, _UINT32_MAX)


def test_rate_bound_is_inclusive() -> None:
    plan = plan_audio_resampling(_UINT32_MAX, _UINT32_MAX, 10)
    assert plan.source_rate_hz == MAX_AUDIO_RATE_HZ
    assert plan.target_frames == 10


@pytest.mark.parametrize("field", ["source_rate", "target_rate", "frames"])
@pytest.mark.parametrize(
    "value,category_name",
    [
        (None, "missing"),
        (True, "boolean"),
        (False, "boolean"),
        (float("nan"), "not_finite"),
        (float("inf"), "not_finite"),
        (16000.0, "not_integer"),
        ("16000", "not_integer"),
        ([], "not_integer"),
        (-1, "not_positive"),
        (_UINT32_MAX + 1, "overflow"),
    ],
)
def test_invalid_counts_fail_closed(field, value, category_name) -> None:
    kwargs = {"source_rate_hz": 16000, "target_rate_hz": 8000, "source_frames": 10}
    kwargs[
        {
            "source_rate": "source_rate_hz",
            "target_rate": "target_rate_hz",
            "frames": "source_frames",
        }[field]
    ] = value
    with pytest.raises(
        AudioResamplePlanError, match=f"^audio_resample_{field}_{category_name}$"
    ):
        plan_audio_resampling(**kwargs)


def test_zero_rates_fail_closed() -> None:
    with pytest.raises(
        AudioResamplePlanError, match="^audio_resample_source_rate_not_positive$"
    ):
        plan_audio_resampling(0, 8000, 10)
    with pytest.raises(
        AudioResamplePlanError, match="^audio_resample_target_rate_not_positive$"
    ):
        plan_audio_resampling(16000, 0, 10)


def test_frames_overflow_fails_closed() -> None:
    with pytest.raises(
        AudioResamplePlanError, match="^audio_resample_frames_overflow$"
    ):
        plan_audio_resampling(16000, 8000, MAX_AUDIO_FRAMES + 1)


@pytest.mark.parametrize("rounding", [None, "floor", 3, RoundingPolicy])
def test_invalid_rounding_policy_fails_with_constant_message(rounding) -> None:
    with pytest.raises(ValueError, match="^rounding must be a RoundingPolicy$"):
        plan_audio_resampling(16000, 8000, 10, rounding=rounding)


def test_tolerance_rejects_excessive_error() -> None:
    with pytest.raises(AudioResamplePlanError, match="^audio_resample_error_exceeded$"):
        plan_audio_resampling(16000, 12000, 5, max_duration_error_seconds=0.0)
    plan = plan_audio_resampling(16000, 12000, 5, max_duration_error_seconds=1 / 12000)
    assert plan.target_frames == 4


def test_tolerance_accepts_exact_plans() -> None:
    plan = plan_audio_resampling(48000, 16000, 9000, max_duration_error_seconds=0.0)
    assert plan.status is ResampleStatus.EXACT


@pytest.mark.parametrize("tolerance", [-1, float("nan"), "0.5", []])
def test_invalid_tolerance_fails_with_constant_message(tolerance) -> None:
    with pytest.raises(ValueError, match="^max_duration_error_seconds"):
        plan_audio_resampling(16000, 8000, 10, max_duration_error_seconds=tolerance)


def test_error_is_a_value_error_without_values() -> None:
    error = AudioResamplePlanError("audio_resample_frames_missing")
    assert isinstance(error, ValueError)
    assert error.category == "audio_resample_frames_missing"
    assert str(error) == "audio_resample_frames_missing"


def test_serialization_is_deterministic() -> None:
    plan = plan_audio_resampling(16000, 12000, 5, rounding=RoundingPolicy.CEILING)
    assert plan.schema_version == AUDIO_RESAMPLE_PLAN_SCHEMA_VERSION
    data = plan.to_dict()
    assert list(data) == [
        "schema_version",
        "source_rate_hz",
        "target_rate_hz",
        "source_frames",
        "target_frames",
        "rate_ratio_numerator",
        "rate_ratio_denominator",
        "rounding",
        "status",
        "duration_error_seconds",
    ]
    assert json.loads(plan.to_json()) == data
    repeat = plan_audio_resampling(16000, 12000, 5, rounding=RoundingPolicy.CEILING)
    assert repeat.to_json() == plan.to_json()
