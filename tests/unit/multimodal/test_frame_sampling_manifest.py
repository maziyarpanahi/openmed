"""Synthetic unit tests for content-free frame-sampling manifests."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.frame_sampling_manifest import (
    FRAME_SAMPLING_SCHEMA_VERSION,
    FrameSamplingManifest,
    FrameSamplingManifestError,
    SamplingMode,
    canonical_coverage_digest,
)

_MS = (1, 1000)


def digest(mode, timestamps, time_base=_MS):
    return canonical_coverage_digest(mode, time_base[0], time_base[1], timestamps)


_COMPUTE = object()


def manifest(
    *,
    mode=SamplingMode.UNIFORM,
    duration=10_000,
    time_base=_MS,
    timestamps=(0, 500, 1000),
    step=None,
    window=None,
    declared_digest=_COMPUTE,
    schema_version=FRAME_SAMPLING_SCHEMA_VERSION,
):
    step_ticks = step
    window_start, window_end = (None, None) if window is None else window
    return FrameSamplingManifest(
        duration_ticks=duration,
        time_base_num=time_base[0],
        time_base_den=time_base[1],
        sampling_mode=mode,
        timestamps=timestamps,
        coverage_digest=digest(mode, timestamps, time_base)
        if declared_digest is _COMPUTE
        else declared_digest,
        step_ticks=step_ticks,
        window_start_ticks=window_start,
        window_end_ticks=window_end,
        schema_version=schema_version,
    )


def test_uniform_manifest_declares_and_matches_its_step() -> None:
    result = manifest(timestamps=(0, 500, 1000), step=500)
    assert result.sampling_mode is SamplingMode.UNIFORM
    assert result.step_ticks == 500
    assert result.timestamps == (0, 500, 1000)


def test_keyframe_manifest_accepts_arbitrary_strictly_increasing_positions() -> None:
    result = manifest(mode=SamplingMode.KEYFRAME, timestamps=(250, 900, 4_000, 9_999))
    assert result.window_start_ticks is None
    assert result.step_ticks is None


def test_bounded_window_manifest_declares_and_matches_its_window() -> None:
    result = manifest(
        mode=SamplingMode.BOUNDED_WINDOW,
        timestamps=(2_000, 2_500),
        window=(2_000, 3_000),
    )
    assert result.window_start_ticks == 2_000
    assert result.window_end_ticks == 3_000


def test_empty_selection_is_valid_in_every_mode() -> None:
    for mode in SamplingMode:
        kwargs = {"timestamps": ()}
        if mode is SamplingMode.UNIFORM:
            kwargs["step"] = 100
        if mode is SamplingMode.BOUNDED_WINDOW:
            kwargs["window"] = (0, 10)
        result = manifest(mode=mode, **kwargs)
        assert result.timestamps == ()


def test_coverage_digest_is_reproducible_and_order_stable() -> None:
    first = digest(SamplingMode.UNIFORM, (0, 500, 1000))
    second = digest(SamplingMode.UNIFORM, (0, 500, 1000))
    assert first == second
    assert first == manifest(timestamps=(0, 500, 1000), step=500).coverage_digest
    assert len(first) == 64


def test_coverage_digest_binds_mode_and_time_base() -> None:
    ticks = (0, 500)
    assert digest(SamplingMode.UNIFORM, ticks) != digest(SamplingMode.KEYFRAME, ticks)
    assert digest(SamplingMode.UNIFORM, ticks, (1, 1000)) != digest(
        SamplingMode.UNIFORM, ticks, (1, 500)
    )


def test_declared_digest_must_match_recomputed_coverage() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_coverage_digest_mismatch$"
    ):
        manifest(
            mode=SamplingMode.KEYFRAME,
            timestamps=(0, 100, 200),
            declared_digest=digest(SamplingMode.KEYFRAME, (0, 100, 201)),
        )


def test_reordered_timestamps_fail_monotonicity() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_timestamps_not_increasing$"
    ):
        manifest(
            timestamps=(500, 0, 1000),
            step=None,
            mode=SamplingMode.KEYFRAME,
            declared_digest="0" * 64,
        )


def test_duplicate_timestamps_fail_monotonicity() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_timestamps_not_increasing$"
    ):
        manifest(
            timestamps=(500, 500, 900),
            step=None,
            mode=SamplingMode.KEYFRAME,
            declared_digest="0" * 64,
        )


@pytest.mark.parametrize("tick", [-1, 10_000, 10_001])
def test_out_of_range_timestamps_fail(tick) -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_timestamps_out_of_range$"
    ):
        manifest(
            timestamps=(0, tick),
            step=None,
            mode=SamplingMode.KEYFRAME,
            declared_digest="0" * 64,
        )


@pytest.mark.parametrize("value", [None, 1.5, "0", True, []])
def test_non_integer_timestamps_fail(value) -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_timestamps_invalid$"
    ):
        manifest(
            timestamps=(0, value),
            step=None,
            mode=SamplingMode.KEYFRAME,
            declared_digest="0" * 64,
        )


@pytest.mark.parametrize(
    "num,den", [(0, 1000), (1, 0), (-1, 1000), (1.5, 1000), (True, 2)]
)
def test_invalid_time_bases_fail(num, den) -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_time_base_invalid$"
    ):
        manifest(time_base=(num, den), step=None, mode=SamplingMode.KEYFRAME)


def test_unreduced_time_base_fails() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_time_base_not_reduced$"
    ):
        manifest(
            time_base=(2, 2000),
            timestamps=(0, 1000),
            step=None,
            mode=SamplingMode.KEYFRAME,
        )


@pytest.mark.parametrize("duration", [None, 0, -5, 1.5, True])
def test_invalid_duration_fails(duration) -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_duration_invalid$"
    ):
        manifest(duration=duration, step=None, mode=SamplingMode.KEYFRAME)


def test_unsupported_mode_fails() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_mode_unsupported$"
    ):
        manifest(mode="uniform", step=None)


@pytest.mark.parametrize(
    "step,category",
    [
        (None, "frame_sampling_step_invalid"),
        (0, "frame_sampling_step_invalid"),
        (-1, "frame_sampling_step_invalid"),
        (1.5, "frame_sampling_step_invalid"),
        (True, "frame_sampling_step_invalid"),
        (100, "frame_sampling_step_mismatch"),
    ],
)
def test_uniform_step_is_validated_against_positions(step, category) -> None:
    with pytest.raises(FrameSamplingManifestError, match=f"^{category}$"):
        manifest(timestamps=(0, 500, 1000), step=step)


def test_uniform_mode_rejects_window_fields() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_strategy_fields_invalid$"
    ):
        manifest(
            timestamps=(0, 500),
            step=500,
            window=(0, 1_000),
            declared_digest=digest(SamplingMode.BOUNDED_WINDOW, (0, 500)),
        )


def test_keyframe_mode_rejects_strategy_fields() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_strategy_fields_invalid$"
    ):
        manifest(
            mode=SamplingMode.KEYFRAME,
            timestamps=(0, 500),
            step=500,
        )


@pytest.mark.parametrize(
    "window,category",
    [
        ((None, 5_000), "frame_sampling_window_invalid"),
        ((2_000, None), "frame_sampling_window_invalid"),
        ((5_000, 2_000), "frame_sampling_window_invalid"),
        ((-1, 5_000), "frame_sampling_window_invalid"),
        ((2_000, 10_001), "frame_sampling_window_invalid"),
        ((2_000, 2_000), "frame_sampling_window_invalid"),
    ],
)
def test_invalid_windows_fail(window, category) -> None:
    with pytest.raises(FrameSamplingManifestError, match=f"^{category}$"):
        manifest(
            mode=SamplingMode.BOUNDED_WINDOW,
            timestamps=(2_000, 2_500),
            window=window,
        )


def test_window_must_cover_every_position() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_window_mismatch$"
    ):
        manifest(
            mode=SamplingMode.BOUNDED_WINDOW,
            timestamps=(1_000, 2_500),
            window=(2_000, 3_000),
        )


def test_window_end_is_exclusive() -> None:
    result = manifest(
        mode=SamplingMode.BOUNDED_WINDOW,
        timestamps=(2_999,),
        window=(2_000, 3_000),
    )
    assert result.timestamps == (2_999,)
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_window_mismatch$"
    ):
        manifest(
            mode=SamplingMode.BOUNDED_WINDOW,
            timestamps=(3_000,),
            window=(2_000, 3_000),
        )


def test_bounded_window_rejects_step_fields() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_strategy_fields_invalid$"
    ):
        manifest(
            mode=SamplingMode.BOUNDED_WINDOW,
            timestamps=(2_000, 2_500),
            window=(2_000, 3_000),
            step=500,
        )


@pytest.mark.parametrize(
    "declared",
    ["", "abc", "0" * 63, "z" * 64, None, 42],
)
def test_malformed_declared_digest_fails(declared) -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_coverage_digest_invalid$"
    ):
        manifest(declared_digest=declared, step=None, mode=SamplingMode.KEYFRAME)


def test_unsupported_schema_version_fails() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_schema_unsupported$"
    ):
        manifest(
            schema_version=2,
            step=None,
            mode=SamplingMode.KEYFRAME,
            declared_digest=digest(SamplingMode.KEYFRAME, (0, 100, 200)),
        )


def test_error_is_a_value_error_without_values() -> None:
    error = FrameSamplingManifestError("frame_sampling_step_invalid")
    assert isinstance(error, ValueError)
    assert error.category == "frame_sampling_step_invalid"
    assert str(error) == "frame_sampling_step_invalid"


def test_digest_helper_validates_its_arguments() -> None:
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_mode_unsupported$"
    ):
        canonical_coverage_digest("uniform", 1, 1000, (0,))
    with pytest.raises(
        FrameSamplingManifestError, match="^frame_sampling_timestamps_invalid$"
    ):
        canonical_coverage_digest(SamplingMode.UNIFORM, 1, 1000, (0, 1.5))


def test_serialization_is_deterministic() -> None:
    result = manifest(
        mode=SamplingMode.BOUNDED_WINDOW,
        timestamps=(2_000, 2_500),
        window=(2_000, 3_000),
    )
    data = result.to_dict()
    assert list(data) == [
        "schema_version",
        "duration_ticks",
        "time_base_num",
        "time_base_den",
        "sampling_mode",
        "timestamps",
        "coverage_digest",
        "window_start_ticks",
        "window_end_ticks",
    ]
    assert json.loads(result.to_json()) == data
    repeat = manifest(
        mode=SamplingMode.BOUNDED_WINDOW,
        timestamps=(2_000, 2_500),
        window=(2_000, 3_000),
    )
    assert repeat.to_json() == result.to_json()


def test_serialization_omits_absent_strategy_fields() -> None:
    data = manifest(mode=SamplingMode.KEYFRAME, timestamps=(0, 100)).to_dict()
    assert "step_ticks" not in data
    assert "window_start_ticks" not in data
    assert "window_end_ticks" not in data


def test_manifest_carries_no_content_or_identifiers() -> None:
    result = manifest(timestamps=(0, 500, 1000), step=500)
    rendered = result.to_json()
    for forbidden in ("frame", "audio", "caption", "patient", "mp4", "id", "path"):
        assert forbidden not in rendered
