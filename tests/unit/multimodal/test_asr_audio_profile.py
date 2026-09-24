"""Synthetic table tests for ASR input profiles and WAV metadata."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.asr_audio_profile import (
    ASR_AUDIO_PROFILE_SCHEMA_VERSION,
    ASR_REASON_CODES,
    MAX_ASR_CHANNEL_COUNT,
    MAX_ASR_DURATION_SECONDS,
    MAX_ASR_SAMPLE_RATE_HZ,
    MONO_16K_PCM_PROFILE,
    SUPPORTED_BIT_DEPTHS,
    SUPPORTED_FORMAT_CODES,
    AsrAudioProfile,
    AsrAudioProfileError,
    AsrCompatibility,
    check_asr_compatibility,
)
from openmed.multimodal.wav_metadata import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    WavMetadata,
)

PROFILE = AsrAudioProfile(
    profile_id="local-asr",
    format_codes=(WAVE_FORMAT_PCM,),
    sample_rates_hz=(8_000, 16_000),
    channel_counts=(1,),
    bit_depths=(16,),
    min_duration_seconds=0.5,
    max_duration_seconds=600.0,
)


def meta(
    *,
    channels=1,
    sample_rate_hz=16_000,
    bit_depth=16,
    format_code=WAVE_FORMAT_PCM,
    frame_count=32_000,
):
    block = channels * bit_depth // 8
    duration = frame_count / sample_rate_hz if sample_rate_hz else 0.0
    return WavMetadata(
        format_code=format_code,
        channels=channels,
        sample_rate_hz=sample_rate_hz,
        bit_depth=bit_depth,
        data_byte_count=frame_count * block,
        frame_count=frame_count,
        duration_seconds=duration,
    )


@pytest.mark.parametrize(
    "overrides,verdict,reasons",
    [
        ({}, AsrCompatibility.COMPATIBLE, ()),
        (
            {"sample_rate_hz": 8_000, "frame_count": 16_000},
            AsrCompatibility.COMPATIBLE,
            (),
        ),
        (
            {"sample_rate_hz": 44_100, "frame_count": 88_200},
            AsrCompatibility.RESAMPLE,
            ("resample_required",),
        ),
        ({"channels": 2}, AsrCompatibility.DOWNMIX, ("downmix_required",)),
        (
            {"channels": 2, "sample_rate_hz": 44_100, "frame_count": 88_200},
            AsrCompatibility.DOWNMIX,
            ("downmix_required", "resample_required"),
        ),
        (
            {"bit_depth": 24},
            AsrCompatibility.INCOMPATIBLE,
            ("bit_depth_unsupported",),
        ),
        (
            {"format_code": WAVE_FORMAT_IEEE_FLOAT, "bit_depth": 32},
            AsrCompatibility.INCOMPATIBLE,
            ("format_unsupported", "bit_depth_unsupported"),
        ),
        (
            {"frame_count": 0},
            AsrCompatibility.INCOMPATIBLE,
            ("empty_audio", "duration_below_minimum"),
        ),
        (
            {"frame_count": 1_600},
            AsrCompatibility.REVIEW,
            ("duration_below_minimum",),
        ),
        (
            {"frame_count": 16_000_000},
            AsrCompatibility.REVIEW,
            ("duration_above_maximum",),
        ),
    ],
)
def test_profile_comparison_table(overrides, verdict, reasons) -> None:
    report = check_asr_compatibility(meta(**overrides), PROFILE)
    assert report.compatibility is verdict
    assert report.reason_codes == reasons
    assert report.is_compatible is (verdict is AsrCompatibility.COMPATIBLE)
    assert report.profile_id == "local-asr"


def test_reason_codes_follow_the_declared_order() -> None:
    report = check_asr_compatibility(
        meta(
            channels=2,
            sample_rate_hz=44_100,
            bit_depth=24,
            format_code=WAVE_FORMAT_IEEE_FLOAT,
            frame_count=1,
        ),
        PROFILE,
    )
    assert report.reason_codes == (
        "format_unsupported",
        "bit_depth_unsupported",
        "duration_below_minimum",
        "downmix_required",
        "resample_required",
    )
    positions = [ASR_REASON_CODES.index(code) for code in report.reason_codes]
    assert positions == sorted(positions)
    assert set(report.reason_codes).issubset(set(ASR_REASON_CODES))


def test_the_worst_verdict_wins() -> None:
    report = check_asr_compatibility(meta(channels=2, bit_depth=24), PROFILE)
    assert report.compatibility is AsrCompatibility.INCOMPATIBLE
    assert report.reason_codes == ("bit_depth_unsupported", "downmix_required")


def test_forbidden_transforms_become_incompatible() -> None:
    strict = replace(PROFILE, allow_resample=False, allow_downmix=False)
    rate = check_asr_compatibility(meta(sample_rate_hz=44_100), strict)
    assert rate.compatibility is AsrCompatibility.INCOMPATIBLE
    assert rate.reason_codes == ("sample_rate_unsupported",)
    channels = check_asr_compatibility(meta(channels=2), strict)
    assert channels.reason_codes == ("channel_count_unsupported",)


def test_too_few_channels_is_never_a_downmix() -> None:
    stereo_only = replace(PROFILE, channel_counts=(2,))
    report = check_asr_compatibility(meta(channels=1), stereo_only)
    assert report.compatibility is AsrCompatibility.INCOMPATIBLE
    assert report.reason_codes == ("channel_count_unsupported",)


def test_duration_bounds_are_inclusive() -> None:
    exact_low = check_asr_compatibility(meta(frame_count=8_000), PROFILE)
    assert exact_low.compatibility is AsrCompatibility.COMPATIBLE
    exact_high = check_asr_compatibility(meta(frame_count=9_600_000), PROFILE)
    assert exact_high.compatibility is AsrCompatibility.COMPATIBLE


def test_report_serialization_is_byte_stable_and_field_ordered() -> None:
    report = check_asr_compatibility(meta(channels=2), PROFILE)
    assert list(report.to_dict()) == [
        "schema_version",
        "profile_id",
        "compatibility",
        "reason_codes",
        "format_code",
        "channels",
        "sample_rate_hz",
        "bit_depth",
        "frame_count",
        "duration_seconds",
    ]
    assert json.loads(report.to_json()) == report.to_dict()
    assert (
        report.to_json() == check_asr_compatibility(meta(channels=2), PROFILE).to_json()
    )
    assert report.schema_version == ASR_AUDIO_PROFILE_SCHEMA_VERSION


def test_reports_carry_only_categorical_and_numeric_metadata() -> None:
    payload = json.loads(check_asr_compatibility(meta(), PROFILE).to_json())
    assert payload["compatibility"] == "compatible"
    assert payload["profile_id"] == "local-asr"
    for key, value in payload.items():
        if key in {"schema_version", "profile_id", "compatibility"}:
            continue
        if key == "reason_codes":
            assert all(item in ASR_REASON_CODES for item in value)
            continue
        assert isinstance(value, (int, float))


def test_profiles_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        PROFILE.sample_rates_hz = (16_000,)  # type: ignore[misc]


def test_bundled_profile_is_generic_and_usable() -> None:
    report = check_asr_compatibility(meta(), MONO_16K_PCM_PROFILE)
    assert report.compatibility is AsrCompatibility.COMPATIBLE
    assert MONO_16K_PCM_PROFILE.profile_id == "mono-16k-pcm"
    assert MONO_16K_PCM_PROFILE.format_codes == (WAVE_FORMAT_PCM,)


@pytest.mark.parametrize("value", ["", "Local", "-lead", "trail-", "a" * 65, 7, None])
def test_invalid_profile_ids_fail_closed(value) -> None:
    with pytest.raises(AsrAudioProfileError, match="^asr_profile_id_invalid$"):
        replace(PROFILE, profile_id=value)


@pytest.mark.parametrize(
    "field,value,category",
    [
        ("sample_rates_hz", (), "asr_profile_rate_values_empty"),
        ("sample_rates_hz", [16_000], "asr_profile_rate_values_invalid"),
        ("sample_rates_hz", (16_000, 8_000), "asr_profile_rate_values_unsorted"),
        ("sample_rates_hz", (16_000, 16_000), "asr_profile_rate_values_unsorted"),
        ("sample_rates_hz", (True,), "asr_profile_rate_values_invalid"),
        ("sample_rates_hz", (16_000.0,), "asr_profile_rate_values_invalid"),
        ("sample_rates_hz", (0,), "asr_profile_rate_values_out_of_range"),
        (
            "sample_rates_hz",
            (MAX_ASR_SAMPLE_RATE_HZ + 1,),
            "asr_profile_rate_values_out_of_range",
        ),
        ("channel_counts", (0,), "asr_profile_channel_values_out_of_range"),
        (
            "channel_counts",
            (MAX_ASR_CHANNEL_COUNT + 1,),
            "asr_profile_channel_values_out_of_range",
        ),
        ("channel_counts", (2, 1), "asr_profile_channel_values_unsorted"),
        ("bit_depths", (12,), "asr_profile_depth_values_unsupported"),
        ("bit_depths", (), "asr_profile_depth_values_empty"),
        ("format_codes", (0xFFFE,), "asr_profile_format_values_unsupported"),
        ("format_codes", (), "asr_profile_format_values_empty"),
    ],
)
def test_invalid_profile_values_fail_closed(field, value, category) -> None:
    with pytest.raises(AsrAudioProfileError) as excinfo:
        replace(PROFILE, **{field: value})
    assert excinfo.value.category == category


@pytest.mark.parametrize("field", ["min_duration_seconds", "max_duration_seconds"])
@pytest.mark.parametrize("value", ["1", None, float("nan"), float("inf")])
def test_invalid_profile_durations_fail_closed(field, value) -> None:
    with pytest.raises(AsrAudioProfileError, match="^asr_profile_duration_invalid$"):
        replace(PROFILE, **{field: value})


@pytest.mark.parametrize("value", [-0.5, MAX_ASR_DURATION_SECONDS + 1])
def test_out_of_range_profile_durations_fail_closed(value) -> None:
    with pytest.raises(
        AsrAudioProfileError, match="^asr_profile_duration_out_of_range$"
    ):
        replace(PROFILE, min_duration_seconds=value)


def test_inverted_duration_range_fails_closed() -> None:
    with pytest.raises(
        AsrAudioProfileError, match="^asr_profile_duration_range_invalid$"
    ):
        replace(PROFILE, min_duration_seconds=10.0, max_duration_seconds=1.0)


def test_integer_durations_are_normalized_to_floats() -> None:
    profile = replace(PROFILE, min_duration_seconds=1, max_duration_seconds=10)
    assert isinstance(profile.min_duration_seconds, float)
    assert (profile.min_duration_seconds, profile.max_duration_seconds) == (1.0, 10.0)


@pytest.mark.parametrize("field", ["allow_resample", "allow_downmix"])
@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_invalid_profile_flags_fail_closed(field, value) -> None:
    with pytest.raises(AsrAudioProfileError, match="^asr_profile_flag_invalid$"):
        replace(PROFILE, **{field: value})


@pytest.mark.parametrize("value", [None, "wav", 7, meta()])
def test_invalid_profile_argument_fails_closed(value) -> None:
    with pytest.raises(AsrAudioProfileError, match="^asr_profile_type_invalid$"):
        check_asr_compatibility(meta(), value)


@pytest.mark.parametrize("value", [None, "wav", 7, PROFILE])
def test_invalid_metadata_argument_fails_closed(value) -> None:
    with pytest.raises(AsrAudioProfileError, match="^asr_metadata_type_invalid$"):
        check_asr_compatibility(value, PROFILE)


def test_supported_vocabularies_are_closed_and_sorted() -> None:
    assert SUPPORTED_BIT_DEPTHS == tuple(sorted(set(SUPPORTED_BIT_DEPTHS)))
    assert set(SUPPORTED_FORMAT_CODES) == {WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT}
    assert len(set(ASR_REASON_CODES)) == len(ASR_REASON_CODES)
    assert [verdict.value for verdict in AsrCompatibility] == [
        "compatible",
        "resample",
        "downmix",
        "review",
        "incompatible",
    ]
