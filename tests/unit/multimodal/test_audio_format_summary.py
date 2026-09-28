"""Synthetic unit tests for content-free audio-format distribution summaries."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.audio_format_summary import (
    AUDIO_FORMAT_SUMMARY_SCHEMA_VERSION,
    DEFAULT_MIN_CELL_COUNT,
    SUPPRESSED_CATEGORY,
    AudioFormat,
    AudioFormatRecord,
    AudioFormatSummaryError,
    CategoryCount,
    summarize_audio_formats,
)

_UINT32_MAX = (1 << 32) - 1


def rec(
    fmt=AudioFormat.WAV,
    channels=1,
    rate=16_000,
    depth=16,
    duration=5.0,
) -> AudioFormatRecord:
    return AudioFormatRecord(
        format=fmt,
        channels=channels,
        sample_rate_hz=rate,
        bit_depth=depth,
        duration_seconds=duration,
    )


def test_empty_input_has_a_stable_golden_output() -> None:
    summary = summarize_audio_formats([])
    assert summary.to_json() == (
        '{"schema_version":1,"total_assets":0,"by_format":[],"by_channels":[],'
        '"by_sample_rate_hz":[],"by_bit_depth":[],"by_duration_bucket":[]}'
    )


def test_single_visible_record_with_min_cell_one() -> None:
    summary = summarize_audio_formats([rec()], min_cell_count=1)
    assert summary.total_assets == 1
    assert summary.by_format == (CategoryCount("wav", 1),)
    data = summary.to_dict()
    assert data["by_channels"] == [{"category": 1, "count": 1}]
    assert data["by_sample_rate_hz"] == [{"category": 16000, "count": 1}]
    assert data["by_bit_depth"] == [{"category": 16, "count": 1}]
    assert data["by_duration_bucket"] == [{"category": "1s_to_10s", "count": 1}]


def test_mixed_input_matches_golden_output_with_suppression() -> None:
    records = [
        rec(fmt=AudioFormat.WAV, duration=0.5),
        rec(fmt=AudioFormat.WAV, duration=15.0),
        rec(fmt=AudioFormat.WAV, channels=2, duration=700.0),
        rec(fmt=AudioFormat.FLAC, rate=48_000, duration=5.0),
    ]
    summary = summarize_audio_formats(records)
    assert summary.total_assets == 4
    assert summary.to_json() == (
        '{"schema_version":1,"total_assets":4,'
        '"by_format":[{"category":"wav","count":3},'
        '{"category":"suppressed","count":1}],'
        '"by_channels":[{"category":1,"count":3},'
        '{"category":"suppressed","count":1}],'
        '"by_sample_rate_hz":[{"category":16000,"count":3},'
        '{"category":"suppressed","count":1}],'
        '"by_bit_depth":[{"category":16,"count":4}],'
        '"by_duration_bucket":[{"category":"suppressed","count":4}]}'
    )


def test_reordered_inputs_produce_identical_output() -> None:
    records = [
        rec(fmt=AudioFormat.WAV, duration=0.5),
        rec(fmt=AudioFormat.MP3, channels=2, rate=44_100, depth=8, duration=120.0),
        rec(fmt=AudioFormat.OGG, channels=2, duration=30.0),
        rec(fmt=AudioFormat.WAV, duration=1.0),
        rec(fmt=AudioFormat.M4A, rate=22_050, depth=24, duration=640.0),
    ]
    first = summarize_audio_formats(records, min_cell_count=1)
    second = summarize_audio_formats(reversed(records), min_cell_count=1)
    assert first.to_json() == second.to_json()


def test_small_cell_threshold_is_inclusive() -> None:
    pair = [rec(fmt=AudioFormat.FLAC), rec(fmt=AudioFormat.FLAC)]
    visible = summarize_audio_formats(pair, min_cell_count=2)
    assert visible.by_format[0].category == "flac"
    assert visible.by_format[0].count == 2
    suppressed = summarize_audio_formats(pair[:1], min_cell_count=2)
    assert suppressed.by_format == (CategoryCount(SUPPRESSED_CATEGORY, 1),)


@pytest.mark.parametrize(
    "duration,bucket",
    [
        (0.0, "under_1s"),
        (0.999, "under_1s"),
        (1.0, "1s_to_10s"),
        (9.999, "1s_to_10s"),
        (10.0, "10s_to_1m"),
        (59.99, "10s_to_1m"),
        (60.0, "1m_to_10m"),
        (599.99, "1m_to_10m"),
        (600.0, "10m_or_more"),
        (5000.0, "10m_or_more"),
    ],
)
def test_duration_bucket_boundaries(duration, bucket) -> None:
    summary = summarize_audio_formats([rec(duration=duration)], min_cell_count=1)
    assert summary.by_duration_bucket[0].category == bucket


@pytest.mark.parametrize(
    "kwargs,category",
    [
        ({"fmt": "wav"}, "audio_format_format_invalid"),
        ({"fmt": 3}, "audio_format_format_invalid"),
        ({"fmt": None}, "audio_format_format_invalid"),
        ({"channels": 0}, "audio_format_channels_invalid"),
        ({"channels": 65}, "audio_format_channels_invalid"),
        ({"channels": True}, "audio_format_channels_invalid"),
        ({"channels": 1.0}, "audio_format_channels_invalid"),
        ({"rate": 0}, "audio_format_sample_rate_invalid"),
        ({"rate": _UINT32_MAX + 1}, "audio_format_sample_rate_invalid"),
        ({"rate": True}, "audio_format_sample_rate_invalid"),
        ({"depth": 0}, "audio_format_bit_depth_invalid"),
        ({"depth": 65}, "audio_format_bit_depth_invalid"),
        ({"depth": True}, "audio_format_bit_depth_invalid"),
        ({"duration": -1.0}, "audio_format_duration_invalid"),
        ({"duration": float("nan")}, "audio_format_duration_invalid"),
        ({"duration": float("inf")}, "audio_format_duration_invalid"),
        ({"duration": True}, "audio_format_duration_invalid"),
        ({"duration": "short"}, "audio_format_duration_invalid"),
        ({"duration": 2**31}, "audio_format_duration_invalid"),
    ],
)
def test_invalid_records_fail_closed(kwargs, category) -> None:
    with pytest.raises(AudioFormatSummaryError, match=f"^{category}$") as raised:
        rec(**kwargs)
    assert raised.value.category == category
    assert str(raised.value) == category


@pytest.mark.parametrize("foreign", [None, b"samples", "note.wav", 42, {"a": 1}])
def test_non_record_entries_are_rejected(foreign) -> None:
    with pytest.raises(AudioFormatSummaryError, match="^audio_format_record_type$"):
        summarize_audio_formats([rec(), foreign])


def test_invalid_records_produce_no_partial_summary() -> None:
    with pytest.raises(AudioFormatSummaryError, match="^audio_format_format_invalid$"):
        summarize_audio_formats([rec(), rec(fmt="mp3")])
    with pytest.raises(AudioFormatSummaryError):
        summarize_audio_formats([rec(fmt="mp3"), rec()])


@pytest.mark.parametrize("min_cell_count", [None, True, 0, -1, 2.5, "3", []])
def test_invalid_min_cell_count_fails_with_constant_message(min_cell_count) -> None:
    with pytest.raises(ValueError, match="^min_cell_count"):
        summarize_audio_formats([rec()], min_cell_count=min_cell_count)


def test_default_min_cell_count_is_three() -> None:
    records = [rec()] * 3
    summary = summarize_audio_formats(records)
    assert summary.by_format[0].count == 3
    assert DEFAULT_MIN_CELL_COUNT == 3


def test_suppression_pools_multiple_small_cells() -> None:
    records = [
        rec(fmt=AudioFormat.WAV),
        rec(fmt=AudioFormat.WAV),
        rec(fmt=AudioFormat.WAV),
        rec(fmt=AudioFormat.FLAC),
        rec(fmt=AudioFormat.MP3),
        rec(fmt=AudioFormat.OGG),
    ]
    summary = summarize_audio_formats(records)
    assert summary.by_format == (
        CategoryCount("wav", 3),
        CategoryCount("suppressed", 3),
    )


def test_error_is_a_value_error_without_values() -> None:
    error = AudioFormatSummaryError("audio_format_record_type")
    assert isinstance(error, ValueError)
    assert error.category == "audio_format_record_type"
    assert str(error) == "audio_format_record_type"


def test_serialization_is_deterministic() -> None:
    summary = summarize_audio_formats([rec()], min_cell_count=1)
    assert summary.schema_version == AUDIO_FORMAT_SUMMARY_SCHEMA_VERSION
    data = summary.to_dict()
    assert list(data) == [
        "schema_version",
        "total_assets",
        "by_format",
        "by_channels",
        "by_sample_rate_hz",
        "by_bit_depth",
        "by_duration_bucket",
    ]
    assert json.loads(summary.to_json()) == data


def test_records_have_no_content_or_free_form_fields() -> None:
    import dataclasses

    assert [field.name for field in dataclasses.fields(AudioFormatRecord)] == [
        "format",
        "channels",
        "sample_rate_hz",
        "bit_depth",
        "duration_seconds",
    ]
    summary = summarize_audio_formats([rec()], min_cell_count=1)
    rendered = summary.to_json()
    for forbidden in (
        "samples",
        "filename",
        "path",
        "transcript",
        "patient",
        ".wav",
    ):
        assert forbidden not in rendered
