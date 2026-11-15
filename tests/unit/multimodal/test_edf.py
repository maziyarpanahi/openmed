"""Offline safety, timing, bounds and hand-checked EDF decoding controls."""

import io
import json
from dataclasses import replace

import pytest

from openmed.multimodal.edf import EDF_NOTICE, EdfError, EdfLimits, read_edf
from tests.fixtures.multimodal.edf import replace_field, synthetic_edf


@pytest.mark.parametrize("kind", ["EDF", "EDF+C", "EDF+D"])
def test_hand_checked_samples_and_timing(kind):
    result = read_edf(synthetic_edf(kind=kind))
    assert result.format == kind
    assert result.record_onsets_seconds == (0, 1)
    assert result.record_duration_seconds == 1
    assert result.signals[0].label == "ECG II"
    assert result.signals[0].sampling_rate_hz == 4
    assert result.signals[0].digital_minimum == -2
    assert result.signals[0].digital_maximum == 2
    assert result.signals[0].physical_minimum == -1
    assert result.signals[0].physical_maximum == 1
    assert result.records[0].signals[0].digital_samples == (-2, -1, 0, 2)
    assert result.records[0].signals[0].physical_samples == (-1, -0.5, 0, 1)
    assert result.records[1].signals[0].physical_samples == (1, 0, -0.5, -1)
    assert result.gaps == ()
    assert result.notice == EDF_NOTICE
    assert result.report()["review_required"] is True
    assert result.reviewed(confirmed=True).report()["review_required"] is False
    with pytest.raises(EdfError, match="^edf_review_required$"):
        result.reviewed(confirmed=1)


def test_discontinuous_gap_and_half_open_window():
    source = synthetic_edf(kind="EDF+D", onsets=("+0.25", "+3.25"))
    result = read_edf(source, start_seconds=0.5, end_seconds=3.5)
    assert result.record_onsets_seconds == (0.25, 3.25)
    assert [
        (g.before_record_index, g.start_seconds, g.end_seconds) for g in result.gaps
    ] == [(1, 1.25, 3.25)]
    assert result.records[0].signals[0].first_sample_index == 1
    assert result.records[0].signals[0].digital_samples == (-1, 0, 2)
    assert result.records[1].signals[0].digital_samples == (2,)
    assert read_edf(source, start_seconds=1.25, end_seconds=3.25).records == ()
    assert len(result.annotations) == 2
    assert result.annotations[0].onset_seconds == 0.25
    assert result.annotations[0].duration_seconds == 0.5
    assert result.annotations[0].count == 2


def test_fractional_continuous_timing_and_unknown_count():
    source = synthetic_edf(
        kind="EDF+C", duration="0.1", onsets=("+0.1", "+0.2"), declared=-1
    )
    result = read_edf(source, start_seconds=0.125, end_seconds=0.2)
    assert result.record_onsets_seconds == (0.1, 0.2)
    assert result.signals[0].sampling_rate_hz == 40
    assert result.records[0].signals[0].digital_samples == (-1, 0, 2)


def test_negative_gain_and_multiple_annotation_channels():
    source = synthetic_edf(kind="EDF+C", extra_annotation_channel=True)
    # Field-major physical minimum and maximum, 3 signals.
    source = replace_field(source, 256 + 104 * 3, 8, 1)
    source = replace_field(source, 256 + 112 * 3, 8, -1)
    result = read_edf(source)
    assert result.records[0].signals[0].physical_samples == (1, 0.5, 0, -1)
    assert result.report()["annotation_count"] == 6


@pytest.mark.parametrize("kind", ["EDF", "EDF+C", "EDF+D"])
def test_privacy_sweep_success_and_errors(kind):
    sentinel = "SYNTHETIC_SECRET"
    source = synthetic_edf(
        kind=kind,
        patient=sentinel,
        recording=sentinel,
        label=sentinel,
        annotation_text=sentinel + " 患者 José",
    )
    result = read_edf(source)
    output = repr(result) + json.dumps(result.report())
    for private in (sentinel, "09.10.26", "12.34.56", "患者", "José"):
        assert private not in output
    assert result.signals[0].label == "withheld"
    assert (
        result.patient.present and result.patient.anonymization_status == "not_verified"
    )
    for offset, size in [
        (0, 8),
        (184, 8),
        (236, 8),
        (244, 8),
        (252, 4),
        (168, 8),
        (176, 8),
    ]:
        bad = replace_field(source, offset, size, b"SECRET"[:size])
        with pytest.raises(EdfError) as caught:
            read_edf(bad)
        assert "SECRET" not in str(caught.value)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "patient,recording,status",
    [
        ("", "", "absent"),
        ("X X X X", "Startdate X X X X", "placeholder_only"),
        ("X X X X extra", "Startdate X X X X extra", "not_verified"),
    ],
)
def test_identity_status_is_not_an_anonymization_claim(patient, recording, status):
    result = read_edf(synthetic_edf(patient=patient, recording=recording))
    assert result.patient.anonymization_status == status
    assert result.recording.anonymization_status == status


@pytest.mark.parametrize(
    "offset,size,value,code",
    [
        (0, 8, "1", "edf_header_invalid"),
        (168, 8, "31.02.26", "edf_header_invalid"),
        (176, 8, "24.00.00", "edf_header_invalid"),
        (184, 8, "257", "edf_header_size_invalid"),
        (236, 8, "0", "edf_record_count_invalid"),
        (236, 8, "-2", "edf_record_count_invalid"),
        (236, 8, "100001", "edf_record_limit"),
        (244, 8, "nan", "edf_numeric_invalid"),
        (244, 8, "1E99", "edf_numeric_invalid"),
        (244, 8, "-1", "edf_duration_limit"),
        (244, 8, "0", "edf_duration_invalid"),
        (252, 4, "65", "edf_signal_limit"),
        (256 + 104, 8, "1", "edf_range_invalid"),
        (256 + 120, 8, "3", "edf_range_invalid"),
        (256 + 128, 8, "32768", "edf_range_invalid"),
        (256 + 216, 8, "0", "edf_samples_invalid"),
        (256 + 216, 8, "99999999", "edf_record_byte_limit"),
    ],
)
def test_header_failures_have_stable_codes(offset, size, value, code):
    with pytest.raises(EdfError, match=f"^{code}$"):
        read_edf(replace_field(synthetic_edf(), offset, size, value))


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data[:-1],
        lambda data: data + b"x",
        lambda data: replace_field(data, 236, 8, "1"),
        lambda data: replace_field(data, 236, 8, "3"),
    ],
)
def test_inconsistent_record_size_and_counts(change):
    with pytest.raises(EdfError) as caught:
        read_edf(change(synthetic_edf()))
    assert caught.value.code in ("edf_truncated", "edf_record_count_invalid")


@pytest.mark.parametrize(
    "name,value,code",
    [
        ("max_bytes", 520, "edf_byte_limit"),
        ("max_signals", 1, "edf_signal_limit"),
        ("max_records", 1, "edf_record_limit"),
        ("max_record_bytes", 4, "edf_record_byte_limit"),
        ("max_duration_seconds", 1, "edf_duration_limit"),
        ("max_window_seconds", 1, "edf_window_invalid"),
        ("max_output_samples", 7, "edf_output_sample_limit"),
        ("max_annotation_lists", 3, "edf_annotation_limit"),
    ],
)
def test_resource_limits(name, value, code):
    with pytest.raises(EdfError, match=f"^{code}$"):
        read_edf(
            synthetic_edf(kind="EDF+C"), limits=replace(EdfLimits(), **{name: value})
        )


@pytest.mark.parametrize(
    "start,end",
    [
        (True, 1),
        (-1, 1),
        (0, 0),
        (1, 0),
        (float("nan"), 1),
        (0, float("inf")),
        (0, 3601),
        (10**1000, 1),
    ],
)
def test_invalid_windows(start, end):
    with pytest.raises(EdfError, match="^edf_window_invalid$"):
        read_edf(synthetic_edf(), start_seconds=start, end_seconds=end)


@pytest.mark.parametrize(
    "kind,onsets,code",
    [
        ("EDF+C", ("+0", "+2"), "edf_record_timing_invalid"),
        ("EDF+D", ("+0", "+0.5"), "edf_record_timing_invalid"),
        ("EDF+D", ("+1", "+2"), "edf_timekeeping_invalid"),
    ],
)
def test_timing_negative_controls(kind, onsets, code):
    with pytest.raises(EdfError, match=f"^{code}$"):
        read_edf(synthetic_edf(kind=kind, onsets=onsets))


@pytest.mark.parametrize(
    "tal",
    [
        b"+0\x14secret\x14\x00",
        b"+0\x150.1\x14\x14\x00",
        b"0\x14\x14\x00",
        b"+0\x14\x14" + b"x" * 252,
        b"+0\x14\x14\x00\x00x",
        b"+0\x14\x14\x00+0\x14\xff\x14\x00",
        b"+0\x14\x14\x00+0\x14secret\x15\x14\x00",
    ],
)
def test_malformed_annotations_never_reflect_text(tal):
    source = synthetic_edf(kind="EDF+C")
    source = source[:776] + tal.ljust(256, b"\0") + source[1032:]
    with pytest.raises(EdfError) as caught:
        read_edf(source)
    assert caught.value.code in (
        "edf_timekeeping_invalid",
        "edf_annotation_invalid",
        "edf_numeric_invalid",
    )
    assert "secret" not in str(caught.value)
    assert caught.value.__context__ is None


def test_unknown_count_exact_byte_budget_and_short_stream_reads():
    source = synthetic_edf(kind="EDF+C", declared=-1)

    class ShortStream(io.BytesIO):
        def read(self, size):
            assert 0 < size <= 65536
            return super().read(min(size, 7))

    stream = ShortStream(b"prefix" + source)
    stream.seek(6)
    result = read_edf(stream, limits=replace(EdfLimits(), max_bytes=len(source)))
    assert result.record_count == 2
    assert stream.tell() == 6
    assert not stream.closed


def test_nonseekable_stream_and_safe_stream_failure():
    class Stream:
        def __init__(self):
            self.inner = io.BytesIO(synthetic_edf())

        def read(self, size):
            return self.inner.read(size)

    assert read_edf(Stream()).record_count == 2

    class Broken:
        def read(self, size):
            raise OSError("SYNTHETIC_PRIVATE_PATH")

    with pytest.raises(EdfError, match="^edf_stream_read_error$") as caught:
        read_edf(Broken())
    assert caught.value.__context__ is None


def test_stream_restored_on_parse_failure():
    stream = io.BytesIO(synthetic_edf()[:-1])
    with pytest.raises(EdfError):
        read_edf(stream)
    assert stream.tell() == 0


def test_zero_duration_discontinuous_point_samples():
    result = read_edf(
        synthetic_edf(
            kind="EDF+D", duration="0", onsets=("+0", "+3"), samples=((-2,), (2,))
        ),
        end_seconds=4,
    )
    assert result.signals[0].sampling_rate_hz is None
    assert result.records[1].signals[0].digital_samples == (2,)
    assert result.gaps[0].end_seconds == 3


def test_sample_outside_declared_digital_range_fails():
    with pytest.raises(EdfError, match="^edf_sample_range_invalid$"):
        read_edf(synthetic_edf(samples=((-3, 0, 1, 2), (2, 1, 0, -2))))


def test_multiple_ordinary_signals_with_different_sample_rates():
    result = read_edf(
        synthetic_edf(kind="EDF+C", other_samples=((-2, 2), (0, 1))),
        start_seconds=0.5,
        end_seconds=1.5,
    )
    assert [(s.label, s.sampling_rate_hz) for s in result.signals] == [
        ("ECG II", 4),
        ("EMG", 2),
    ]
    assert result.records[0].signals[1].first_sample_index == 1
    assert result.records[0].signals[1].digital_samples == (2,)
    assert result.records[0].signals[1].physical_samples == (10,)
    assert result.records[1].signals[1].physical_samples == (0,)


def test_annotation_only_zero_duration_and_default_event_selection():
    source = synthetic_edf(
        kind="EDF+D", annotation_only=True, duration="0", onsets=("+0", "+3")
    )
    result = read_edf(source)
    assert result.signals == ()
    assert result.window_end_seconds == 4
    assert result.report()["annotation_count"] == 4
    source = synthetic_edf(kind="EDF+C")
    tal = b"+0\x14\x14\x00-0.25\x151\x14before\x14\x00+100\x14after\x14\x00"
    source = source[:776] + tal.ljust(256, b"\0") + source[1032:]
    result = read_edf(source)
    assert [a.onset_seconds for a in result.annotations] == [-0.25, 1]
    assert result.window_end_seconds == 2


def test_annotation_timing_without_duration_and_missing_channel():
    source = synthetic_edf(kind="EDF+C")
    tal = b"+0\x14\x14\x00+0.75\x14event\x14\x00"
    source = source[:776] + tal.ljust(256, b"\0") + source[1032:]
    result = read_edf(source, start_seconds=0.75, end_seconds=1)
    assert result.annotations[0].duration_seconds is None
    assert result.annotations[0].count == 1
    with pytest.raises(EdfError, match="^edf_annotation_channel_missing$"):
        read_edf(replace_field(source, 272, 16, "ECG I"))


def test_timing_outside_window_still_validated():
    with pytest.raises(EdfError, match="^edf_record_timing_invalid$"):
        read_edf(
            synthetic_edf(kind="EDF+C", onsets=("+0", "+2")),
            start_seconds=0,
            end_seconds=0.5,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limit_configuration(value):
    with pytest.raises(EdfError, match="^edf_limits_invalid$"):
        read_edf(synthetic_edf(), limits=replace(EdfLimits(), max_records=value))


def test_non_header_free_text_is_withheld_and_bad_stream_contract_is_safe():
    source = synthetic_edf()
    for offset, size in [(272, 80), (352, 8), (392, 80), (480, 32), (192, 44)]:
        source = replace_field(source, offset, size, "PRIVATE"[:size])
    result = read_edf(source)
    assert result.signals[0].physical_dimension == "withheld"
    assert "PRIVATE" not in repr(result) + json.dumps(result.report())

    class BadStream:
        def read(self, size):
            return b"PRIVATE" * size

    with pytest.raises(EdfError, match="^edf_stream_read_error$") as caught:
        read_edf(BadStream())
    assert caught.value.__context__ is None
