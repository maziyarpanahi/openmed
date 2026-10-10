"""Offline decoding, privacy, resource and transport regression controls."""

from __future__ import annotations

import dataclasses
import io
import json
import struct
import traceback
import tracemalloc

import pytest

from openmed.multimodal.wfdb import WFDB_NOTICE, WfdbError, WfdbLimits, read_wfdb_record
from tests.fixtures.multimodal.wfdb import (
    FORMAT_CASES,
    SYNTHETIC_ANNOTATIONS,
    annotation_word,
    synthetic_header,
)


def one_header(fmt=16, count=3, gain="200", tail="", name="signal.dat"):
    return f"SYNTHETIC_RECORD_NAME 1 250 {count}\n{name} {fmt} {gain}{tail}\n".encode()


def assert_code(code, header, signals, **kwargs):
    with pytest.raises(WfdbError) as caught:
        read_wfdb_record(header, signals, **kwargs)
    assert caught.value.reason_code == code
    assert str(caught.value) == code
    assert caught.value.__context__ is None
    return caught.value


@pytest.mark.parametrize("fmt,payload,first,second", FORMAT_CASES)
@pytest.mark.parametrize("streamed", (False, True))
def test_hand_checked_formats_calibration_and_timing(
    fmt, payload, first, second, streamed
):
    header = synthetic_header(fmt, first, second)
    if streamed:
        header, payload = io.BytesIO(header), io.BytesIO(payload)
    record = read_wfdb_record(header, [payload], annotations=SYNTHETIC_ANNOTATIONS)
    assert record.signals[0].samples == first
    assert record.signals[1].samples == second
    assert [s.gain for s in record.signals] == [200, 1000]
    assert [s.baseline for s in record.signals] == [17, 23]
    assert [s.unit for s in record.signals] == ["mV", "uV"]
    assert [s.lead_label for s in record.signals] == ["II", "V1"]
    assert all(s.checksum_verified for s in record.signals)
    assert record.sample_rate_hz == 250
    assert record.duration_seconds == 0.012
    assert record.start_seconds == 0
    assert record.annotations.total_count == 2
    assert record.annotations.sample_positions == (0, 2)
    assert record.annotations.auxiliary_text_present
    assert record.comments_present and record.record_name_present
    assert record.path_fields_present and record.timing_fields_present
    assert record.notice == WFDB_NOTICE


@pytest.mark.parametrize("fmt,payload,first,second", FORMAT_CASES)
@pytest.mark.parametrize("start,count", ((0, 0), (0, 1), (1, 1), (1, 2), (3, 0)))
def test_window_checksum_and_annotation_positions(
    fmt, payload, first, second, start, count
):
    record = read_wfdb_record(
        synthetic_header(fmt, first, second),
        [payload],
        start_sample=start,
        sample_count=count,
        annotations=SYNTHETIC_ANNOTATIONS,
    )
    assert record.signals[0].samples == first[start : start + count]
    assert record.signals[1].samples == second[start : start + count]
    assert record.start_seconds == start / 250
    assert record.duration_seconds == count / 250
    assert record.annotations.sample_positions == tuple(
        p for p in (0, 2) if start <= p < start + count
    )
    assert all(s.checksum_verified for s in record.signals)


@pytest.mark.parametrize("fmt,payload,first,second", FORMAT_CASES)
def test_truncation_and_corruption_outside_window(fmt, payload, first, second):
    header = synthetic_header(fmt, first, second)
    assert_code("wfdb_signal_truncated", header, [payload[:-1]], sample_count=1)
    # Last-byte corruption is outside the first returned frame.
    corrupt = payload[:-1] + bytes([payload[-1] ^ 1])
    assert_code("wfdb_checksum_mismatch", header, [corrupt], sample_count=1)


@pytest.mark.parametrize(
    "fmt,payload", ((16, b"\0\x80\xff\x7f"), (212, b"\0\x78\xff"), (80, b"\0\xff"))
)
def test_encoding_extrema(fmt, payload):
    expected = (
        (-128, 127) if fmt == 80 else ((-2048, 2047) if fmt == 212 else (-32768, 32767))
    )
    assert (
        read_wfdb_record(one_header(fmt, 2), [payload]).signals[0].samples == expected
    )


@pytest.mark.parametrize("padded", (False, True))
def test_odd_212_single_channel_and_byte_offset(padded):
    # Samples [-1, 1, -2048], last packed sample is padding and not returned.
    payload = bytes.fromhex("ff 0f 01 00 08") + (b"\0" if padded else b"")
    record = read_wfdb_record(
        one_header("212x1:0+4"), [b"PHI!" + payload], start_sample=1
    )
    assert record.signals[0].samples == (1, -2048)


def test_separate_files_and_per_group_interleaving():
    header = b"r 3 62.5/125(0) 2\na 16 200\na 16 200\nb 80 200\n"
    record = read_wfdb_record(header, [struct.pack("<4h", 1, 11, 2, 12), b"\x81\x82"])
    assert [s.samples for s in record.signals] == [(1, 2), (11, 12), (1, 2)]
    assert record.sample_rate_hz == 62.5
    assert record.duration_seconds == 0.032


@pytest.mark.parametrize(
    "gain", ("0", "0.0", "-0", "1e999", "nan", "inf", "200(1234567890123)")
)
def test_invalid_calibration(gain):
    code = (
        "wfdb_gain_invalid"
        if gain in ("0", "0.0", "-0", "1e999")
        else "wfdb_header_invalid"
    )
    assert_code(code, one_header(gain=gain), [b"\0" * 6])


@pytest.mark.parametrize(
    "header,code",
    (
        (b"private/2 1 250 3\nprivate 3\n", "wfdb_multisegment_unsupported"),
        (b"r 33 250 3\n", "wfdb_signal_limit_exceeded"),
        (b"r 1 250 20000001\n", "wfdb_sample_limit_exceeded"),
        (b"r 1 250 0\n", "wfdb_sample_limit_exceeded"),
        (b"r 1\na 16\n", "wfdb_sample_count_required"),
        (b"r 1 0 3\na 16\n", "wfdb_rate_invalid"),
        (b"r 1 1e999 3\na 16\n", "wfdb_rate_invalid"),
        (b"r 1 nan 3\na 16\n", "wfdb_header_invalid"),
        (b"r 1 250 3\na 24\n", "wfdb_format_unsupported"),
        (b"r 1 250 3\na 16x2\n", "wfdb_layout_unsupported"),
        (b"r 1 250 3\na 16:1\n", "wfdb_layout_unsupported"),
        (b"r 1 250 3\na 16 200 16 0 0 0 4\n", "wfdb_layout_unsupported"),
        (b"r 2 250 3\na 16\na 80\n", "wfdb_signal_group_invalid"),
        (b"r 3 250 3\na 16\nb 16\na 16\n", "wfdb_signal_group_invalid"),
        (b"r 2 250 3\na 16\n", "wfdb_header_invalid"),
        (b"r 1 250 3\na 16\nextra line\n", "wfdb_header_invalid"),
        (b"r 1 250 3\na 16 200 16 0 0 65536\n", "wfdb_header_invalid"),
        (b"\xff\n", "wfdb_header_invalid"),
    ),
)
def test_stable_refusal_codes(header, code):
    assert_code(code, header, [b"\0" * 6])


@pytest.mark.parametrize(
    "start,count", ((-1, 1), (4, 0), (True, 1), (0, True), (0, -1), (2, 2), (0.0, 1))
)
def test_invalid_windows(start, count):
    assert_code(
        "wfdb_window_invalid",
        one_header(),
        [b"\0" * 6],
        start_sample=start,
        sample_count=count,
    )


def test_limits_and_source_count():
    assert_code(
        "wfdb_window_limit_exceeded",
        one_header(),
        [b"\0" * 6],
        limits=WfdbLimits(max_window_samples=2),
    )
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header(),
        [b"\0" * 7],
        limits=WfdbLimits(max_file_bytes=6),
    )
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header(count=4),
        [b"\0" * 6],
        limits=WfdbLimits(max_file_bytes=6),
    )
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header("16+100"),
        [b""],
        limits=WfdbLimits(max_file_bytes=99),
    )
    assert_code("wfdb_source_count_invalid", one_header(), [])
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header(),
        [b"\0" * 6],
        limits=WfdbLimits(max_header_bytes=8),
    )
    for value in (0, -1, True, 1.5):
        with pytest.raises(WfdbError, match="wfdb_limits_invalid"):
            WfdbLimits(max_samples=value)


@pytest.mark.parametrize(
    "marker",
    (
        "SYNTHETIC_COMMENT_NAME",
        "SYNTHETIC_RECORD_NAME",
        "/synthetic/private/signal.dat",
        "SYNTHETIC_AUXILIARY_TEXT",
        "患者姓名",
        "patient@example.invalid",
    ),
)
def test_metadata_privacy_sweep(marker, caplog):
    fmt, payload, first, second = FORMAT_CASES[0]
    header = synthetic_header(fmt, first, second)
    # Put each direct-identifier marker in comments, a description and units.
    header += f"# {marker}\n".encode()
    header = header.replace(b"0 II", f"0 {marker}".encode())
    header = header.replace(
        b"200(17)/mV", f"200(17)/{marker.replace(' ', '_')}".encode()
    )
    # A path has '/' which is still a syntactically accepted unit token.
    record = read_wfdb_record(header, [payload], annotations=SYNTHETIC_ANNOTATIONS)
    views = (
        repr(record),
        json.dumps(dataclasses.asdict(record)),
        json.dumps(record.report()),
        caplog.text,
    )
    assert all(marker not in view for view in views)
    assert record.descriptions_withheld and record.units_withheld
    assert record.signals[0].lead_label is None
    assert record.signals[0].unit is None
    error = assert_code("wfdb_signal_truncated", header, [payload[:-1]])
    assert marker not in "".join(traceback.format_exception(error))


def test_comments_can_contain_non_utf8_without_being_decoded():
    record = read_wfdb_record(b"#\xff\n" + one_header() + b"#secret\xff\n", [b"\0" * 6])
    assert record.comments_present


def test_review_confirmation_and_content_free_report():
    record = read_wfdb_record(one_header(), [b"\0" * 6])
    for confirmed in (False, 1, "yes", None):
        with pytest.raises(WfdbError, match="wfdb_reviewer_confirmation_required"):
            record.require_reviewer_confirmation(confirmed=confirmed)
    record.require_reviewer_confirmation(confirmed=True)
    assert record.report()["notice"] == WFDB_NOTICE
    assert record.report()["review_required"] is True
    assert "samples" not in record.report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.sample_count = 10


class ShortReads(io.BytesIO):
    def read(self, size=-1):
        assert size > 0
        return super().read(min(size, 3))


class ForwardOnly:
    def __init__(self, payload):
        self.inner = ShortReads(payload)

    def seekable(self):
        return False

    def read(self, size):
        return self.inner.read(size)


@pytest.mark.parametrize("factory", (ShortReads, ForwardOnly))
def test_short_reads_and_forward_streams(factory):
    fmt, payload, first, second = FORMAT_CASES[1]
    record = read_wfdb_record(
        factory(synthetic_header(fmt, first, second)),
        [factory(payload)],
        annotations=factory(SYNTHETIC_ANNOTATIONS),
    )
    assert record.signals[0].samples == first
    assert record.annotations.sample_positions == (0, 2)


@pytest.mark.parametrize("failure", (False, True))
def test_restore_nonzero_positions_on_success_and_failure(failure):
    header = ShortReads(b"prefix" + one_header())
    signal = ShortReads(b"prefix" + b"\0" * (5 if failure else 6))
    header.seek(6)
    signal.seek(6)
    if failure:
        assert_code("wfdb_signal_truncated", header, [signal])
    else:
        read_wfdb_record(header, [signal])
    assert header.tell() == signal.tell() == 6
    assert not header.closed and not signal.closed


def test_stream_exception_does_not_retain_private_context():
    class Broken(ForwardOnly):
        def read(self, size):
            raise OSError("/SYNTHETIC_PRIVATE_PATH patient@example.invalid")

    error = assert_code("wfdb_stream_read_error", one_header(), [Broken(b"")])
    assert "SYNTHETIC_PRIVATE_PATH" not in "".join(traceback.format_exception(error))


def test_stream_contract_and_forward_trailing_limit():
    class OverRead(ForwardOnly):
        def read(self, size):
            return b"x" * (size + 1)

    assert_code("wfdb_stream_contract_error", OverRead(b""), [b"\0" * 6])
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header(),
        [ForwardOnly(b"\0" * 7)],
        limits=WfdbLimits(max_file_bytes=6),
    )
    assert_code("wfdb_signal_truncated", one_header(), [ForwardOnly(b"\0" * 5)])


def test_annotation_skip_aux_padding_and_window_filter():
    # PDP-11: high word first, little endian within each word; delta = 65536.
    annotation = (
        annotation_word(1, 0)
        + annotation_word(59, 0)
        + b"\x01\0\0\0"
        + annotation_word(1, 2)
        + annotation_word(63, 3)
        + b"PHI\0"
        + b"\0\0"
    )
    record = read_wfdb_record(
        one_header(80, 65540),
        [b"\x80" * 65540],
        annotations=annotation,
        start_sample=65538,
        sample_count=1,
    )
    assert record.annotations.total_count == 2
    assert record.annotations.sample_positions == (65538,)
    assert record.annotations.auxiliary_text_present


@pytest.mark.parametrize(
    "payload,code",
    (
        (b"", "wfdb_annotation_truncated"),
        (b"\x01", "wfdb_annotation_truncated"),
        (annotation_word(59, 0) + b"\0", "wfdb_annotation_truncated"),
        (annotation_word(59, 1), "wfdb_annotation_invalid"),
        (annotation_word(63, 1), "wfdb_annotation_invalid"),
        (
            annotation_word(1, 0) + annotation_word(63, 3) + b"PHI",
            "wfdb_annotation_truncated",
        ),
        (annotation_word(1, 3) + b"\0\0", "wfdb_annotation_position_invalid"),
        (annotation_word(50, 0), "wfdb_annotation_format_unsupported"),
    ),
)
def test_annotation_negative_controls(payload, code):
    assert_code(code, one_header(), [b"\0" * 6], annotations=payload)


def test_annotation_budgets():
    annotations = annotation_word(1, 0) * 2 + b"\0\0"
    assert_code(
        "wfdb_annotation_limit_exceeded",
        one_header(),
        [b"\0" * 6],
        annotations=annotations,
        limits=WfdbLimits(max_annotations=1),
    )
    assert_code(
        "wfdb_file_limit_exceeded",
        one_header(),
        [b"\0" * 6],
        annotations=annotations,
        limits=WfdbLimits(max_annotation_bytes=5),
    )


def test_long_record_allocates_window_and_fixed_scratch_only():
    class GeneratedSignal:
        def __init__(self, length):
            self.remaining = length
            self.max_request = 0

        def seekable(self):
            return False

        def read(self, size):
            assert 0 < size <= 8192
            self.max_request = max(self.max_request, size)
            count = min(size, self.remaining)
            self.remaining -= count
            return b"\0" * count

    def peak_for(total):
        source = GeneratedSignal(total * 2)
        tracemalloc.start()
        try:
            record = read_wfdb_record(
                one_header(count=total),
                [source],
                start_sample=total - 10,
                sample_count=10,
            )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert record.signals[0].samples == (0,) * 10
        assert source.remaining == 0
        return peak

    short_peak = peak_for(10_000)
    long_peak = peak_for(300_000)
    # This bound rejects buffering even a single full long signal payload.
    assert long_peak < 150_000
    assert long_peak < short_peak + 50_000


def test_failed_stream_length_probe_restores_before_refusal():
    class BrokenLength(io.BytesIO):
        def tell(self):
            position = super().tell()
            if position == 6:
                raise OSError("/SYNTHETIC_PRIVATE_PATH")
            return position

    signal = BrokenLength(b"\0" * 6)
    signal.seek(1)
    assert_code("wfdb_stream_contract_error", one_header(count=2), [signal])
    assert signal.tell() == 1


def test_signed_checksum_wraparound():
    header = one_header(count=2, tail=" 16 0 32767 -2 0 II")
    record = read_wfdb_record(header, [struct.pack("<2h", 32767, 32767)])
    assert record.signals[0].checksum_verified
    assert record.signals[0].samples == (32767, 32767)


def test_truncated_odd_packed_final_sample():
    assert_code(
        "wfdb_signal_truncated", one_header(212), [bytes.fromhex("ff 0f 01 00")]
    )
