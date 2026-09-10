"""Tests for bounded, privacy-safe WAV metadata parsing."""

from __future__ import annotations

import io
import struct

import pytest

from openmed.multimodal.wav_metadata import (
    DEFAULT_MAX_WAV_HEADER_BYTES,
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    WavMetadataError,
    read_wav_metadata,
)


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) & 1 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _fmt(
    *,
    format_code: int = WAVE_FORMAT_PCM,
    channels: int = 1,
    sample_rate: int = 8000,
    bit_depth: int = 16,
    byte_rate: int | None = None,
    block_align: int | None = None,
) -> bytes:
    sample_bytes = (bit_depth + 7) // 8
    align = channels * sample_bytes if block_align is None else block_align
    rate = sample_rate * align if byte_rate is None else byte_rate
    return struct.pack(
        "<HHIIHH", format_code, channels, sample_rate, rate, align, bit_depth
    )


def _wav(*chunks: bytes, riff_size: int | None = None) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    declared_size = len(body) if riff_size is None else riff_size
    return b"RIFF" + struct.pack("<I", declared_size) + body


@pytest.mark.parametrize(
    ("fmt", "samples", "expected"),
    [
        (
            _fmt(channels=1, sample_rate=8000, bit_depth=16),
            b"\x00" * 8,
            (WAVE_FORMAT_PCM, 1, 8000, 16, 8, 4, 0.0005),
        ),
        (
            _fmt(channels=2, sample_rate=44100, bit_depth=24),
            b"\x00" * 12,
            (WAVE_FORMAT_PCM, 2, 44100, 24, 12, 2, 2 / 44100),
        ),
        (
            _fmt(
                format_code=WAVE_FORMAT_IEEE_FLOAT,
                channels=1,
                sample_rate=48000,
                bit_depth=32,
            ),
            b"\x00" * 8,
            (WAVE_FORMAT_IEEE_FLOAT, 1, 48000, 32, 8, 2, 2 / 48000),
        ),
    ],
)
def test_reads_hand_checked_pcm_and_float_metadata(fmt, samples, expected):
    result = read_wav_metadata(_wav(_chunk(b"fmt ", fmt), _chunk(b"data", samples)))

    assert (
        result.format_code,
        result.channels,
        result.sample_rate_hz,
        result.bit_depth,
        result.data_byte_count,
        result.frame_count,
        result.duration_seconds,
    ) == expected


def test_skips_extra_chunks_and_their_odd_byte_padding() -> None:
    payload = _wav(
        _chunk(b"JUNK", b"abc"),
        _chunk(b"fmt ", _fmt()),
        _chunk(b"LIST", b"12345"),
        _chunk(b"data", b"\x00" * 4),
    )

    result = read_wav_metadata(payload)

    assert result.frame_count == 2
    assert result.data_byte_count == 4


def test_empty_data_chunk_has_zero_frames_and_duration() -> None:
    result = read_wav_metadata(_wav(_chunk(b"fmt ", _fmt()), _chunk(b"data", b"")))

    assert result.data_byte_count == 0
    assert result.frame_count == 0
    assert result.duration_seconds == 0.0


def test_parser_stops_at_data_header_without_reading_samples() -> None:
    samples = b"do-not-read-these-audio-samples"
    payload = _wav(_chunk(b"fmt ", _fmt(bit_depth=8)), _chunk(b"data", samples))
    data_start = payload.index(b"data") + 8

    class NonSeekableStream:
        def __init__(self) -> None:
            self.stream = io.BytesIO(payload)
            self.closed = False

        def seekable(self) -> bool:
            return False

        def read(self, size: int) -> bytes:
            return self.stream.read(size)

    stream = NonSeekableStream()
    result = read_wav_metadata(stream)

    assert result.data_byte_count == len(samples)
    assert stream.stream.tell() == data_start
    assert not stream.closed


def test_seekable_stream_is_restored_and_never_closed() -> None:
    wav = _wav(_chunk(b"fmt ", _fmt()), _chunk(b"data", b"\x00" * 4))
    stream = io.BytesIO(b"prefix" + wav)
    stream.seek(len(b"prefix"))

    result = read_wav_metadata(stream)

    assert result.frame_count == 2
    assert stream.tell() == len(b"prefix")
    assert not stream.closed


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (b"RIFX\x00\x00\x00\x04WAVE", "wav_endianness_unsupported"),
        (b"not a wave!!", "wav_signature_invalid"),
        (b"RIFF\x03\x00\x00\x00WAVE", "wav_riff_size_invalid"),
        (
            b"RIFF\xff\xff\xff\xffWAVEJUNK\xff\xff\xff\xff",
            "wav_chunk_size_invalid",
        ),
        (_wav(_chunk(b"fmt ", b"\x00" * 15)), "wav_fmt_chunk_invalid"),
        (_wav(_chunk(b"JUNK", b"abc")), "wav_fmt_chunk_missing"),
        (_wav(_chunk(b"fmt ", _fmt())), "wav_data_chunk_missing"),
        (
            _wav(_chunk(b"data", b""), _chunk(b"fmt ", _fmt())),
            "wav_fmt_chunk_missing",
        ),
        (
            _wav(_chunk(b"fmt ", _fmt(format_code=6)), _chunk(b"data", b"")),
            "wav_format_unsupported",
        ),
        (
            _wav(
                _chunk(
                    b"fmt ",
                    _fmt(format_code=WAVE_FORMAT_IEEE_FLOAT, bit_depth=16),
                ),
                _chunk(b"data", b""),
            ),
            "wav_float_bit_depth_invalid",
        ),
        (
            _wav(
                _chunk(b"fmt ", _fmt(block_align=3)),
                _chunk(b"data", b""),
            ),
            "wav_format_consistency_invalid",
        ),
        (
            _wav(
                _chunk(b"fmt ", _fmt(byte_rate=1)),
                _chunk(b"data", b""),
            ),
            "wav_format_consistency_invalid",
        ),
        (
            _wav(_chunk(b"fmt ", _fmt()), _chunk(b"data", b"\x00")),
            "wav_data_size_invalid",
        ),
    ],
)
def test_malformed_or_unsupported_headers_fail_closed(payload, category):
    with pytest.raises(WavMetadataError) as raised:
        read_wav_metadata(payload)

    assert raised.value.category == category
    assert str(raised.value) == category


def test_truncated_header_fails_closed() -> None:
    fmt_header = b"fmt " + struct.pack("<I", 16) + b"\x00" * 8
    payload = _wav(fmt_header, riff_size=4 + 8 + 16)

    with pytest.raises(WavMetadataError, match="wav_header_truncated"):
        read_wav_metadata(payload)


def test_configured_header_limit_is_enforced_before_an_oversized_read() -> None:
    payload = _wav(
        _chunk(b"JUNK", b"x" * 32),
        _chunk(b"fmt ", _fmt()),
        _chunk(b"data", b""),
    )

    with pytest.raises(WavMetadataError, match="wav_header_limit_exceeded"):
        read_wav_metadata(payload, max_header_bytes=20)


def test_stream_errors_are_value_free_and_do_not_close_the_stream() -> None:
    sentinel = "/private/patient-name.wav?token=secret"

    class BrokenStream:
        closed = False

        def seekable(self) -> bool:
            return False

        def read(self, size: int) -> bytes:
            raise OSError(sentinel)

    stream = BrokenStream()
    with pytest.raises(WavMetadataError) as raised:
        read_wav_metadata(stream)

    assert str(raised.value) == "wav_stream_read_error"
    assert sentinel not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not stream.closed


@pytest.mark.parametrize("max_header_bytes", [0, -1, True, 1.5, None])
def test_invalid_header_limits_are_rejected(max_header_bytes) -> None:
    with pytest.raises(ValueError, match="max_header_bytes"):
        read_wav_metadata(
            b"RIFF\x04\x00\x00\x00WAVE",
            max_header_bytes=max_header_bytes,
        )


def test_default_header_limit_is_bounded() -> None:
    assert DEFAULT_MAX_WAV_HEADER_BYTES == 64 * 1024
