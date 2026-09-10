"""Bounded, dependency-free WAV metadata parsing for multimodal preflight."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Callable, Final

__all__ = [
    "DEFAULT_MAX_WAV_HEADER_BYTES",
    "WAVE_FORMAT_IEEE_FLOAT",
    "WAVE_FORMAT_PCM",
    "WavMetadata",
    "WavMetadataError",
    "read_wav_metadata",
]

DEFAULT_MAX_WAV_HEADER_BYTES: Final[int] = 64 * 1024
WAVE_FORMAT_PCM: Final[int] = 0x0001
WAVE_FORMAT_IEEE_FLOAT: Final[int] = 0x0003

_READ_CHUNK_BYTES: Final[int] = 8192
_UINT16_MAX: Final[int] = (1 << 16) - 1
_UINT32_MAX: Final[int] = (1 << 32) - 1


class WavMetadataError(ValueError):
    """Value-free failure raised for malformed or unsupported WAV metadata."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class WavMetadata:
    """Audio metadata read without decoding or retaining sample bytes."""

    format_code: int
    channels: int
    sample_rate_hz: int
    bit_depth: int
    data_byte_count: int
    frame_count: int
    duration_seconds: float


def read_wav_metadata(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_WAV_HEADER_BYTES,
) -> WavMetadata:
    """Read PCM or IEEE-float metadata from a bounded RIFF/WAVE header.

    Streams are read from their current position. Seekable streams are restored
    before return or failure, and caller-owned streams are never closed.
    Parsing stops at the ``data`` chunk header without reading audio samples.
    """

    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")

    if isinstance(source, bytes):
        return _parse_wav(_BoundedReader(source, max_header_bytes))

    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")

    initial_position = _stream_position(source)
    try:
        return _parse_wav(_BoundedReader(read, max_header_bytes))
    finally:
        if initial_position is not None:
            _restore_position(source, initial_position)


class _BoundedReader:
    def __init__(
        self,
        source: bytes | Callable[[int], bytes],
        limit: int,
    ) -> None:
        self._buffer = memoryview(source) if isinstance(source, bytes) else None
        self._read = source if callable(source) else None
        self._limit = limit
        self.offset = 0

    def read_exact(self, size: int) -> bytes:
        self._check_limit(size)
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise WavMetadataError("wav_header_truncated")
            chunk = bytes(self._buffer[self.offset : end])
            self.offset = end
            return chunk

        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = _read_stream_chunk(self._read, remaining)
            if not chunk:
                raise WavMetadataError("wav_header_truncated")
            chunks.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def skip(self, size: int) -> None:
        self._check_limit(size)
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise WavMetadataError("wav_header_truncated")
            self.offset = end
            return

        remaining = size
        while remaining:
            request_bytes = min(remaining, _READ_CHUNK_BYTES)
            chunk = _read_stream_chunk(self._read, request_bytes)
            if not chunk:
                raise WavMetadataError("wav_header_truncated")
            self.offset += len(chunk)
            remaining -= len(chunk)

    def _check_limit(self, size: int) -> None:
        if size > self._limit - self.offset:
            raise WavMetadataError("wav_header_limit_exceeded")


def _parse_wav(reader: _BoundedReader) -> WavMetadata:
    riff_header = reader.read_exact(12)
    if riff_header[:4] == b"RIFX":
        raise WavMetadataError("wav_endianness_unsupported")
    if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
        raise WavMetadataError("wav_signature_invalid")

    riff_size = struct.unpack_from("<I", riff_header, 4)[0]
    if riff_size < 4:
        raise WavMetadataError("wav_riff_size_invalid")
    riff_end = riff_size + 8

    audio_format: tuple[int, int, int, int, int] | None = None
    while reader.offset < riff_end:
        if riff_end - reader.offset < 8:
            raise WavMetadataError("wav_chunk_size_invalid")

        chunk_header = reader.read_exact(8)
        chunk_id = chunk_header[:4]
        chunk_size = struct.unpack_from("<I", chunk_header, 4)[0]
        padded_size = chunk_size + (chunk_size & 1)
        if padded_size > riff_end - reader.offset:
            raise WavMetadataError("wav_chunk_size_invalid")

        if chunk_id == b"fmt ":
            if audio_format is not None:
                raise WavMetadataError("wav_fmt_chunk_duplicate")
            if chunk_size < 16:
                raise WavMetadataError("wav_fmt_chunk_invalid")
            audio_format = _parse_format(reader.read_exact(16))
            reader.skip(padded_size - 16)
            continue

        if chunk_id == b"data":
            if audio_format is None:
                raise WavMetadataError("wav_fmt_chunk_missing")
            return _build_metadata(audio_format, chunk_size)

        reader.skip(padded_size)

    if audio_format is None:
        raise WavMetadataError("wav_fmt_chunk_missing")
    raise WavMetadataError("wav_data_chunk_missing")


def _parse_format(payload: bytes) -> tuple[int, int, int, int, int]:
    format_code, channels, sample_rate, byte_rate, block_align, bit_depth = (
        struct.unpack("<HHIIHH", payload)
    )
    if format_code not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise WavMetadataError("wav_format_unsupported")
    if channels == 0 or sample_rate == 0 or block_align == 0 or bit_depth == 0:
        raise WavMetadataError("wav_format_values_invalid")
    if format_code == WAVE_FORMAT_IEEE_FLOAT and bit_depth not in (32, 64):
        raise WavMetadataError("wav_float_bit_depth_invalid")

    bytes_per_sample = (bit_depth + 7) // 8
    expected_block_align = channels * bytes_per_sample
    expected_byte_rate = sample_rate * expected_block_align
    if (
        expected_block_align > _UINT16_MAX
        or expected_byte_rate > _UINT32_MAX
        or block_align != expected_block_align
        or byte_rate != expected_byte_rate
    ):
        raise WavMetadataError("wav_format_consistency_invalid")
    return format_code, channels, sample_rate, block_align, bit_depth


def _build_metadata(
    audio_format: tuple[int, int, int, int, int], data_byte_count: int
) -> WavMetadata:
    format_code, channels, sample_rate, block_align, bit_depth = audio_format
    if data_byte_count % block_align:
        raise WavMetadataError("wav_data_size_invalid")
    frame_count = data_byte_count // block_align
    return WavMetadata(
        format_code=format_code,
        channels=channels,
        sample_rate_hz=sample_rate,
        bit_depth=bit_depth,
        data_byte_count=data_byte_count,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


def _read_stream_chunk(
    read: Callable[[int], bytes] | None, request_bytes: int
) -> bytes:
    if read is None:
        raise WavMetadataError("wav_stream_contract_error")
    try:
        chunk = read(request_bytes)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= request_bytes:
            return chunk
        raise WavMetadataError("wav_stream_contract_error")
    raise WavMetadataError("wav_stream_read_error")


def _stream_position(stream: BinaryIO) -> int | None:
    seekable = getattr(stream, "seekable", None)
    if not callable(seekable):
        return None
    try:
        if not seekable():
            return None
        position = stream.tell()
    except Exception:
        pass
    else:
        if type(position) is int and position >= 0:
            return position
    raise WavMetadataError("wav_stream_position_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise WavMetadataError("wav_stream_restore_error")
