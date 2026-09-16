"""Tiny synthetic RF64 envelopes; no audio samples or descriptive metadata.

The ds64 values exercise refusal, not support for decoding RF64. Every fixture
is at most 128 bytes even when it declares a multi-gigabyte chunk.
"""

from __future__ import annotations

import struct

_RF64_HEADER = b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
_FMT = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
_EMPTY_DATA = b"data" + struct.pack("<I", 0xFFFFFFFF)


def _ds64(riff_size: int) -> bytes:
    return b"ds64" + struct.pack("<IQQQI", 28, riff_size, 0, 0, 0)


RF64_CASES: tuple[tuple[str, bytes], ...] = (
    ("missing-ds64", _RF64_HEADER),
    ("truncated-ds64", _RF64_HEADER + b"ds64" + struct.pack("<I", 28) + b"\x00"),
    ("valid-looking-ds64", _RF64_HEADER + _ds64(72) + _FMT + _EMPTY_DATA),
    (
        "duplicate-ds64",
        _RF64_HEADER + _ds64(108) + _ds64(108) + _FMT + _EMPTY_DATA,
    ),
    ("oversized-ds64", _RF64_HEADER + b"ds64" + struct.pack("<I", 0xFFFFFFFF)),
)
