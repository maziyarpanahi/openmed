"""Hand-checked, synthetic WFDB fixtures; no datasets, patients or model assets."""

import struct

# Interleaved channels: (-2048, 2047), (-1, 0), (100, -100).
# Format 212: low bytes flank the two high nibbles in the middle byte.
FORMAT_CASES = (
    (
        16,
        bytes.fromhex("00 f8 ff 07 ff ff 00 00 64 00 9c ff"),
        (-2048, -1, 100),
        (2047, 0, -100),
    ),
    (
        212,
        bytes.fromhex("00 78 ff ff 0f 00 64 f0 9c"),
        (-2048, -1, 100),
        (2047, 0, -100),
    ),
    (80, bytes.fromhex("00 ff 7f 80 81 7e"), (-128, -1, 1), (127, 0, -2)),
)


def synthetic_header(
    fmt: int, first: tuple[int, ...], second: tuple[int, ...]
) -> bytes:
    """Declare two calibrated leads and optional PHI-like synthetic metadata."""

    def signed_sum(samples):
        value = sum(samples) & 65535
        return value - 65536 if value >= 32768 else value

    return (
        f"SYNTHETIC_RECORD_NAME 2 250 3 12:00:00 01/01/2000\n"
        f"/synthetic/private/signal.dat {fmt} 200(17)/mV 12 0 {first[0]} {signed_sum(first)} 0 II\n"
        f"/synthetic/private/signal.dat {fmt} 1000/uV 12 23 {second[0]} {signed_sum(second)} 0 V1\n"
        "# SYNTHETIC_COMMENT_NAME age=77 diagnosis=example-only\n"
    ).encode()


def annotation_word(code: int, interval: int) -> bytes:
    """Encode a single MIT annotation word for synthetic tests."""
    return struct.pack("<H", code * 1024 + interval)


SYNTHETIC_ANNOTATIONS = (
    annotation_word(1, 0)
    + annotation_word(63, 24)
    + b"SYNTHETIC_AUXILIARY_TEXT"
    + annotation_word(60, 1)
    + annotation_word(61, 2)
    + annotation_word(62, 0)
    + annotation_word(1, 2)
    + b"\0\0"
)
