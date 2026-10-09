"""Tiny synthetic media declarations for pre-decode admission tests."""

from __future__ import annotations

import struct
import zlib


def png(width: int = 8, height: int = 8, *, frames: int | None = None) -> bytes:
    """Build CRC-correct PNG headers, optionally declaring an APNG frame count."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    payload = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    if frames is not None:
        payload += chunk(b"acTL", struct.pack(">II", frames, 0))
    return payload + chunk(b"IEND", b"")


def tiff(
    dimensions: tuple[tuple[int, int], ...], *, order: str = "<", cycle: bool = False
) -> bytes:
    """Build classic TIFF directories with numeric geometry and no pixels."""
    directory_size = 2 + 3 * 12 + 4
    payload = bytearray(
        (b"II" if order == "<" else b"MM") + struct.pack(order + "HI", 42, 8)
    )
    for index, (width, height) in enumerate(dimensions):
        payload += struct.pack(order + "H", 3)
        for tag, value in ((256, width), (257, height), (262, 1)):
            payload += struct.pack(order + "HHII", tag, 4, 1, value)
        following = (
            8 + (index + 1) * directory_size if index + 1 < len(dimensions) else 0
        )
        if cycle and index + 1 == len(dimensions):
            following = 8
        payload += struct.pack(order + "I", following)
    return bytes(payload)


def pdf(
    pages: int = 1, *, width: int = 72, height: int = 72, declared: int | None = None
) -> bytes:
    """Build a blank, valid PDF with independently selectable count and boxes."""
    kids = " ".join(f"{index + 3} 0 R" for index in range(pages))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages if declared is None else declared} >>",
        *[
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] >>"
            for _ in range(pages)
        ],
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n{obj}\nendobj\n".encode("ascii"))
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def dicom(width: int = 8, height: int = 8, *, frames: int = 1) -> bytes:
    """Build a Part 10 DICOM with only geometry, encoding, and zero-valued pixels."""

    def element(group: int, tag: int, vr: bytes, value: bytes) -> bytes:
        if len(value) % 2:
            value += b"\0" if vr == b"UI" else b" "
        prefix = struct.pack("<HH", group, tag) + vr
        if vr in {b"OB", b"OW"}:
            return prefix + b"\0\0" + struct.pack("<I", len(value)) + value
        return prefix + struct.pack("<H", len(value)) + value

    output = b"\0" * 128 + b"DICM"
    output += element(2, 0x10, b"UI", b"1.2.840.10008.1.2.1")
    output += element(0x28, 2, b"US", struct.pack("<H", 1))
    output += element(0x28, 4, b"CS", b"MONOCHROME2")
    output += element(0x28, 8, b"IS", str(frames).encode("ascii"))
    for tag, value in (
        (0x10, height),
        (0x11, width),
        (0x100, 8),
        (0x101, 8),
        (0x102, 7),
        (0x103, 0),
    ):
        output += element(0x28, tag, b"US", struct.pack("<H", value))
    # The payload is deliberately tiny even for enormous declared dimensions.
    output += element(0x7FE0, 0x10, b"OB", b"\0" * 64)
    return output
