"""Minimal synthetic encoded media and deterministic malformed mutations.

No patient data, downloaded corpora, model weights or exploit payloads. Native
crash/resource fault injection lives separately in the private worker controls.
"""

from __future__ import annotations

import io
import struct
import wave
import zlib


def tiff(*, width: int = 2, height: int = 2, recursive: bool = False) -> bytes:
    """Encode one classic grayscale TIFF IFD and four synthetic pixels."""
    tags = (
        (256, 4, width),
        (257, 4, height),
        (258, 3, 8),
        (259, 3, 1),
        (262, 3, 1),
        (273, 4, 122),
        (277, 3, 1),
        (278, 4, height),
        (279, 4, 4),
    )
    entries = b"".join(
        struct.pack("<HHII", tag, typ, 1, value) for tag, typ, value in tags
    )
    return (
        b"II*\0\x08\0\0\0"
        + struct.pack("<H", len(tags))
        + entries
        + struct.pack("<I", 8 if recursive else 0)
        + b"\x00\x40\x80\xff"
    )


def wav() -> bytes:
    """Encode a short silent mono PCM control entirely in memory."""
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0" * 32)
    return stream.getvalue()


def dicom(*, oversized: bool = False) -> bytes:
    """Encode a native explicit-little-endian two-by-two secondary capture."""

    def item(group: int, element: int, vr: bytes, data: bytes) -> bytes:
        data += b"\0" * (len(data) % 2)
        prefix = struct.pack("<HH", group, element) + vr
        if vr in (b"OB", b"OW"):
            return prefix + b"\0\0" + struct.pack("<I", len(data)) + data
        return prefix + struct.pack("<H", len(data)) + data

    meta = item(2, 0x10, b"UI", b"1.2.840.10008.1.2.1")
    body = item(0x28, 2, b"US", struct.pack("<H", 1))
    body += item(0x28, 4, b"CS", b"MONOCHROME2")
    for tag, value in (
        (0x10, 65535 if oversized else 2),
        (0x11, 65535 if oversized else 2),
        (0x100, 8),
        (0x101, 8),
        (0x102, 7),
        (0x103, 0),
    ):
        body += item(0x28, tag, b"US", struct.pack("<H", value))
    body += item(0x7FE0, 0x10, b"OB", b"\x00\x40\x80\xff")
    return b"\0" * 128 + b"DICM" + meta + body


def pdf(*, recursive: bool = False, compressed_bytes: int = 0) -> bytes:
    """Encode a blank one-page PDF, optional cyclic tree or small compressed control."""
    stream = zlib.compress(b" " * compressed_bytes) if compressed_bytes else b""
    content = (
        b"<< /Length "
        + str(len(stream)).encode()
        + (b" /Filter /FlateDecode" if compressed_bytes else b"")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + (b"2" if recursive else b"3")
        + b" 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 16 16] /Resources << >> /Contents 4 0 R >>",
        content,
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(str(number).encode() + b" 0 obj\n" + obj + b"\nendobj\n")
    start = len(output)
    output.extend(b"xref\n0 5\n0000000000 65535 f \n")
    output.extend(
        b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    )
    output.extend(
        b"trailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n"
        + str(start).encode()
        + b"\n%%EOF\n"
    )
    return bytes(output)


def video(*, oversized: bool = False) -> bytes:
    """Encode one raw two-by-two YUV4MPEG frame, without a codec dependency."""
    geometry = b"W999999 H999999" if oversized else b"W2 H2"
    return (
        b"YUV4MPEG2 "
        + geometry
        + b" F1:1 Ip A1:1 C420jpeg\nFRAME\n"
        + b"\x10" * 4
        + b"\x80" * 2
    )


def controls() -> dict[str, bytes]:
    """Return genuine encoded positive controls for all six declared kinds."""
    return {
        "pdf": pdf(),
        "tiff": tiff(),
        "dicom": dicom(),
        "wsi_tiff": tiff(),
        "wav": wav(),
        "video": video(),
    }
