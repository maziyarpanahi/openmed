"""Bounded, dependency-free media-type detection for multimodal preflight."""

from __future__ import annotations

import re
from enum import Enum
from typing import Final

__all__ = [
    "MAX_MEDIA_TYPE_PREFIX_BYTES",
    "MediaTypeStatus",
    "detect_media_type",
    "validate_media_type",
]

MAX_MEDIA_TYPE_PREFIX_BYTES: Final[int] = 132

_MEDIA_TYPE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$"
)


class MediaTypeStatus(str, Enum):
    """Categorical result of comparing detected and declared media types."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


def detect_media_type(prefix: bytes | bytearray | memoryview) -> str | None:
    """Detect a supported media type from at most 132 leading bytes.

    The detector returns ``None`` for truncated, ambiguous, and unsupported
    inputs. It does not inspect filenames, decode content, or log input bytes.
    """

    if not isinstance(prefix, (bytes, bytearray, memoryview)):
        raise TypeError("prefix must be bytes-like")

    bounded = bytes(prefix[:MAX_MEDIA_TYPE_PREFIX_BYTES])
    if bounded.startswith(b"%PDF-"):
        return "application/pdf"
    if bounded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if bounded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if bounded.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(bounded) >= 132 and bounded[128:132] == b"DICM":
        return "application/dicom"
    if len(bounded) >= 12 and bounded.startswith(b"RIFF") and bounded[8:12] == b"WAVE":
        return "audio/wav"
    return _detect_bmff_type(bounded)


def _detect_bmff_type(prefix: bytes) -> str | None:
    """Recognize only a complete, bounded first file-type box."""
    if len(prefix) < 16 or prefix[4:8] != b"ftyp":
        return None
    size = int.from_bytes(prefix[:4], "big")
    header = 8
    if size == 1:
        size = int.from_bytes(prefix[8:16], "big")
        header = 16
    if size < header + 8 or size > len(prefix) or (size - header) % 4:
        return None
    brands = {prefix[header : header + 4]}
    brands.update(prefix[i : i + 4] for i in range(header + 8, size, 4))
    # AVIF shares the structural mif1 brand; do not mistake it for HEIC.
    if brands & {b"avif", b"avis"}:
        return None
    image = None
    if brands & {b"hevc", b"hevx"}:
        image = "image/heic-sequence"
    elif b"msf1" in brands:
        image = "image/heif-sequence"
    elif brands & {b"heic", b"heix", b"heim", b"heis"}:
        image = "image/heic"
    elif b"mif1" in brands:
        image = "image/heif"
    audio = "audio/mp4" if b"M4A " in brands else None
    return None if image and audio else image or audio


def validate_media_type(
    prefix: bytes | bytearray | memoryview, declared_media_type: str
) -> MediaTypeStatus:
    """Compare detected and declared media types without exposing input bytes."""

    if (
        type(declared_media_type) is not str
        or declared_media_type != declared_media_type.lower()
        or _MEDIA_TYPE_RE.fullmatch(declared_media_type) is None
    ):
        raise ValueError("declared media type is unsupported")

    detected = detect_media_type(prefix)
    if detected is None:
        return MediaTypeStatus.UNKNOWN
    if detected == declared_media_type:
        return MediaTypeStatus.MATCH
    return MediaTypeStatus.MISMATCH
