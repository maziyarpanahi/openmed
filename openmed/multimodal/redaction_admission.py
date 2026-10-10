"""Bounded admission for the existing Python media redaction handlers.

This is resource admission, not decoder isolation or evidence of PHI absence.
Only controlled codes and numeric findings cross its error boundary.
"""

from __future__ import annotations

import math
import struct
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

from .asset_limits import MOBILE_V1, LimitProfile, evaluate_asset_limits
from .image_header import ImageHeaderError, read_image_header
from .media_type import MAX_MEDIA_TYPE_PREFIX_BYTES, detect_media_type
from .pdf_geometry import PdfGeometryStatus, read_pdf_geometry
from .tiff_metadata import (
    DEFAULT_MAX_IFD_ENTRIES,
    TiffMetadataError,
    read_tiff_metadata,
)

_HEADER_BYTES = 64 * 1024
_HEADER_ITEMS = 4096


class RedactionAdmissionError(ValueError):
    """A refusal containing controlled reason/field codes and numbers only.

    Attributes:
        reason_code: Preflight reason or bounded-reader category.
        field_name: Limit field code, or ``None`` for structural failures.
        limit: Numeric ceiling, when available.
        observed: Numeric observation, when available.
    """

    def __init__(
        self,
        reason_code: str,
        field_name: str | None = None,
        limit: int | float | None = None,
        observed: int | float | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.field_name = field_name
        self.limit = limit
        self.observed = observed
        super().__init__(reason_code, field_name, limit, observed)


def redaction_limit_profile(policy: Any = None) -> LimitProfile:
    """Resolve ``asset_limit_profile`` from a mapping or policy object.

    The conservative mobile profile is the default. Overrides must be supplied
    as a validated LimitProfile through policy; there is no bypass switch.
    """
    profile = (
        policy.get("asset_limit_profile", MOBILE_V1)
        if isinstance(policy, Mapping)
        else getattr(policy, "asset_limit_profile", MOBILE_V1)
    )
    if not isinstance(profile, LimitProfile):
        raise RedactionAdmissionError("limit_profile_invalid")
    return profile


def check_limit(profile: LimitProfile, field: str, observed: int | float) -> None:
    """Reject a numeric observation above an inclusive policy ceiling."""
    limit = profile.limit_for(field)
    if observed > limit:
        raise RedactionAdmissionError("limit_exceeded", field, limit, observed)


def check_geometry(
    profile: LimitProfile, width: int, height: int, *, frames: int = 1
) -> None:
    """Evaluate declared or runtime raster geometry using existing limit rules."""
    try:
        findings = evaluate_asset_limits(
            profile,
            {"byte_size": 1, "width": width, "height": height, "frames": frames},
            "dicom",
        )
    except ValueError:
        raise RedactionAdmissionError("metadata_invalid") from None
    if findings:
        finding = findings[0]
        raise RedactionAdmissionError(
            finding.reason_code, finding.field_name, finding.limit, finding.observed
        )


@contextmanager
def _source(source: Any, profile: LimitProfile, media_types: set[str]) -> Iterator[Any]:
    owned = isinstance(source, (str, Path, bytes))
    stream = None
    position = None
    try:
        stream = (
            BytesIO(source)
            if isinstance(source, bytes)
            else open(source, "rb")
            if isinstance(source, (str, Path))
            else source
        )
        position = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        check_limit(profile, "byte_size", size)
        stream.seek(0)
        detected = detect_media_type(stream.read(MAX_MEDIA_TYPE_PREFIX_BYTES))
        if detected is None:
            raise RedactionAdmissionError("unknown")
        if detected not in media_types:
            raise RedactionAdmissionError("mismatch")
        stream.seek(0)
        yield stream, detected
    except RedactionAdmissionError:
        raise
    except Exception:
        raise RedactionAdmissionError("preflight_source_read_error") from None
    finally:
        if stream is not None:
            if owned:
                stream.close()
            elif position is not None:
                try:
                    stream.seek(position)
                except Exception:
                    raise RedactionAdmissionError(
                        "preflight_source_restore_error"
                    ) from None


class _HeaderStream:
    """Seekable metadata reader with a cumulative read budget and offset bound."""

    def __init__(self, stream: Any, size: int) -> None:
        self.stream = stream
        self.size = size
        self.remaining = _HEADER_BYTES

    def read(self, size: int = -1) -> bytes:
        # Reject unbounded reads (including deflated DICOM dataset expansion).
        if size < 0 or size > self.remaining:
            raise RedactionAdmissionError("header_byte_limit", limit=_HEADER_BYTES)
        self.remaining -= size
        return self.stream.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset if whence == 0 else self.tell() + offset
        if whence == 2:
            target = self.size + offset
        if not 0 <= target <= self.size:
            raise RedactionAdmissionError("header_offset_invalid")
        return self.stream.seek(target)

    def tell(self) -> int:
        return self.stream.tell()

    def exact(self, size: int) -> bytes:
        value = self.read(size)
        if len(value) != size:
            raise RedactionAdmissionError("header_truncated")
        return value


def admit_image(source: Any, profile: LimitProfile) -> int:
    """Check image geometry and frame declarations without decoding pixels."""
    with _source(source, profile, {"image/png", "image/jpeg", "image/tiff"}) as pair:
        stream, media_type = pair
        try:
            if media_type == "image/tiff":
                return _tiff_frames(stream, profile)
            if media_type == "image/jpeg":
                return _jpeg_frames(stream, profile)
            header = read_image_header(stream, max_pixels=profile.max_pixels)
            frames = _png_frames(stream, profile)
            check_geometry(profile, header.width, header.height, frames=frames)
            return frames
        except (ImageHeaderError, TiffMetadataError) as exc:
            raise RedactionAdmissionError(exc.category) from None


def _jpeg_frames(stream: Any, profile: LimitProfile) -> int:
    stream.seek(0, 2)
    reader = _HeaderStream(stream, stream.tell())
    reader.seek(2)
    offsets = [0]
    found_index = False
    # MPF is a TIFF-formatted numeric index inside APP2, before JPEG scan data.
    for _ in range(_HEADER_ITEMS):
        if reader.exact(1) != b"\xff":
            raise RedactionAdmissionError("jpeg_marker_unexpected")
        marker = reader.exact(1)
        while marker == b"\xff":
            marker = reader.exact(1)
        if marker in {b"\xda", b"\xd9"}:
            break
        if marker == b"\x01" or 0xD0 <= marker[0] <= 0xD7:
            continue
        length = struct.unpack(">H", reader.exact(2))[0]
        if length < 2:
            raise RedactionAdmissionError("jpeg_segment_length_invalid")
        end = reader.tell() + length - 2
        if marker == b"\xe2" and length >= 6:
            signature = reader.exact(4)
            if signature == b"MPF\0":
                if found_index:
                    raise RedactionAdmissionError("metadata_invalid")
                found_index = True
                base = reader.tell()
                index = reader.exact(length - 6)
                offsets = _mpf_offsets(index, base, profile)
        reader.seek(end)
    else:
        raise RedactionAdmissionError("header_item_limit", limit=_HEADER_ITEMS)
    total_pixels = 0
    for offset in offsets:
        reader.seek(offset)
        header = read_image_header(reader, max_pixels=profile.max_pixels)
        check_geometry(profile, header.width, header.height)
        total_pixels += header.width * header.height
        check_limit(profile, "total_pixels", total_pixels)
    return len(offsets)


def _mpf_offsets(index: bytes, base: int, profile: LimitProfile) -> list[int]:
    if len(index) < 8 or index[:4] not in {b"II*\0", b"MM\0*"}:
        raise RedactionAdmissionError("metadata_invalid")
    order = "<" if index[:2] == b"II" else ">"
    try:
        offset = struct.unpack_from(order + "I", index, 4)[0]
        entries = struct.unpack_from(order + "H", index, offset)[0]
        if offset < 8 or entries > DEFAULT_MAX_IFD_ENTRIES:
            raise RedactionAdmissionError("metadata_invalid")
        frames = None
        table_offset = table_bytes = None
        for number in range(entries):
            tag, kind, count, value = struct.unpack_from(
                order + "HHII", index, offset + 2 + number * 12
            )
            if tag == 0xB001:
                if frames is not None or kind != 4 or count != 1 or value == 0:
                    raise RedactionAdmissionError("metadata_invalid")
                check_limit(profile, "frames", value)
                frames = value
            elif tag == 0xB002:
                if table_offset is not None or kind != 7:
                    raise RedactionAdmissionError("metadata_invalid")
                table_offset, table_bytes = value, count
        if frames is None or table_bytes != frames * 16 or table_offset is None:
            raise RedactionAdmissionError("insufficient_metadata")
        offsets = []
        for number in range(frames):
            attributes, _, relative, _, _ = struct.unpack_from(
                order + "IIIHH", index, table_offset + number * 16
            )
            if attributes & (7 << 24):
                raise RedactionAdmissionError("metadata_invalid")
            offsets.append(0 if number == 0 else base + relative)
        if len(set(offsets)) != len(offsets):
            raise RedactionAdmissionError("frame_count_mismatch")
        return offsets
    except struct.error:
        raise RedactionAdmissionError("header_truncated") from None


def _png_frames(stream: Any, profile: LimitProfile) -> int:
    stream.seek(0, 2)
    reader = _HeaderStream(stream, stream.tell())
    reader.seek(8)
    frames = 1
    declared = None
    frame_controls = 0
    separate_default = False
    for _ in range(_HEADER_ITEMS):
        length, kind = struct.unpack(">I4s", reader.exact(8))
        end = reader.tell() + length + 4
        if kind == b"acTL":
            if declared is not None or length != 8:
                raise RedactionAdmissionError("metadata_invalid")
            declared, _ = struct.unpack(">II", reader.exact(8))
            if declared == 0:
                raise RedactionAdmissionError("metadata_invalid")
            check_limit(profile, "frames", declared)
            frames = declared
        elif kind == b"fcTL":
            if declared is None or length != 26:
                raise RedactionAdmissionError("metadata_invalid")
            frame_controls += 1
            check_limit(profile, "frames", frame_controls + int(separate_default))
            _, width, height, _, _, _, _, _, _ = struct.unpack(
                ">IIIIIHHBB", reader.exact(26)
            )
            check_geometry(profile, width, height)
        elif kind == b"IDAT" and declared is not None and frame_controls == 0:
            separate_default = True
            frames = declared + 1
            check_limit(profile, "frames", frames)
        reader.seek(end)
        if kind == b"IEND":
            if declared is not None and frame_controls != declared:
                raise RedactionAdmissionError("frame_count_mismatch")
            return frames
    raise RedactionAdmissionError("header_item_limit", limit=_HEADER_ITEMS)


def _tiff_frames(stream: Any, profile: LimitProfile) -> int:
    # Read sparse directories without reading strips/tiles or retaining tags.
    stream.seek(0, 2)
    reader = _HeaderStream(stream, stream.tell())
    reader.seek(0)
    header = reader.exact(8)
    order = "<" if header[:2] == b"II" else ">"
    offset = struct.unpack(order + "I", header[4:])[0]
    seen: set[int] = set()
    count = total_pixels = 0
    while offset:
        if offset in seen or offset < 8:
            raise RedactionAdmissionError("tiff_ifd_offset_invalid")
        seen.add(offset)
        count += 1
        check_limit(profile, "frames", count)
        reader.seek(offset)
        entries = struct.unpack(order + "H", reader.exact(2))[0]
        if entries > DEFAULT_MAX_IFD_ENTRIES:
            raise RedactionAdmissionError("tiff_ifd_entry_limit_exceeded")
        table = bytearray(reader.exact(entries * 12))
        offset = struct.unpack(order + "I", reader.exact(4))[0]
        # Rebase the allowlisted first-IFD reader onto this sparse directory.
        tail = bytearray()
        for index in range(entries):
            start = index * 12
            tag, value_type, values, raw = struct.unpack(
                order + "HHI4s", table[start : start + 12]
            )
            if tag not in {256, 257, 258, 259, 262, 277, 284}:
                continue
            unit = {1: 1, 3: 2, 4: 4}.get(value_type)
            if unit is None or values > 256:
                raise RedactionAdmissionError("metadata_invalid")
            size = values * unit
            if size > 4:
                reader.seek(struct.unpack(order + "I", raw)[0])
                rebased = 8 + 2 + len(table) + 4 + len(tail)
                table[start + 8 : start + 12] = struct.pack(order + "I", rebased)
                tail.extend(reader.exact(size))
        payload = (
            header[:4]
            + struct.pack(order + "I", 8)
            + struct.pack(order + "H", entries)
            + table
            + b"\0" * 4
            + tail
        )
        metadata = read_tiff_metadata(bytes(payload), max_pixels=profile.max_pixels)
        check_geometry(profile, metadata.width, metadata.height)
        total_pixels += metadata.width * metadata.height
        check_limit(profile, "total_pixels", total_pixels)
    if count == 0:
        raise RedactionAdmissionError(
            "insufficient_metadata", "frames", profile.max_frames
        )
    return count


def pdf_page_pixels(width: float, height: float, resolution: int) -> int:
    """Conservatively convert PDF points to raster pixels at the actual DPI."""
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
        or type(resolution) is not int
        or resolution <= 0
    ):
        raise RedactionAdmissionError("metadata_invalid")
    return math.ceil(width * resolution / 72) * math.ceil(height * resolution / 72)


def admit_pdf(source: Any, profile: LimitProfile, *, resolution: int = 150) -> int:
    """Check page-tree geometry and raster budgets before opening a decoder."""
    with _source(source, profile, {"application/pdf"}) as pair:
        report = read_pdf_geometry(
            pair[0], max_bytes=profile.max_byte_size, max_pages=profile.max_pages
        )
        if report.status is not PdfGeometryStatus.READABLE:
            raise RedactionAdmissionError(report.reason_codes[0])
        total_pixels = 0
        for page in report.pages:
            # pdfplumber renders the complete media box before cropping. Budget
            # that allocation rather than just the visible crop box.
            x0, y0, x1, y1 = page.media_box
            pixels = pdf_page_pixels(x1 - x0, y1 - y0, resolution)
            check_limit(profile, "pixels", pixels)
            total_pixels += pixels
            check_limit(profile, "total_pixels", total_pixels)
        if not report.page_count:
            raise RedactionAdmissionError(
                "insufficient_metadata", "pages", profile.max_pages
            )
        return report.page_count


def admit_dicom(source: Any, profile: LimitProfile, pydicom: Any) -> None:
    """Read only numeric DICOM geometry through a bounded metadata transport."""
    with _source(source, profile, {"application/dicom"}) as pair:
        stream = pair[0]
        stream.seek(0, 2)
        reader = _HeaderStream(stream, stream.tell())
        reader.seek(0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                dataset = pydicom.dcmread(
                    reader,
                    stop_before_pixels=True,
                    specific_tags=["Rows", "Columns", "NumberOfFrames"],
                )
                # A metadata-only object reaches EOF without a pixel tag and
                # cannot allocate frames. Preserve the existing header-only path.
                if reader.tell() == reader.size:
                    return
                width = int(dataset.Columns)
                height = int(dataset.Rows)
                frames = int(getattr(dataset, "NumberOfFrames", 1))
        except RedactionAdmissionError:
            raise
        except Exception:
            raise RedactionAdmissionError("insufficient_metadata") from None
        check_geometry(profile, width, height, frames=frames)
