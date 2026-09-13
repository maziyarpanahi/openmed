"""Synthetic unit and file-level tests for BMP header preflight."""

from __future__ import annotations

import io
import struct
from dataclasses import FrozenInstanceError, asdict

import pytest

from openmed.multimodal.bmp_dimensions import BmpDimensionsError, read_bmp_dimensions


def bmp(width=7, height=5, *, core=False, depth=24, planes=1, compression=0):
    header_size = 12 if core else 40
    palette_entries = 2**depth if depth in (1, 4, 8) else 0
    offset = 14 + header_size + palette_entries * (3 if core else 4)
    rows = ((abs(width) * depth + 31) // 32) * 4 * abs(height)
    file_header = struct.pack("<2sIHHI", b"BM", offset + rows, 0, 0, offset)
    if core:
        return file_header + struct.pack("<IHHHH", 12, width, height, planes, depth)
    return file_header + struct.pack(
        "<IiiHHIIiiII", 40, width, height, planes, depth, compression, rows, 0, 0, 0, 0
    )


def valid_payload():
    payload = bmp()
    return payload, len(payload)


def edit(payload: bytes, offset: int, fmt: str, value: int) -> bytes:
    changed = bytearray(payload)
    struct.pack_into(fmt, changed, offset, value)
    return bytes(changed)


@pytest.mark.parametrize("depth", [1, 4, 8, 24])
def test_core_headers(depth) -> None:
    result = read_bmp_dimensions(bmp(core=True, depth=depth))
    assert asdict(result) == {
        "width": 7,
        "height": 5,
        "planes": 1,
        "bit_depth": depth,
        "top_down": False,
        "dib_header_bytes": 12,
    }


@pytest.mark.parametrize("height", [5, -5])
@pytest.mark.parametrize("depth", [1, 4, 8, 16, 24, 32])
def test_info_headers_and_top_down_rows(height, depth) -> None:
    result = read_bmp_dimensions(bmp(height=height, depth=depth))
    assert (result.width, result.height, result.bit_depth) == (7, 5, depth)
    assert result.top_down is (height < 0)
    assert result.planes == 1
    assert result.dib_header_bytes == 40


@pytest.mark.parametrize("core,boundary", [(True, 26), (False, 54)])
def test_all_truncated_header_boundaries(core, boundary) -> None:
    for cut in range(boundary):
        with pytest.raises(BmpDimensionsError, match="^bmp_header_truncated$"):
            read_bmp_dimensions(bmp(core=core)[:cut])
    assert read_bmp_dimensions(bmp(core=core), max_header_bytes=boundary).width == 7
    with pytest.raises(BmpDimensionsError, match="^bmp_header_limit_exceeded$"):
        read_bmp_dimensions(bmp(core=core), max_header_bytes=boundary - 1)


@pytest.mark.parametrize("header_size", [0, 11, 16, 52, 56, 64, 108, 124, 0xFFFFFFFF])
def test_unsupported_dib_never_triggers_unbounded_reads(header_size) -> None:
    payload = edit(bmp(), 14, "<I", header_size)
    stream = ShortStream(payload, chunk=14, boundary=18)
    with pytest.raises(BmpDimensionsError, match="^bmp_dib_header_unsupported$"):
        read_bmp_dimensions(stream)
    assert stream.stream.tell() == 18


@pytest.mark.parametrize("planes", [0, 2, 65535])
@pytest.mark.parametrize("core", [False, True])
def test_plane_count_must_be_one(planes, core) -> None:
    with pytest.raises(BmpDimensionsError, match="^bmp_planes_invalid$"):
        read_bmp_dimensions(bmp(planes=planes, core=core))


@pytest.mark.parametrize("depth", [0, 2, 3, 64, 65535])
def test_invalid_bit_depths(depth) -> None:
    payload = edit(bmp(), 28, "<H", depth)
    with pytest.raises(BmpDimensionsError, match="^bmp_bit_depth_unsupported$"):
        read_bmp_dimensions(payload)


@pytest.mark.parametrize("compression", [1, 2, 3, 4, 5, 6, 0xFFFFFFFF])
def test_compressed_layouts_are_explicitly_unsupported(compression) -> None:
    with pytest.raises(BmpDimensionsError, match="^bmp_compression_unsupported$"):
        read_bmp_dimensions(bmp(compression=compression))


@pytest.mark.parametrize(
    "offset,fmt,value,category",
    [
        (0, "<H", 0, "bmp_signature_invalid"),
        (6, "<H", 1, "bmp_reserved_fields_invalid"),
        (8, "<H", 1, "bmp_reserved_fields_invalid"),
        (18, "<i", 0, "bmp_dimensions_invalid"),
        (18, "<i", -1, "bmp_dimensions_invalid"),
        (22, "<i", 0, "bmp_dimensions_invalid"),
        (10, "<I", 53, "bmp_pixel_offset_invalid"),
        (2, "<I", 53, "bmp_file_size_invalid"),
        (34, "<I", 1, "bmp_image_size_invalid"),
    ],
)
def test_malformed_fields_have_stable_categories(offset, fmt, value, category) -> None:
    with pytest.raises(BmpDimensionsError) as raised:
        read_bmp_dimensions(edit(bmp(), offset, fmt, value))
    assert raised.value.category == category
    assert str(raised.value) == category


def test_zero_image_size_is_allowed_for_uncompressed_info() -> None:
    assert read_bmp_dimensions(edit(bmp(), 34, "<I", 0)).width == 7


def test_palette_count_and_offset_are_checked_without_reading_palette() -> None:
    payload = bmp(depth=8)
    assert read_bmp_dimensions(payload).bit_depth == 8
    with pytest.raises(BmpDimensionsError, match="^bmp_color_table_size_invalid$"):
        read_bmp_dimensions(edit(payload, 46, "<I", 257))
    with pytest.raises(BmpDimensionsError, match="^bmp_pixel_offset_invalid$"):
        read_bmp_dimensions(edit(payload, 10, "<I", 54))
    reduced = edit(edit(payload, 46, "<I", 2), 10, "<I", 62)
    assert read_bmp_dimensions(reduced).bit_depth == 8


def test_uint32_pixel_extent_cannot_overflow() -> None:
    payload = edit(edit(bmp(), 10, "<I", 0xFFFFFFFE), 2, "<I", 0xFFFFFFFF)
    with pytest.raises(BmpDimensionsError, match="^bmp_file_size_invalid$"):
        read_bmp_dimensions(payload)
    payload = edit(edit(bmp(), 22, "<i", -(1 << 31)), 2, "<I", 0xFFFFFFFF)
    with pytest.raises(BmpDimensionsError, match="^bmp_file_size_invalid$"):
        read_bmp_dimensions(payload, max_pixels=1 << 63)


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA"])
def test_pillow_generated_file_end_to_end(tmp_path, size, mode) -> None:
    from PIL import Image

    path = tmp_path / "synthetic.bmp"
    Image.new(mode, size).save(path, format="BMP")
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        result = read_bmp_dimensions(stream)
        assert stream.tell() == 0
    assert (result.width, result.height) == expected == size


class ShortStream:
    """A real non-seekable byte stream that records every bounded read."""

    def __init__(self, data: bytes, chunk: int = 3, boundary: int | None = None):
        self.stream = io.BytesIO(data)
        self.chunk = chunk
        self.boundary = len(data) if boundary is None else boundary
        self.requests: list[int] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        assert size >= 0
        assert self.stream.tell() + size <= self.boundary
        self.requests.append(size)
        return self.stream.read(min(size, self.chunk))


def test_partial_nonseekable_reads_stop_at_header() -> None:
    payload, boundary = valid_payload()
    stream = ShortStream(payload + b"unread-payload", boundary=boundary)
    result = read_bmp_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.stream.tell() == boundary
    assert not stream.closed
    assert sum(min(n, stream.chunk) for n in stream.requests) == boundary


def test_seekable_stream_at_nonzero_position_is_restored() -> None:
    payload, _ = valid_payload()
    stream = io.BytesIO(b"prefix" + payload)
    stream.seek(6)
    result = read_bmp_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.tell() == 6
    assert not stream.closed


def test_failure_restores_caller_owned_stream() -> None:
    stream = io.BytesIO(b"prefix" + b"truncated")
    stream.seek(6)
    with pytest.raises(BmpDimensionsError):
        read_bmp_dimensions(stream)
    assert stream.tell() == 6
    assert not stream.closed


@pytest.mark.parametrize("option", ["max_header_bytes", "max_pixels"])
@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "2", None])
def test_invalid_limits_do_not_read(option, value) -> None:
    class Unreadable:
        def read(self, size):
            pytest.fail("invalid options must be checked before I/O")

    with pytest.raises(ValueError, match=option):
        read_bmp_dimensions(Unreadable(), **{option: value})


@pytest.mark.parametrize("source", [None, "not-bytes", bytearray(b"data"), 12])
def test_invalid_source_type(source) -> None:
    with pytest.raises(TypeError, match="source must be bytes or a binary stream"):
        read_bmp_dimensions(source)


@pytest.mark.parametrize("returned", [None, "text", bytearray(b"x"), b"x" * 1000])
def test_binary_stream_contract_errors_are_value_free(returned) -> None:
    class WrongStream:
        def read(self, size):
            return returned

    with pytest.raises(BmpDimensionsError) as raised:
        read_bmp_dimensions(WrongStream())
    assert raised.value.category == "bmp_stream_contract_error"
    assert str(raised.value) == raised.value.category


def test_read_error_does_not_retain_io_details() -> None:
    class BrokenStream:
        def read(self, size):
            raise OSError("synthetic-private-file-detail")

    with pytest.raises(BmpDimensionsError) as raised:
        read_bmp_dimensions(BrokenStream())
    assert str(raised.value) == "bmp_stream_read_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("failure", ["seekable", "tell", "bad-position", "seek"])
def test_position_errors_do_not_retain_io_details(failure) -> None:
    payload, _ = valid_payload()

    class PositionStream(io.BytesIO):
        def seekable(self):
            if failure == "seekable":
                raise OSError("synthetic-private-file-detail")
            return True

        def tell(self):
            if failure == "tell":
                raise OSError("synthetic-private-file-detail")
            if failure == "bad-position":
                return True
            return super().tell()

        def seek(self, offset, whence=0):
            if failure == "seek":
                raise OSError("synthetic-private-file-detail")
            return super().seek(offset, whence)

    stream = PositionStream(payload)
    with pytest.raises(BmpDimensionsError) as raised:
        read_bmp_dimensions(stream)
    expected = "restore" if failure == "seek" else "position"
    assert str(raised.value) == f"bmp_stream_{expected}_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not stream.closed


def test_pixel_limit_boundary() -> None:
    payload, _ = valid_payload()
    result = read_bmp_dimensions(payload, max_pixels=35)
    assert result.width * result.height == 35
    with pytest.raises(BmpDimensionsError, match="^bmp_pixel_limit_exceeded$"):
        read_bmp_dimensions(payload, max_pixels=34)


def test_output_contains_only_declared_metadata() -> None:
    payload, _ = valid_payload()
    result = read_bmp_dimensions(payload)
    assert all(isinstance(v, (int, str, bool)) for v in asdict(result).values())
    with pytest.raises(FrozenInstanceError):
        result.width = 1


@pytest.mark.integration
def test_real_file_read_preserves_position_and_ownership(tmp_path) -> None:
    payload, _ = valid_payload()
    path = tmp_path / "synthetic-image.bin"
    path.write_bytes(b"prefix" + payload + b"not-part-of-the-header")
    with path.open("rb") as stream:
        stream.seek(6)
        result = read_bmp_dimensions(stream)
        assert (result.width, result.height) == (7, 5)
        assert stream.tell() == 6
        assert not stream.closed


@pytest.mark.integration
@pytest.mark.parametrize(
    "core,depth,height",
    [(True, 1, 5), (True, 8, 5), (True, 24, 5), (False, 24, -5)],
)
def test_core_and_top_down_files_against_pillow(tmp_path, core, depth, height) -> None:
    from PIL import Image

    header = bmp(core=core, depth=depth, height=height)
    declared_size = struct.unpack_from("<I", header, 2)[0]
    path = tmp_path / "synthetic-supported-layout.bmp"
    path.write_bytes(header + bytes(declared_size - len(header)))
    with Image.open(path) as oracle:
        oracle.load()
        expected = oracle.size
    with path.open("rb") as stream:
        result = read_bmp_dimensions(stream)
    assert (result.width, result.height) == expected == (7, 5)
    assert result.top_down is (height < 0)


def test_explicit_nonseekable_stream_does_not_attempt_restoration() -> None:
    payload, boundary = valid_payload()

    class NoSeek(ShortStream):
        def seekable(self):
            return False

        def seek(self, *args):
            pytest.fail("a nonseekable stream must not be rewound")

    stream = NoSeek(payload, boundary=boundary)
    assert read_bmp_dimensions(stream).width == 7
    assert stream.stream.tell() == boundary
