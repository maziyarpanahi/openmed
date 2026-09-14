"""Synthetic unit and codec-generated file tests for WebP header preflight."""

from __future__ import annotations

import io
import struct
from dataclasses import FrozenInstanceError, asdict

import pytest

from openmed.multimodal.webp_dimensions import WebpDimensionsError, read_webp_dimensions


def webp(width=7, height=5, *, kind=b"VP8 ", flags=0):
    if kind == b"VP8 ":
        payload = b"\x10\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    elif kind == b"VP8L":
        bits = (width - 1) | ((height - 1) << 14) | ((flags & 1) << 28)
        payload = b"\x2f" + struct.pack("<I", bits)
    else:
        payload = (
            bytes([flags, 0, 0, 0])
            + (width - 1).to_bytes(3, "little")
            + (height - 1).to_bytes(3, "little")
        )
    chunk = kind + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        chunk += b"\x00"
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def valid_payload():
    payload = webp()
    return payload, 30


def edit(payload: bytes, offset: int, fmt: str, value: int) -> bytes:
    changed = bytearray(payload)
    struct.pack_into(fmt, changed, offset, value)
    return bytes(changed)


@pytest.mark.parametrize("kind", [b"VP8 ", b"VP8L", b"VP8X"])
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (258, 772), (16383, 1)])
def test_each_header_layout(kind, size) -> None:
    result = read_webp_dimensions(webp(*size, kind=kind))
    assert (result.width, result.height) == size
    assert result.chunk_type == kind.decode("ascii").strip()
    assert not result.is_animated
    assert not result.has_alpha


@pytest.mark.parametrize("flags", [0, 2, 4, 8, 16, 32, 62])
def test_extended_flags_are_metadata_not_animation_traversal(flags) -> None:
    payload = webp(kind=b"VP8X", flags=flags)
    stream = ShortStream(payload + b"do-not-read-ANIM-EXIF-XMP", boundary=30)
    result = read_webp_dimensions(stream)
    assert result.is_animated is bool(flags & 2)
    assert result.has_alpha is bool(flags & 16)
    assert stream.stream.tell() == 30


def test_lossless_alpha_flag() -> None:
    assert read_webp_dimensions(webp(kind=b"VP8L", flags=1)).has_alpha


def test_lossy_scaling_bits_do_not_change_canvas_size() -> None:
    payload = webp(7 | 0xC000, 5 | 0x8000)
    result = read_webp_dimensions(payload)
    assert (result.width, result.height) == (7, 5)


@pytest.mark.parametrize("kind,required", [(b"VP8 ", 30), (b"VP8L", 25), (b"VP8X", 30)])
def test_truncation_and_exact_byte_budget(kind, required) -> None:
    payload = webp(kind=kind)
    for cut in range(required):
        with pytest.raises(WebpDimensionsError, match="^webp_header_truncated$"):
            read_webp_dimensions(payload[:cut])
    result = read_webp_dimensions(payload[:required], max_header_bytes=required)
    assert result.width == 7
    with pytest.raises(WebpDimensionsError, match="^webp_header_limit_exceeded$"):
        read_webp_dimensions(payload, max_header_bytes=required - 1)


@pytest.mark.parametrize("size", [0, 3, 4, 11, 13, 0xFFFFFFFF, 0xFFFFFFF8])
def test_invalid_riff_size(size) -> None:
    with pytest.raises(WebpDimensionsError, match="^webp_riff_size_invalid$"):
        read_webp_dimensions(edit(webp(), 4, "<I", size))


@pytest.mark.parametrize("offset", [0, 8])
def test_invalid_riff_or_webp_signature(offset) -> None:
    with pytest.raises(WebpDimensionsError, match="^webp_signature_invalid$"):
        read_webp_dimensions(edit(webp(), offset, "<I", 0))


@pytest.mark.parametrize(
    "kind,size",
    [
        (b"VP8 ", 0),
        (b"VP8 ", 9),
        (b"VP8L", 4),
        (b"VP8X", 9),
        (b"VP8X", 11),
        (b"VP8 ", 0xFFFFFFFF),
    ],
)
def test_declared_chunk_sizes(kind, size) -> None:
    with pytest.raises(WebpDimensionsError, match="^webp_chunk_size_invalid$"):
        read_webp_dimensions(edit(webp(kind=kind), 16, "<I", size))


def test_odd_chunk_padding_must_fit_declared_riff_extent() -> None:
    payload = edit(webp(), 16, "<I", 11)
    with pytest.raises(WebpDimensionsError, match="^webp_chunk_size_invalid$"):
        read_webp_dimensions(payload)


def test_unknown_first_chunk_is_not_scanned() -> None:
    payload = webp(kind=b"JUNK")
    stream = ShortStream(payload, chunk=12, boundary=20)
    with pytest.raises(WebpDimensionsError, match="^webp_layout_unsupported$"):
        read_webp_dimensions(stream)
    assert stream.stream.tell() == 20


@pytest.mark.parametrize(
    "offset,value,category",
    [
        (20, 17, "webp_vp8_header_invalid"),
        (20, 24, "webp_vp8_version_unsupported"),
        (23, 0, "webp_vp8_header_invalid"),
        (26, 0, "webp_dimensions_invalid"),
        (28, 0, "webp_dimensions_invalid"),
    ],
)
def test_invalid_lossy_headers(offset, value, category) -> None:
    with pytest.raises(WebpDimensionsError) as raised:
        read_webp_dimensions(edit(webp(), offset, "<B", value))
    assert str(raised.value) == category


def test_invalid_lossless_signature_and_version() -> None:
    with pytest.raises(WebpDimensionsError, match="^webp_vp8l_header_invalid$"):
        read_webp_dimensions(edit(webp(kind=b"VP8L"), 20, "<B", 0))
    with pytest.raises(WebpDimensionsError, match="^webp_vp8l_version_unsupported$"):
        read_webp_dimensions(edit(webp(kind=b"VP8L"), 24, "<B", 0x20))


@pytest.mark.parametrize(
    "offset,value", [(20, 1), (20, 64), (20, 128), (21, 1), (22, 1), (23, 1)]
)
def test_extended_reserved_fields(offset, value) -> None:
    with pytest.raises(WebpDimensionsError, match="^webp_vp8x_reserved_bits_invalid$"):
        read_webp_dimensions(edit(webp(kind=b"VP8X"), offset, "<B", value))


def test_extended_canvas_product_limit_cannot_be_disabled() -> None:
    payload = webp(1 << 24, 1 << 24, kind=b"VP8X")
    with pytest.raises(WebpDimensionsError, match="^webp_dimensions_invalid$"):
        read_webp_dimensions(payload, max_pixels=1 << 63)
    assert read_webp_dimensions(webp(1 << 24, 1, kind=b"VP8X")).width == 1 << 24


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("lossless", [False, True])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_pillow_generated_file_end_to_end(tmp_path, size, lossless, mode) -> None:
    from PIL import Image, features

    assert features.check("webp"), "Pillow must include WebP for this integration test"
    path = tmp_path / "synthetic.webp"
    Image.new(mode, size).save(path, format="WEBP", lossless=lossless)
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        result = read_webp_dimensions(stream)
        assert stream.tell() == 0
    assert (result.width, result.height) == expected == size


@pytest.mark.integration
def test_real_animated_webp_uses_canvas_not_frame_payloads(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "synthetic-animated.webp"
    first = Image.new("RGB", (19, 13), (0, 0, 0))
    second = Image.new("RGB", (19, 13), (255, 255, 255))
    first.save(path, format="WEBP", save_all=True, append_images=[second], duration=10)
    with Image.open(path) as decoded:
        assert decoded.n_frames == 2
    stream = ShortStream(path.read_bytes(), boundary=30)
    result = read_webp_dimensions(stream)
    assert (result.width, result.height) == (19, 13)
    assert result.chunk_type == "VP8X"
    assert result.is_animated
    assert stream.stream.tell() == 30


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
    result = read_webp_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.stream.tell() == boundary
    assert not stream.closed
    assert sum(min(n, stream.chunk) for n in stream.requests) == boundary


def test_seekable_stream_at_nonzero_position_is_restored() -> None:
    payload, _ = valid_payload()
    stream = io.BytesIO(b"prefix" + payload)
    stream.seek(6)
    result = read_webp_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.tell() == 6
    assert not stream.closed


def test_failure_restores_caller_owned_stream() -> None:
    stream = io.BytesIO(b"prefix" + b"truncated")
    stream.seek(6)
    with pytest.raises(WebpDimensionsError):
        read_webp_dimensions(stream)
    assert stream.tell() == 6
    assert not stream.closed


@pytest.mark.parametrize("option", ["max_header_bytes", "max_pixels"])
@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "2", None])
def test_invalid_limits_do_not_read(option, value) -> None:
    class Unreadable:
        def read(self, size):
            pytest.fail("invalid options must be checked before I/O")

    with pytest.raises(ValueError, match=option):
        read_webp_dimensions(Unreadable(), **{option: value})


@pytest.mark.parametrize("source", [None, "not-bytes", bytearray(b"data"), 12])
def test_invalid_source_type(source) -> None:
    with pytest.raises(TypeError, match="source must be bytes or a binary stream"):
        read_webp_dimensions(source)


@pytest.mark.parametrize("returned", [None, "text", bytearray(b"x"), b"x" * 1000])
def test_binary_stream_contract_errors_are_value_free(returned) -> None:
    class WrongStream:
        def read(self, size):
            return returned

    with pytest.raises(WebpDimensionsError) as raised:
        read_webp_dimensions(WrongStream())
    assert raised.value.category == "webp_stream_contract_error"
    assert str(raised.value) == raised.value.category


def test_read_error_does_not_retain_io_details() -> None:
    class BrokenStream:
        def read(self, size):
            raise OSError("synthetic-private-file-detail")

    with pytest.raises(WebpDimensionsError) as raised:
        read_webp_dimensions(BrokenStream())
    assert str(raised.value) == "webp_stream_read_error"
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
    with pytest.raises(WebpDimensionsError) as raised:
        read_webp_dimensions(stream)
    expected = "restore" if failure == "seek" else "position"
    assert str(raised.value) == f"webp_stream_{expected}_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not stream.closed


def test_pixel_limit_boundary() -> None:
    payload, _ = valid_payload()
    result = read_webp_dimensions(payload, max_pixels=35)
    assert result.width * result.height == 35
    with pytest.raises(WebpDimensionsError, match="^webp_pixel_limit_exceeded$"):
        read_webp_dimensions(payload, max_pixels=34)


def test_output_contains_only_declared_metadata() -> None:
    payload, _ = valid_payload()
    result = read_webp_dimensions(payload)
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
        result = read_webp_dimensions(stream)
        assert (result.width, result.height) == (7, 5)
        assert stream.tell() == 6
        assert not stream.closed


def test_explicit_nonseekable_stream_does_not_attempt_restoration() -> None:
    payload, boundary = valid_payload()

    class NoSeek(ShortStream):
        def seekable(self):
            return False

        def seek(self, *args):
            pytest.fail("a nonseekable stream must not be rewound")

    stream = NoSeek(payload, boundary=boundary)
    assert read_webp_dimensions(stream).width == 7
    assert stream.stream.tell() == boundary
