"""Synthetic unit and file-level tests for GIF header preflight."""

from __future__ import annotations

import io
import struct
from dataclasses import FrozenInstanceError, asdict

import pytest

from openmed.multimodal.gif_dimensions import (
    DEFAULT_MAX_GIF_HEADER_BYTES,
    GifDimensionsError,
    read_gif_dimensions,
)


def gif(width=7, height=5, *, version=b"GIF89a", table_bits=None, background=0):
    packed = 0x77 if table_bits is None else 0x80 | table_bits
    table = b"" if table_bits is None else bytes(range(256)) * 3
    table = table[: 3 * (1 << (table_bits + 1))] if table_bits is not None else b""
    return version + struct.pack("<HHBBB", width, height, packed, background, 0) + table


def valid_payload():
    payload = gif(table_bits=2)
    return payload, len(payload)


@pytest.mark.parametrize("version", [b"GIF87a", b"GIF89a"])
@pytest.mark.parametrize("table_bits", [None, *range(8)])
def test_versions_and_all_global_table_sizes(version, table_bits) -> None:
    result = read_gif_dimensions(gif(version=version, table_bits=table_bits))
    entries = 0 if table_bits is None else 2 ** (table_bits + 1)
    assert asdict(result) == {
        "width": 7,
        "height": 5,
        "version": version[3:].decode("ascii"),
        "global_color_table_entries": entries,
    }
    assert result.global_color_table_bytes == entries * 3


def test_hand_encoded_little_endian_dimensions() -> None:
    payload = bytes.fromhex("474946383761 0201 0403 00 00 00")
    result = read_gif_dimensions(payload)
    assert (result.width, result.height) == (258, 772)


@pytest.mark.parametrize("cut", range(13))
def test_every_truncated_fixed_header(cut) -> None:
    with pytest.raises(GifDimensionsError, match="^gif_header_truncated$"):
        read_gif_dimensions(gif()[:cut])


@pytest.mark.parametrize("table_bits", range(8))
def test_each_truncated_global_table(table_bits) -> None:
    with pytest.raises(GifDimensionsError, match="^gif_header_truncated$"):
        read_gif_dimensions(gif(table_bits=table_bits)[:-1])


@pytest.mark.parametrize("dimensions", [(0, 1), (1, 0), (0, 0)])
def test_zero_dimensions(dimensions) -> None:
    with pytest.raises(GifDimensionsError, match="^gif_dimensions_invalid$"):
        read_gif_dimensions(gif(*dimensions))


def test_maximum_dimensions_have_checked_area() -> None:
    payload = gif(65535, 65535)
    with pytest.raises(GifDimensionsError, match="^gif_pixel_limit_exceeded$"):
        read_gif_dimensions(payload)
    result = read_gif_dimensions(payload, max_pixels=65535**2)
    assert (result.width, result.height) == (65535, 65535)


@pytest.mark.parametrize("signature", [b"GIF88a", b"gif89a", b"notgif"])
def test_signature_is_validated(signature) -> None:
    with pytest.raises(GifDimensionsError, match="^gif_signature_invalid$"):
        read_gif_dimensions(gif(version=signature))


def test_background_index_must_fit_a_present_table() -> None:
    with pytest.raises(GifDimensionsError, match="^gif_background_index_invalid$"):
        read_gif_dimensions(gif(table_bits=0, background=2))
    assert read_gif_dimensions(gif(background=255)).global_color_table_entries == 0


def test_header_limit_checked_before_palette_read() -> None:
    payload = gif(table_bits=7)
    assert len(payload) == DEFAULT_MAX_GIF_HEADER_BYTES
    assert read_gif_dimensions(payload, max_header_bytes=len(payload)).width == 7
    stream = ShortStream(payload, chunk=13, boundary=13)
    with pytest.raises(GifDimensionsError, match="^gif_header_limit_exceeded$"):
        read_gif_dimensions(stream, max_header_bytes=len(payload) - 1)
    assert stream.stream.tell() == 13
    with pytest.raises(GifDimensionsError, match="^gif_header_limit_exceeded$"):
        read_gif_dimensions(gif(), max_header_bytes=12)


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("mode", ["1", "L", "RGB"])
def test_pillow_generated_file_end_to_end(tmp_path, size, mode) -> None:
    from PIL import Image

    path = tmp_path / "synthetic.gif"
    Image.new(mode, size).save(path, format="GIF")
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        result = read_gif_dimensions(stream)
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
    result = read_gif_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.stream.tell() == boundary
    assert not stream.closed
    assert sum(min(n, stream.chunk) for n in stream.requests) == boundary


def test_seekable_stream_at_nonzero_position_is_restored() -> None:
    payload, _ = valid_payload()
    stream = io.BytesIO(b"prefix" + payload)
    stream.seek(6)
    result = read_gif_dimensions(stream)
    assert (result.width, result.height) == (7, 5)
    assert stream.tell() == 6
    assert not stream.closed


def test_failure_restores_caller_owned_stream() -> None:
    stream = io.BytesIO(b"prefix" + b"truncated")
    stream.seek(6)
    with pytest.raises(GifDimensionsError):
        read_gif_dimensions(stream)
    assert stream.tell() == 6
    assert not stream.closed


@pytest.mark.parametrize("option", ["max_header_bytes", "max_pixels"])
@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "2", None])
def test_invalid_limits_do_not_read(option, value) -> None:
    class Unreadable:
        def read(self, size):
            pytest.fail("invalid options must be checked before I/O")

    with pytest.raises(ValueError, match=option):
        read_gif_dimensions(Unreadable(), **{option: value})


@pytest.mark.parametrize("source", [None, "not-bytes", bytearray(b"data"), 12])
def test_invalid_source_type(source) -> None:
    with pytest.raises(TypeError, match="source must be bytes or a binary stream"):
        read_gif_dimensions(source)


@pytest.mark.parametrize("returned", [None, "text", bytearray(b"x"), b"x" * 1000])
def test_binary_stream_contract_errors_are_value_free(returned) -> None:
    class WrongStream:
        def read(self, size):
            return returned

    with pytest.raises(GifDimensionsError) as raised:
        read_gif_dimensions(WrongStream())
    assert raised.value.category == "gif_stream_contract_error"
    assert str(raised.value) == raised.value.category


def test_read_error_does_not_retain_io_details() -> None:
    class BrokenStream:
        def read(self, size):
            raise OSError("synthetic-private-file-detail")

    with pytest.raises(GifDimensionsError) as raised:
        read_gif_dimensions(BrokenStream())
    assert str(raised.value) == "gif_stream_read_error"
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
    with pytest.raises(GifDimensionsError) as raised:
        read_gif_dimensions(stream)
    expected = "restore" if failure == "seek" else "position"
    assert str(raised.value) == f"gif_stream_{expected}_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not stream.closed


def test_pixel_limit_boundary() -> None:
    payload, _ = valid_payload()
    result = read_gif_dimensions(payload, max_pixels=35)
    assert result.width * result.height == 35
    with pytest.raises(GifDimensionsError, match="^gif_pixel_limit_exceeded$"):
        read_gif_dimensions(payload, max_pixels=34)


def test_output_contains_only_declared_metadata() -> None:
    payload, _ = valid_payload()
    result = read_gif_dimensions(payload)
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
        result = read_gif_dimensions(stream)
        assert (result.width, result.height) == (7, 5)
        assert stream.tell() == 6
        assert not stream.closed


@pytest.mark.integration
def test_real_gif_with_local_but_no_global_color_table(tmp_path) -> None:
    from PIL import Image

    # A complete 1x1 GIF89a with a two-entry local palette, not a global one.
    payload = bytes.fromhex(
        "474946383961010001000000002c000000000100010080000000ffffff02024401003b"
    )
    path = tmp_path / "synthetic-local-palette.gif"
    path.write_bytes(payload)
    with Image.open(path) as oracle:
        oracle.load()
        assert oracle.size == (1, 1)
    stream = ShortStream(payload, boundary=13)
    result = read_gif_dimensions(stream)
    assert (result.width, result.height) == (1, 1)
    assert result.global_color_table_entries == 0
    assert stream.stream.tell() == 13


def test_explicit_nonseekable_stream_does_not_attempt_restoration() -> None:
    payload, boundary = valid_payload()

    class NoSeek(ShortStream):
        def seekable(self):
            return False

        def seek(self, *args):
            pytest.fail("a nonseekable stream must not be rewound")

    stream = NoSeek(payload, boundary=boundary)
    assert read_gif_dimensions(stream).width == 7
    assert stream.stream.tell() == boundary
