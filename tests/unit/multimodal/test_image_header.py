"""Synthetic unit and file-level tests for PNG and JPEG header preflight."""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import FrozenInstanceError, asdict

import pytest

from openmed.multimodal.image_header import (
    DEFAULT_MAX_IMAGE_HEADER_BYTES,
    DEFAULT_MAX_JPEG_MARKERS,
    ImageFormat,
    ImageHeaderError,
    read_image_header,
    read_jpeg_header,
    read_png_header,
)

PNG_PREFIX_BYTES = 8 + 4 + 4
PNG_HEADER_BYTES = PNG_PREFIX_BYTES + 13 + 4


def ihdr(width=7, height=5, depth=8, color=2, compression=0, filtering=0, interlace=0):
    return struct.pack(
        ">IIBBBBB", width, height, depth, color, compression, filtering, interlace
    )


def png(*, length=13, chunk=b"IHDR", crc=None, trailer=b"IDAT-not-read", **fields):
    payload = ihdr(**fields)
    checksum = zlib.crc32(chunk + payload) if crc is None else crc
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", length)
        + chunk
        + payload
        + struct.pack(">I", checksum)
        + trailer
    )


def segment(marker: int, payload: bytes = b"") -> bytes:
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def sof(width=7, height=5, precision=8, components=3, marker=0xC0, length=None):
    payload = struct.pack(">BHHB", precision, height, width, components)
    payload += b"\x01\x11\x00" * components
    declared = len(payload) + 2 if length is None else length
    return b"\xff" + bytes([marker]) + struct.pack(">H", declared) + payload


def jpeg(prefix: bytes = b"", trailer: bytes | None = None, **fields):
    tail = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f" if trailer is None else trailer
    return b"\xff\xd8" + prefix + sof(**fields) + tail


APP0 = segment(0xE0, b"JFIF\x00\x01\x02\x00\x00\x01\x00\x01\x00\x00")
COM = segment(0xFE, b"synthetic comment")


def test_png_truecolor_header() -> None:
    assert asdict(read_image_header(png())) == {
        "image_format": ImageFormat.PNG,
        "width": 7,
        "height": 5,
        "bit_depth": 8,
        "component_count": 3,
        "color_type": 2,
        "progressive": False,
    }


@pytest.mark.parametrize(
    "color,depth,components",
    [
        (0, 1, 1),
        (0, 16, 1),
        (2, 8, 3),
        (2, 16, 3),
        (3, 4, 1),
        (4, 8, 2),
        (6, 16, 4),
    ],
)
def test_png_color_types_and_depths(color, depth, components) -> None:
    header = read_png_header(png(color=color, depth=depth))
    assert (header.color_type, header.bit_depth) == (color, depth)
    assert header.component_count == components
    assert header.image_format is ImageFormat.PNG


def test_png_adam7_interlace_is_categorical() -> None:
    assert read_png_header(png(interlace=1)).progressive is True
    assert read_png_header(png(interlace=0)).progressive is False


def test_png_large_dimensions_are_read_without_allocation() -> None:
    header = read_png_header(png(width=100_000, height=100_000), max_pixels=10**10)
    assert (header.width, header.height) == (100_000, 100_000)


def test_baseline_jpeg_header() -> None:
    assert asdict(read_image_header(jpeg())) == {
        "image_format": ImageFormat.JPEG,
        "width": 7,
        "height": 5,
        "bit_depth": 8,
        "component_count": 3,
        "color_type": None,
        "progressive": False,
    }


@pytest.mark.parametrize("marker", [0xC2, 0xC6, 0xCA, 0xCE])
def test_progressive_jpeg_frames_are_categorical(marker) -> None:
    assert read_jpeg_header(jpeg(marker=marker)).progressive is True


@pytest.mark.parametrize("marker", [0xC0, 0xC1, 0xC3, 0xC5, 0xC7, 0xC9, 0xCB, 0xCF])
def test_sequential_jpeg_frames_are_categorical(marker) -> None:
    assert read_jpeg_header(jpeg(marker=marker)).progressive is False


@pytest.mark.parametrize("precision", [8, 12, 16])
@pytest.mark.parametrize("components", [1, 3, 4])
def test_jpeg_precision_and_component_counts(precision, components) -> None:
    header = read_jpeg_header(jpeg(precision=precision, components=components))
    assert (header.bit_depth, header.component_count) == (precision, components)


def test_jpeg_segments_before_the_frame_are_skipped() -> None:
    header = read_image_header(jpeg(prefix=APP0 + COM + segment(0xDB, b"\x00" * 65)))
    assert (header.width, header.height) == (7, 5)


def test_jpeg_restart_and_standalone_markers_are_skipped() -> None:
    standalone = b"\xff\x01" + b"".join(b"\xff" + bytes([m]) for m in range(0xD0, 0xD8))
    assert read_image_header(jpeg(prefix=standalone)).width == 7


def test_jpeg_fill_bytes_before_a_marker_are_tolerated() -> None:
    assert (
        read_image_header(b"\xff\xd8" + b"\xff" * 8 + sof()[1:] + b"\xff\xd9").width
        == 7
    )


@pytest.mark.parametrize("payload", [b"", b"\x89PNG", b"\xff", b"\xff\xd9", b"GIF89a"])
def test_unsupported_or_short_signatures_fail_closed(payload) -> None:
    with pytest.raises(ImageHeaderError) as excinfo:
        read_image_header(payload)
    assert excinfo.value.category in {
        "image_signature_unsupported",
        "image_header_truncated",
        "jpeg_marker_invalid",
    }


def test_png_reader_rejects_a_jpeg_and_the_reverse() -> None:
    with pytest.raises(ImageHeaderError, match="^png_signature_invalid$"):
        read_png_header(jpeg())
    with pytest.raises(ImageHeaderError, match="^jpeg_signature_invalid$"):
        read_jpeg_header(png())


def test_all_truncated_png_boundaries_fail_closed() -> None:
    payload = png()
    for cut in range(PNG_HEADER_BYTES):
        with pytest.raises(ImageHeaderError, match="^image_header_truncated$"):
            read_image_header(payload[:cut])
    assert read_image_header(payload, max_header_bytes=PNG_HEADER_BYTES).width == 7
    with pytest.raises(ImageHeaderError, match="^image_header_limit_exceeded$"):
        read_image_header(payload, max_header_bytes=PNG_HEADER_BYTES - 1)


def test_all_truncated_jpeg_boundaries_fail_closed() -> None:
    payload = jpeg(trailer=b"")
    for cut in range(len(payload)):
        with pytest.raises(ImageHeaderError) as excinfo:
            read_image_header(payload[:cut])
        assert excinfo.value.category in {
            "image_header_truncated",
            "jpeg_frame_length_invalid",
        }
    assert read_image_header(payload, max_header_bytes=len(payload)).width == 7
    with pytest.raises(ImageHeaderError, match="^image_header_limit_exceeded$"):
        read_image_header(payload, max_header_bytes=len(payload) - 1)


@pytest.mark.parametrize("width,height", [(0, 5), (7, 0), (0, 0)])
def test_zero_sized_images_fail_closed(width, height) -> None:
    with pytest.raises(ImageHeaderError, match="^png_dimensions_invalid$"):
        read_png_header(png(width=width, height=height))
    with pytest.raises(ImageHeaderError, match="^jpeg_dimensions_invalid$"):
        read_jpeg_header(jpeg(width=width, height=height))


def test_png_pixel_overflow_is_checked_by_division() -> None:
    with pytest.raises(ImageHeaderError, match="^image_pixel_limit_exceeded$"):
        read_png_header(png(width=1 << 30, height=1 << 30))
    with pytest.raises(ImageHeaderError, match="^image_pixel_limit_exceeded$"):
        read_jpeg_header(jpeg(width=65535, height=65535), max_pixels=1000)


@pytest.mark.parametrize("chunk", [b"iHDR", b"PLTE", b"IDAT", b"\x00\x00\x00\x00"])
def test_png_requires_ihdr_first(chunk) -> None:
    with pytest.raises(ImageHeaderError, match="^png_ihdr_missing$"):
        read_png_header(png(chunk=chunk))


@pytest.mark.parametrize("length", [0, 12, 14, 0xFFFFFFFF])
def test_png_invalid_ihdr_length_fails_closed(length) -> None:
    with pytest.raises(ImageHeaderError, match="^png_ihdr_length_invalid$"):
        read_png_header(png(length=length))


def test_png_crc_mismatch_fails_closed() -> None:
    with pytest.raises(ImageHeaderError, match="^png_ihdr_crc_mismatch$"):
        read_png_header(png(crc=0))


@pytest.mark.parametrize("color", [1, 5, 7, 255])
def test_png_unsupported_color_type_fails_closed(color) -> None:
    with pytest.raises(ImageHeaderError, match="^png_color_type_unsupported$"):
        read_png_header(png(color=color))


@pytest.mark.parametrize("color,depth", [(0, 0), (0, 3), (2, 4), (3, 16), (6, 1)])
def test_png_unsupported_bit_depth_fails_closed(color, depth) -> None:
    with pytest.raises(ImageHeaderError, match="^png_bit_depth_unsupported$"):
        read_png_header(png(color=color, depth=depth))


@pytest.mark.parametrize("field", ["compression", "filtering"])
def test_png_unsupported_methods_fail_closed(field) -> None:
    with pytest.raises(ImageHeaderError, match="^png_method_unsupported$"):
        read_png_header(png(**{field: 1}))


@pytest.mark.parametrize("interlace", [2, 255])
def test_png_unsupported_interlace_fails_closed(interlace) -> None:
    with pytest.raises(ImageHeaderError, match="^png_interlace_unsupported$"):
        read_png_header(png(interlace=interlace))


def test_jpeg_without_a_frame_header_fails_closed() -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_frame_header_missing$"):
        read_image_header(b"\xff\xd8" + APP0 + b"\xff\xd9")
    with pytest.raises(ImageHeaderError, match="^jpeg_frame_header_missing$"):
        read_image_header(b"\xff\xd8" + APP0 + b"\xff\xda\x00\x08......")


def test_jpeg_nested_start_of_image_fails_closed() -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_marker_unexpected$"):
        read_image_header(b"\xff\xd8\xff\xd8" + sof())


@pytest.mark.parametrize("length", [0, 1])
def test_jpeg_invalid_segment_length_fails_closed(length) -> None:
    payload = b"\xff\xd8\xff\xe0" + struct.pack(">H", length) + sof()
    with pytest.raises(ImageHeaderError, match="^jpeg_segment_length_invalid$"):
        read_image_header(payload)


@pytest.mark.parametrize("length", [2, 7, 10, 0xFFFF])
def test_jpeg_invalid_frame_length_fails_closed(length) -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_frame_length_invalid$"):
        read_image_header(jpeg(length=length))


@pytest.mark.parametrize("precision", [0, 17, 255])
def test_jpeg_unsupported_precision_fails_closed(precision) -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_precision_unsupported$"):
        read_jpeg_header(jpeg(precision=precision))


def test_jpeg_zero_component_frame_fails_closed() -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_component_count_invalid$"):
        read_jpeg_header(jpeg(components=0))


@pytest.mark.parametrize("payload", [b"\xff\xd8\x00\xc0", b"\xff\xd8\xff\x00"])
def test_jpeg_malformed_marker_fails_closed(payload) -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_marker_invalid$"):
        read_image_header(payload + b"\x00" * 32)


def test_jpeg_marker_fill_run_is_bounded() -> None:
    with pytest.raises(ImageHeaderError, match="^jpeg_marker_fill_limit_exceeded$"):
        read_image_header(b"\xff\xd8" + b"\xff" * 4096)


def test_jpeg_marker_loop_is_bounded() -> None:
    restarts = b"\xff\xd0" * (DEFAULT_MAX_JPEG_MARKERS + 8)
    with pytest.raises(ImageHeaderError, match="^jpeg_marker_limit_exceeded$"):
        read_image_header(b"\xff\xd8" + restarts + sof())
    with pytest.raises(ImageHeaderError, match="^jpeg_marker_limit_exceeded$"):
        read_image_header(b"\xff\xd8" + b"\xff\xd0" * 8 + sof(), max_jpeg_markers=4)


def test_jpeg_segment_skipping_stays_inside_the_header_budget() -> None:
    padded = jpeg(prefix=segment(0xE1, b"\x00" * 4096))
    assert read_image_header(padded).width == 7
    with pytest.raises(ImageHeaderError, match="^image_header_limit_exceeded$"):
        read_image_header(padded, max_header_bytes=1024)


@pytest.mark.parametrize("payload_factory", [png, jpeg])
def test_seekable_streams_are_restored_after_success(payload_factory) -> None:
    stream = io.BytesIO(b"pad" + payload_factory())
    stream.seek(3)
    assert read_image_header(stream).width == 7
    assert stream.tell() == 3
    assert not stream.closed


@pytest.mark.parametrize("payload_factory", [png, jpeg])
def test_seekable_streams_are_restored_after_failure(payload_factory) -> None:
    stream = io.BytesIO(payload_factory()[:6])
    with pytest.raises(ImageHeaderError):
        read_image_header(stream)
    assert stream.tell() == 0
    assert not stream.closed


def test_short_reads_are_supported_without_overreading() -> None:
    payload = png()
    stream = ShortStream(payload, chunk=3, boundary=PNG_HEADER_BYTES)
    assert read_image_header(stream).width == 7
    assert stream.stream.tell() == PNG_HEADER_BYTES
    assert max(stream.requests) <= PNG_HEADER_BYTES


def test_nonseekable_streams_are_never_rewound() -> None:
    class NoSeek(ShortStream):
        def seekable(self):
            return False

        def seek(self, *args):
            pytest.fail("a nonseekable stream must not be rewound")

    stream = NoSeek(png(), boundary=PNG_HEADER_BYTES)
    assert read_image_header(stream).height == 5
    assert stream.closed is False


def test_stream_contract_and_read_failures_are_value_free() -> None:
    class Overreader:
        def read(self, size):
            return b"\x00" * (size + 1)

    class Failing:
        def read(self, size):
            raise OSError("/var/phi/scan.png could not be read")

    with pytest.raises(ImageHeaderError, match="^image_stream_contract_error$"):
        read_image_header(Overreader())
    with pytest.raises(ImageHeaderError) as excinfo:
        read_image_header(Failing())
    assert excinfo.value.category == "image_stream_read_error"
    assert "phi" not in str(excinfo.value)


def test_invalid_api_arguments_are_reported_separately() -> None:
    with pytest.raises(TypeError, match="^source must be bytes or a binary stream$"):
        read_image_header(object())
    for kwargs in (
        {"max_header_bytes": 0},
        {"max_header_bytes": True},
        {"max_pixels": -1},
        {"max_jpeg_markers": 0},
    ):
        with pytest.raises(ValueError, match="must be a positive integer$"):
            read_image_header(png(), **kwargs)
    assert DEFAULT_MAX_IMAGE_HEADER_BYTES > PNG_HEADER_BYTES


def test_headers_are_immutable_and_carry_no_pixels() -> None:
    header = read_image_header(png())
    with pytest.raises(FrozenInstanceError):
        header.width = 1  # type: ignore[misc]
    assert set(asdict(header)) == {
        "image_format",
        "width",
        "height",
        "bit_depth",
        "component_count",
        "color_type",
        "progressive",
    }


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_pillow_generated_png_end_to_end(tmp_path, size, mode) -> None:
    from PIL import Image

    path = tmp_path / "synthetic.png"
    Image.new(mode, size).save(path, format="PNG")
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        header = read_image_header(stream)
        assert stream.tell() == 0
    assert (header.width, header.height) == expected == size
    assert header.image_format is ImageFormat.PNG


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("progressive", [False, True])
def test_pillow_generated_jpeg_end_to_end(tmp_path, size, progressive) -> None:
    from PIL import Image

    path = tmp_path / "synthetic.jpg"
    Image.new("RGB", size).save(path, format="JPEG", progressive=progressive)
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        header = read_image_header(stream)
        assert stream.tell() == 0
    assert (header.width, header.height) == expected == size
    assert (header.bit_depth, header.component_count) == (8, 3)
    assert header.progressive is progressive


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

    def close(self) -> None:
        raise AssertionError("caller-owned streams must not be closed")
