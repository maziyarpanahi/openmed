"""Synthetic unit and file-level tests for classic TIFF header preflight."""

from __future__ import annotations

import io
import struct
from dataclasses import FrozenInstanceError

import pytest

from openmed.multimodal.tiff_metadata import (
    DEFAULT_MAX_IFD_ENTRIES,
    DEFAULT_MAX_TIFF_HEADER_BYTES,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
    TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH,
    TAG_PHOTOMETRIC,
    TAG_PLANAR_CONFIGURATION,
    TAG_SAMPLES_PER_PIXEL,
    TiffByteOrder,
    TiffMetadataError,
    read_tiff_metadata,
)

SHORT = 3
LONG = 4
RATIONAL = 5

GRAY_TAGS = {
    TAG_IMAGE_WIDTH: (SHORT, (7,)),
    TAG_IMAGE_LENGTH: (SHORT, (5,)),
    TAG_BITS_PER_SAMPLE: (SHORT, (8,)),
    TAG_PHOTOMETRIC: (SHORT, (1,)),
}
RGB_TAGS = {
    TAG_IMAGE_WIDTH: (LONG, (7,)),
    TAG_IMAGE_LENGTH: (LONG, (5,)),
    TAG_BITS_PER_SAMPLE: (SHORT, (8, 8, 8)),
    TAG_SAMPLES_PER_PIXEL: (SHORT, (3,)),
    TAG_PHOTOMETRIC: (SHORT, (2,)),
    TAG_COMPRESSION: (SHORT, (1,)),
    TAG_PLANAR_CONFIGURATION: (SHORT, (1,)),
}

_FORMATS = {1: "B", SHORT: "H", LONG: "I", RATIONAL: "I"}
_SIZES = {1: 1, SHORT: 2, LONG: 4, RATIONAL: 8}


def tiff(
    tags=None,
    *,
    order="<",
    magic=42,
    ifd_offset=8,
    entry_count=None,
    next_ifd=0,
    trailer=b"",
):
    items = sorted((tags if tags is not None else GRAY_TAGS).items())
    declared = len(items) if entry_count is None else entry_count
    data_offset = ifd_offset + 2 + len(items) * 12 + 4
    blobs = b""
    packed = b""
    for tag, (value_type, values) in items:
        payload = struct.pack(f"{order}{len(values)}{_FORMATS[value_type]}", *values)
        span = len(values) * _SIZES[value_type]
        if span <= 4:
            raw = payload.ljust(4, b"\x00")
        else:
            raw = struct.pack(f"{order}I", data_offset + len(blobs))
            blobs += payload
        packed += struct.pack(f"{order}HHI", tag, value_type, len(values)) + raw
    header = (b"II" if order == "<" else b"MM") + struct.pack(
        f"{order}HI", magic, ifd_offset
    )
    body = (
        struct.pack(f"{order}H", declared) + packed + struct.pack(f"{order}I", next_ifd)
    )
    return header + b"\x00" * (ifd_offset - 8) + body + blobs + trailer


def replaced(tags, overrides):
    merged = dict(tags)
    merged.update(overrides)
    return merged


@pytest.mark.parametrize("order,expected", [("<", "little"), (">", "big")])
def test_both_byte_orders_are_read(order, expected) -> None:
    metadata = read_tiff_metadata(tiff(order=order))
    assert metadata.byte_order.value == expected
    assert (metadata.width, metadata.height) == (7, 5)


@pytest.mark.parametrize("order", ["<", ">"])
def test_grayscale_inline_values(order) -> None:
    metadata = read_tiff_metadata(tiff(order=order))
    assert metadata.samples_per_pixel == 1
    assert metadata.bits_per_sample == (8,)
    assert metadata.photometric == 1
    assert (metadata.compression, metadata.planar_configuration) == (1, 1)
    assert metadata.ifd_entry_count == 4


@pytest.mark.parametrize("order", ["<", ">"])
def test_rgb_values_resolved_through_an_offset(order) -> None:
    metadata = read_tiff_metadata(tiff(RGB_TAGS, order=order))
    assert metadata.samples_per_pixel == 3
    assert metadata.bits_per_sample == (8, 8, 8)
    assert metadata.photometric == 2
    assert metadata.ifd_entry_count == 7


def test_short_and_long_dimension_types_are_both_accepted() -> None:
    tags = replaced(GRAY_TAGS, {TAG_IMAGE_WIDTH: (LONG, (300,))})
    assert read_tiff_metadata(tiff(tags)).width == 300


def test_optional_tags_fall_back_to_tiff_defaults() -> None:
    tags = {
        TAG_IMAGE_WIDTH: (SHORT, (7,)),
        TAG_IMAGE_LENGTH: (SHORT, (5,)),
        TAG_PHOTOMETRIC: (SHORT, (1,)),
    }
    metadata = read_tiff_metadata(tiff(tags))
    assert metadata.samples_per_pixel == 1
    assert metadata.bits_per_sample == (1,)
    assert (metadata.compression, metadata.planar_configuration) == (1, 1)


def test_unknown_tags_are_skipped_without_following_their_offsets() -> None:
    tags = replaced(GRAY_TAGS, {700: (RATIONAL, (0xDEADBEEF,)), 305: (SHORT, (1,))})
    metadata = read_tiff_metadata(tiff(tags))
    assert (metadata.width, metadata.height) == (7, 5)
    assert metadata.ifd_entry_count == 6


def test_a_nonzero_next_ifd_offset_is_not_followed() -> None:
    metadata = read_tiff_metadata(tiff(next_ifd=0xFFFFFFFF))
    assert metadata.width == 7


def test_ifd_may_start_after_a_padded_header() -> None:
    assert read_tiff_metadata(tiff(ifd_offset=64)).width == 7


@pytest.mark.parametrize("marker", [b"IM", b"MI", b"\x00\x00", b"BM"])
def test_invalid_byte_order_fails_closed(marker) -> None:
    payload = marker + tiff()[2:]
    with pytest.raises(TiffMetadataError, match="^tiff_byte_order_invalid$"):
        read_tiff_metadata(payload)


@pytest.mark.parametrize("magic", [0, 41, 43, 0x2B])
def test_invalid_magic_fails_closed(magic) -> None:
    with pytest.raises(TiffMetadataError, match="^tiff_magic_invalid$"):
        read_tiff_metadata(tiff(magic=magic))


@pytest.mark.parametrize("offset", [0, 1, 7])
def test_ifd_offset_inside_the_header_fails_closed(offset) -> None:
    payload = bytearray(tiff())
    struct.pack_into("<I", payload, 4, offset)
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_offset_invalid$"):
        read_tiff_metadata(bytes(payload))


def test_ifd_offset_past_the_prefix_fails_closed() -> None:
    payload = bytearray(tiff())
    struct.pack_into("<I", payload, 4, 0xFFFFFF)
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_truncated$"):
        read_tiff_metadata(bytes(payload))


def test_empty_ifd_fails_closed() -> None:
    payload = bytearray(tiff())
    struct.pack_into("<H", payload, 8, 0)
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_empty$"):
        read_tiff_metadata(bytes(payload))


def test_oversized_ifd_fails_closed_before_reading_entries() -> None:
    payload = bytearray(tiff())
    struct.pack_into("<H", payload, 8, 0xFFFF)
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_entry_limit_exceeded$"):
        read_tiff_metadata(bytes(payload))
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_entry_limit_exceeded$"):
        read_tiff_metadata(tiff(), max_ifd_entries=3)
    assert DEFAULT_MAX_IFD_ENTRIES > 0


def test_declared_entry_count_beyond_the_buffer_fails_closed() -> None:
    with pytest.raises(TiffMetadataError, match="^tiff_ifd_truncated$"):
        read_tiff_metadata(tiff(entry_count=len(GRAY_TAGS) + 2))


def test_duplicate_tags_fail_closed() -> None:
    payload = bytearray(tiff(RGB_TAGS))
    struct.pack_into("<H", payload, 10 + 12, TAG_IMAGE_WIDTH)
    with pytest.raises(TiffMetadataError, match="^tiff_tag_duplicate$"):
        read_tiff_metadata(bytes(payload))


@pytest.mark.parametrize("value_type", [0, 2, 5, 7, 12, 65535])
def test_unsupported_value_types_fail_closed(value_type) -> None:
    payload = bytearray(tiff())
    struct.pack_into("<H", payload, 10 + 2, value_type)
    with pytest.raises(TiffMetadataError, match="^tiff_value_type_unsupported$"):
        read_tiff_metadata(bytes(payload))


def test_zero_value_count_fails_closed() -> None:
    payload = bytearray(tiff())
    struct.pack_into("<I", payload, 10 + 4, 0)
    with pytest.raises(TiffMetadataError, match="^tiff_value_count_invalid$"):
        read_tiff_metadata(bytes(payload))


def test_value_length_overflow_fails_closed() -> None:
    payload = bytearray(tiff())
    struct.pack_into("<H", payload, 10 + 2, LONG)
    struct.pack_into("<I", payload, 10 + 4, 0xFFFFFFFF)
    with pytest.raises(TiffMetadataError, match="^tiff_value_length_overflow$"):
        read_tiff_metadata(bytes(payload))


def test_scalar_tags_reject_multi_valued_entries() -> None:
    tags = replaced(RGB_TAGS, {TAG_PHOTOMETRIC: (SHORT, (1, 2))})
    with pytest.raises(TiffMetadataError, match="^tiff_value_count_invalid$"):
        read_tiff_metadata(tiff(tags))
    tags = replaced(GRAY_TAGS, {TAG_IMAGE_WIDTH: (SHORT, (7, 7))})
    with pytest.raises(TiffMetadataError, match="^tiff_value_count_invalid$"):
        read_tiff_metadata(tiff(tags))


@pytest.mark.parametrize("offset", [0, 7])
def test_value_offset_inside_the_header_fails_closed(offset) -> None:
    payload = bytearray(tiff(RGB_TAGS))
    entry = 10 + sorted(RGB_TAGS).index(TAG_BITS_PER_SAMPLE) * 12
    struct.pack_into("<I", payload, entry + 8, offset)
    with pytest.raises(TiffMetadataError, match="^tiff_value_offset_invalid$"):
        read_tiff_metadata(bytes(payload))


def test_value_offset_pointing_at_the_ifd_fails_closed() -> None:
    payload = bytearray(tiff(RGB_TAGS))
    entry = 10 + sorted(RGB_TAGS).index(TAG_BITS_PER_SAMPLE) * 12
    struct.pack_into("<I", payload, entry + 8, 8)
    with pytest.raises(TiffMetadataError, match="^tiff_value_offset_invalid$"):
        read_tiff_metadata(bytes(payload))


def test_repeated_value_offsets_are_reported_as_a_cycle() -> None:
    tags = replaced(RGB_TAGS, {TAG_IMAGE_WIDTH: (LONG, (7, 7))})
    payload = bytearray(tiff(tags, trailer=b"\x00" * 16))
    order = sorted(tags)
    width_entry = 10 + order.index(TAG_IMAGE_WIDTH) * 12
    bits_entry = 10 + order.index(TAG_BITS_PER_SAMPLE) * 12
    shared = struct.unpack_from("<I", payload, bits_entry + 8)[0]
    struct.pack_into("<I", payload, width_entry + 8, shared)
    with pytest.raises(TiffMetadataError, match="^tiff_value_offset_cycle$"):
        read_tiff_metadata(bytes(payload))


def test_value_offset_past_the_prefix_fails_closed() -> None:
    payload = bytearray(tiff(RGB_TAGS))
    entry = 10 + sorted(RGB_TAGS).index(TAG_BITS_PER_SAMPLE) * 12
    struct.pack_into("<I", payload, entry + 8, 0xFFFF)
    with pytest.raises(TiffMetadataError, match="^tiff_value_truncated$"):
        read_tiff_metadata(bytes(payload))


def test_all_truncated_prefixes_fail_closed() -> None:
    payload = tiff(RGB_TAGS)
    for cut in range(len(payload)):
        with pytest.raises(TiffMetadataError) as excinfo:
            read_tiff_metadata(payload[:cut])
        assert excinfo.value.category in {
            "tiff_header_truncated",
            "tiff_ifd_truncated",
            "tiff_value_truncated",
        }
    assert read_tiff_metadata(payload, max_header_bytes=len(payload)).width == 7
    with pytest.raises(TiffMetadataError, match="^tiff_value_truncated$"):
        read_tiff_metadata(payload, max_header_bytes=len(payload) - 1)


def test_missing_dimensions_fail_closed() -> None:
    for tag in (TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH):
        tags = {key: value for key, value in GRAY_TAGS.items() if key != tag}
        with pytest.raises(TiffMetadataError, match="^tiff_dimensions_missing$"):
            read_tiff_metadata(tiff(tags))


def test_zero_dimensions_fail_closed() -> None:
    for tag in (TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH):
        tags = replaced(GRAY_TAGS, {tag: (SHORT, (0,))})
        with pytest.raises(TiffMetadataError, match="^tiff_dimensions_invalid$"):
            read_tiff_metadata(tiff(tags))


def test_pixel_limit_is_checked_by_division() -> None:
    tags = replaced(
        GRAY_TAGS,
        {
            TAG_IMAGE_WIDTH: (LONG, (100_000,)),
            TAG_IMAGE_LENGTH: (LONG, (100_000,)),
        },
    )
    with pytest.raises(TiffMetadataError, match="^tiff_pixel_limit_exceeded$"):
        read_tiff_metadata(tiff(tags))
    assert read_tiff_metadata(tiff(tags), max_pixels=10**10).width == 100_000


def test_missing_photometric_fails_closed() -> None:
    tags = {key: value for key, value in GRAY_TAGS.items() if key != TAG_PHOTOMETRIC}
    with pytest.raises(TiffMetadataError, match="^tiff_photometric_missing$"):
        read_tiff_metadata(tiff(tags))


@pytest.mark.parametrize("samples", [0, 65, 1000])
def test_invalid_samples_per_pixel_fails_closed(samples) -> None:
    tags = replaced(GRAY_TAGS, {TAG_SAMPLES_PER_PIXEL: (LONG, (samples,))})
    with pytest.raises(TiffMetadataError, match="^tiff_samples_per_pixel_invalid$"):
        read_tiff_metadata(tiff(tags))


def test_bits_per_sample_count_must_match_samples_per_pixel() -> None:
    tags = replaced(RGB_TAGS, {TAG_BITS_PER_SAMPLE: (SHORT, (8, 8))})
    with pytest.raises(TiffMetadataError, match="^tiff_bits_per_sample_count_invalid$"):
        read_tiff_metadata(tiff(tags))


@pytest.mark.parametrize("bits", [0, 65])
def test_invalid_bits_per_sample_fails_closed(bits) -> None:
    tags = replaced(GRAY_TAGS, {TAG_BITS_PER_SAMPLE: (SHORT, (bits,))})
    with pytest.raises(TiffMetadataError, match="^tiff_bits_per_sample_invalid$"):
        read_tiff_metadata(tiff(tags))


def test_seekable_streams_are_restored_and_never_closed() -> None:
    stream = io.BytesIO(b"pad" + tiff(RGB_TAGS))
    stream.seek(3)
    assert read_tiff_metadata(stream).width == 7
    assert stream.tell() == 3
    assert not stream.closed

    broken = io.BytesIO(tiff()[:6])
    with pytest.raises(TiffMetadataError, match="^tiff_header_truncated$"):
        read_tiff_metadata(broken)
    assert broken.tell() == 0
    assert not broken.closed


def test_short_and_nonseekable_streams_are_supported() -> None:
    payload = tiff(RGB_TAGS)

    class NoSeek(ShortStream):
        def seekable(self):
            return False

        def seek(self, *args):
            pytest.fail("a nonseekable stream must not be rewound")

    stream = NoSeek(payload)
    assert read_tiff_metadata(stream).bits_per_sample == (8, 8, 8)
    assert stream.closed is False
    assert max(stream.requests) <= DEFAULT_MAX_TIFF_HEADER_BYTES


def test_stream_contract_and_read_failures_are_value_free() -> None:
    class Overreader:
        def read(self, size):
            return b"\x00" * (size + 1)

    class Failing:
        def read(self, size):
            raise OSError("/var/phi/scan.tif could not be read")

    with pytest.raises(TiffMetadataError, match="^tiff_stream_contract_error$"):
        read_tiff_metadata(Overreader())
    with pytest.raises(TiffMetadataError) as excinfo:
        read_tiff_metadata(Failing())
    assert excinfo.value.category == "tiff_stream_read_error"
    assert "phi" not in str(excinfo.value)


def test_invalid_api_arguments_are_reported_separately() -> None:
    with pytest.raises(TypeError, match="^source must be bytes or a binary stream$"):
        read_tiff_metadata(object())
    for kwargs in (
        {"max_header_bytes": 0},
        {"max_header_bytes": True},
        {"max_pixels": -1},
        {"max_ifd_entries": 0},
    ):
        with pytest.raises(ValueError, match="must be a positive integer$"):
            read_tiff_metadata(tiff(), **kwargs)


def test_metadata_is_immutable_and_carries_no_pixels() -> None:
    metadata = read_tiff_metadata(tiff())
    with pytest.raises(FrozenInstanceError):
        metadata.width = 1  # type: ignore[misc]
    assert metadata.byte_order is TiffByteOrder.LITTLE


@pytest.mark.integration
@pytest.mark.parametrize("size", [(1, 1), (7, 5), (257, 129)])
@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA"])
def test_pillow_generated_tiff_end_to_end(tmp_path, size, mode) -> None:
    from PIL import Image

    path = tmp_path / "synthetic.tif"
    Image.new(mode, size).save(path, format="TIFF")
    with Image.open(path) as decoded:
        decoded.load()
        expected = decoded.size
    with path.open("rb") as stream:
        metadata = read_tiff_metadata(stream)
        assert stream.tell() == 0
    assert (metadata.width, metadata.height) == expected == size
    assert metadata.samples_per_pixel == len(metadata.bits_per_sample)
    assert metadata.bits_per_sample[0] == (1 if mode == "1" else 8)


@pytest.mark.integration
@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate"])
def test_pillow_compressed_tiff_headers(tmp_path, compression) -> None:
    from PIL import Image

    path = tmp_path / "compressed.tif"
    Image.new("RGB", (40, 24)).save(path, format="TIFF", compression=compression)
    metadata = read_tiff_metadata(path.read_bytes())
    assert (metadata.width, metadata.height) == (40, 24)
    assert metadata.samples_per_pixel == 3
    assert metadata.compression >= 1


class ShortStream:
    """A real byte stream that returns short reads and records every request."""

    def __init__(self, data: bytes, chunk: int = 5):
        self.stream = io.BytesIO(data)
        self.chunk = chunk
        self.requests: list[int] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        assert size >= 0
        self.requests.append(size)
        return self.stream.read(min(size, self.chunk))

    def close(self) -> None:
        raise AssertionError("caller-owned streams must not be closed")
