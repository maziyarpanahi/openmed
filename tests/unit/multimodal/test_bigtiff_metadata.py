"""Offline geometry, hostile declarations and privacy controls for BigTIFF."""

import io
import json
import struct
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from openmed.multimodal.bigtiff_metadata import (
    BigTiffLimits,
    BigTiffProfile,
    read_bigtiff_metadata,
)
from openmed.multimodal.tiff_metadata import TiffMetadataError, read_tiff_metadata
from tests.fixtures.bigtiff_pyramid import changed_entry, entry_offset, pyramid


def read(data, *, subifds=False, **kwargs):
    return read_bigtiff_metadata(
        data,
        profile=BigTiffProfile.SUBIFDS if subifds else BigTiffProfile.IFD_CHAIN,
        source_size=len(data),
        **kwargs,
    )


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("subifds", [False, True])
def test_deterministic_geometry_without_pixel_decode(order, subifds):
    data, header_end = pyramid(order=order, subifds=subifds)
    limits = BigTiffLimits(max_header_bytes=header_end)
    result = read(data, subifds=subifds, limits=limits)
    assert result == read(data, subifds=subifds, limits=limits)
    assert result.byte_order.value == ("little" if order == "<" else "big")
    assert result.total_tiles == 5
    assert [asdict(level) for level in result.levels] == [
        dict(
            index=0,
            width=8,
            height=8,
            tile_width=4,
            tile_height=4,
            columns=2,
            rows=2,
            tile_count=4,
            compression="uncompressed",
        ),
        dict(
            index=1,
            width=4,
            height=4,
            tile_width=4,
            tile_height=4,
            columns=1,
            rows=1,
            tile_count=1,
            compression="uncompressed",
        ),
    ]


@pytest.mark.parametrize(
    "compression,category",
    [
        (1, "uncompressed"),
        (5, "lzw"),
        (7, "jpeg"),
        (8, "deflate"),
        (32946, "deflate"),
    ],
)
def test_compression_is_a_declared_category_only(compression, category):
    assert read(pyramid(compression=compression)[0]).levels[0].compression == category


def test_shared_python_swift_manifests():
    fixture = Path("tests/fixtures/multimodal/bigtiff_pyramids.json")
    for row in json.loads(fixture.read_text()):
        data = bytes.fromhex(row["hex"])
        result = read_bigtiff_metadata(
            data,
            profile=row["profile"],
            source_size=len(data),
            limits=BigTiffLimits(max_header_bytes=row["header_bytes"]),
        )
        assert json.loads(json.dumps(asdict(result))) == row["expected"]


def test_all_truncated_metadata_prefixes_fail_closed():
    data, header_end = pyramid()
    for end in range(header_end):
        with pytest.raises(TiffMetadataError):
            read_bigtiff_metadata(
                io.BytesIO(data[:end]),
                profile=BigTiffProfile.IFD_CHAIN,
                source_size=len(data),
            )


@pytest.mark.parametrize(
    "limits,category",
    [
        (BigTiffLimits(max_header_bytes=16), "header_limit_or_truncated"),
        (BigTiffLimits(max_ifd_entries=2), "entry_limit_exceeded"),
        (BigTiffLimits(max_directories=1), "directory_limit_exceeded"),
        (BigTiffLimits(max_levels=1), "level_limit_exceeded"),
        (BigTiffLimits(max_tiles=4), "tile_limit_exceeded"),
        (BigTiffLimits(max_pixels=63), "pixel_limit_exceeded"),
        (BigTiffLimits(max_offset=16), "offset_limit_exceeded"),
    ],
)
def test_resource_ceilings(limits, category):
    with pytest.raises(TiffMetadataError, match="^bigtiff_" + category + "$"):
        read(pyramid()[0], limits=limits)


@pytest.mark.parametrize(
    "tag,changes,category",
    [
        (256, {"value": 0}, "geometry_invalid"),
        (256, {"value": 9, "level": 1}, "pyramid_geometry_inconsistent"),
        (259, {"value": 999}, "compression_unsupported"),
        (254, {"value": 0, "level": 1}, "layout_unsupported"),
        (284, {"value": 2}, "layout_unsupported"),
        (277, {"value": 65}, "layout_unsupported"),
        (324, {"count": 3}, "tile_table_inconsistent"),
        (325, {"count": 5}, "tile_table_inconsistent"),
        (324, {"count": 2**64 - 1}, "offset_overflow"),
        (324, {"value": 2**64 - 8}, "offset_overflow"),
        (324, {"value": 17}, "value_offset_invalid"),
        (324, {"kind": 2}, "value_type_unsupported"),
        (324, {"kind": 18}, "value_type_unsupported"),
        (256, {"count": 2}, "value_count_limit_exceeded"),
        (325, {"count": 0}, "value_count_invalid"),
        (324, {"value": 16}, "tile_range_invalid"),
    ],
)
def test_malformed_entries(tag, changes, category):
    with pytest.raises(TiffMetadataError, match="^bigtiff_" + category + "$"):
        read(changed_entry(pyramid()[0], tag, **changes))


@pytest.mark.parametrize(
    "offset,size,category",
    [
        (2**64 - 1, 4, "offset_overflow"),
        (16, 4, "range_overlap"),
        (100_000, 4, "range_out_of_source"),
        (0, 4, "tile_range_invalid"),
        (600, 0, "tile_range_invalid"),
    ],
)
def test_hostile_tile_ranges(offset, size, category):
    data = changed_entry(pyramid()[0], 324, level=1, value=offset)
    data = changed_entry(data, 325, level=1, value=size)
    with pytest.raises(TiffMetadataError, match="^bigtiff_" + category + "$"):
        read(data)


def test_directory_cycles_and_duplicate_children():
    data = bytearray(pyramid()[0])
    struct.pack_into("<Q", data, 16 + 8 + 10 * 20, 16)
    with pytest.raises(TiffMetadataError, match="bigtiff_directory_cycle"):
        read(bytes(data))
    data, _ = pyramid(subifds=True, dimensions=((16, 16), (8, 8), (4, 4)))
    base = struct.unpack_from("<Q", data, entry_offset(data, 330) + 12)[0]
    payload = bytearray(data)
    struct.pack_into("<Q", payload, base + 8, 272)
    with pytest.raises(TiffMetadataError, match="bigtiff_directory_cycle"):
        read(bytes(payload), subifds=True)


def test_subifd_profile_rejects_chain_and_chain_rejects_subifds():
    with pytest.raises(TiffMetadataError, match="bigtiff_layout_unsupported"):
        read(pyramid()[0], subifds=True)
    with pytest.raises(TiffMetadataError, match="bigtiff_layout_unsupported"):
        read(pyramid(subifds=True)[0])


@pytest.mark.parametrize("tag", [273, 278, 279, 32997, 32998, 65000])
def test_unsupported_layouts_are_explicit(tag):
    data = bytearray(pyramid()[0])
    struct.pack_into("<H", data, entry_offset(data, 277), tag)
    with pytest.raises(TiffMetadataError, match="bigtiff_layout_unsupported"):
        read(bytes(data))


def test_identifying_metadata_is_never_followed_or_reported():
    data = bytearray(pyramid()[0])
    start = entry_offset(data, 277)
    struct.pack_into("<HHQQ", data, start, 270, 2, 2**64 - 1, 2**64 - 1)
    data.extend(b"SYNTHETIC-PRIVATE-NAME /synthetic/private/slide.tif")
    result = read(bytes(data))
    assert "SYNTHETIC" not in repr(result)
    assert "private" not in json.dumps(asdict(result))


def test_stream_errors_do_not_retain_identifiers_or_paths():
    class Broken:
        def read(self, size):
            raise OSError("SYNTHETIC-PRIVATE-NAME /synthetic/private/slide.tif")

    with pytest.raises(TiffMetadataError) as caught:
        read_bigtiff_metadata(
            Broken(), source_size=1000, profile=BigTiffProfile.IFD_CHAIN
        )
    assert str(caught.value) == "tiff_stream_read_error"
    assert caught.value.__context__ is None
    assert "SYNTHETIC-PRIVATE" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("field", list(BigTiffLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [0, -1, True, 1.5, 2**64])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError, match="^bigtiff_limit_invalid$"):
        replace(BigTiffLimits(), **{field: value})


def test_unknown_profile_and_classic_tiff_remain_explicit():
    data = pyramid()[0]
    with pytest.raises(ValueError, match="^bigtiff_profile_unsupported$"):
        read_bigtiff_metadata(data, profile="SYNTHETIC-PRIVATE", source_size=len(data))
    with pytest.raises(TiffMetadataError, match="tiff_magic_invalid"):
        read_tiff_metadata(data)
    with pytest.raises(ValueError, match="bigtiff_source_size_invalid"):
        read_bigtiff_metadata(data, profile=BigTiffProfile.IFD_CHAIN, source_size=17)
