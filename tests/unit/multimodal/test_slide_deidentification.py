"""Offline structural privacy controls and TIFF-family negative controls."""

import hashlib
import json
import struct
from dataclasses import replace
from io import BytesIO

import pytest

from openmed.multimodal.slide_deidentification import (
    SlideDeidLimits,
    deidentify_tiff_slide,
)
from openmed.multimodal.tiff_metadata import TiffMetadataError
from tests.fixtures.multimodal.slides import SENTINELS, slide


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_remove_associated_images_and_descriptions_preserve_every_tile(big, endian):
    source, _ = slide(big=big, endian=endian)
    result = deidentify_tiff_slide(source)
    assert result.outcome == "deidentified"
    assert [d.role for d in result.directories] == [
        "level",
        "thumbnail",
        "level",
        "label",
        "macro",
    ]
    for sentinel in (*SENTINELS, b"SYNTHETIC_ORPHAN_IDENTIFIER"):
        assert sentinel not in result.data
        assert sentinel.decode(errors="replace") not in json.dumps(
            result.to_audit_report()
        )
        assert sentinel.decode(errors="replace") not in repr(result)
    second = deidentify_tiff_slide(result.data)
    assert second.outcome == "no_associated_images"
    assert second.data == result.data
    assert second.directories == tuple(
        d for d in result.directories if d.role == "level"
    )
    assert result.output_sha256 == hashlib.sha256(result.data).hexdigest()
    assert b"|AppMag=40|MPP=0.25" in result.data
    # Independent decoder confirms the rebuilt chain is a readable pyramid.
    # Pillow 12 does not recognize big-endian BigTIFF (its header detection
    # assumes little-endian magic); that variant has native parity coverage.
    if not (big and endian == ">"):
        from PIL import Image

        with Image.open(BytesIO(result.data)) as parsed:
            assert parsed.n_frames == 2
            assert parsed.size == (16, 16)
            assert set(parsed.get_flattened_data()) == {30}
            parsed.seek(1)
            assert parsed.size == (8, 8)
            assert set(parsed.get_flattened_data()) == {32}


@pytest.mark.parametrize("big", [False, True])
def test_no_associated_images_still_scrubs_description_private_tags_and_orphans(big):
    source, _ = slide(
        big=big,
        roles=("level", "level"),
        vendor=False,
        extra_tags={65000: (2, len(SENTINELS[0]), SENTINELS[0])},
    )
    result = deidentify_tiff_slide(source)
    assert result.outcome == "no_associated_images"
    assert all(d.role == "level" for d in result.directories)
    assert all(s not in result.data for s in SENTINELS)


@pytest.mark.parametrize(
    "roles,vendor",
    [
        (("level", "unknown"), True),
        (("level", "label"), False),
        (("label",), True),
        (("level", "label", "level"), True),
        (("level", "label", "label"), True),
        (("level", "thumbnail", "thumbnail"), True),
    ],
)
def test_unknown_or_unexpected_associated_images_refused(roles, vendor):
    source, _ = slide(roles=roles, vendor=vendor)
    with pytest.raises(TiffMetadataError, match="slide_"):
        deidentify_tiff_slide(source)


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_cycles_bounds_aliases_and_incomplete_payloads_refused(big, endian):
    source, positions = slide(big=big, endian=endian)
    wf = "Q" if big else "I"
    word = 8 if big else 4
    cycle = bytearray(source)
    cycle[positions[-1]["next"] : positions[-1]["next"] + word] = struct.pack(
        endian + wf, positions[0]["ifd"]
    )
    with pytest.raises(TiffMetadataError, match="slide_directory_cycle"):
        deidentify_tiff_slide(bytes(cycle))
    for limits, code in [
        (SlideDeidLimits(max_file_bytes=10), "file_limit"),
        (SlideDeidLimits(max_directories=2), "directory_limit"),
        (SlideDeidLimits(max_ifd_entries=2), "entry_limit"),
        (SlideDeidLimits(max_value_bytes=4), "value_limit"),
        (SlideDeidLimits(max_blocks=2), "block_limit"),
    ]:
        with pytest.raises(TiffMetadataError, match="slide_" + code):
            deidentify_tiff_slide(source, limits=limits)
    alias = bytearray(source)
    at = positions[0]["entries"][324] + 4 + word
    alias[at : at + word] = struct.pack(endian + wf, positions[0]["ifd"])
    with pytest.raises(TiffMetadataError, match="slide_range_overlap"):
        deidentify_tiff_slide(bytes(alias))
    # Every truncated prefix must refuse; no partially sanitized result escapes.
    for n in range(positions[-1]["next"] + word):
        with pytest.raises(TiffMetadataError):
            deidentify_tiff_slide(source[:n])


def test_hidden_ifds_duplicate_tags_and_jpeg_metadata_refused():
    for tags in [
        {330: (4, 1, struct.pack("<I", 8))},
        {347: (7, 9, b"\xff\xd8\xff\xfe\x00\x03X\xff\xd9")},
    ]:
        source, _ = slide(extra_tags=tags)
        with pytest.raises(TiffMetadataError):
            deidentify_tiff_slide(source)
    source, locations = slide()
    duplicate = bytearray(source)
    at = locations[0]["entries"][257]
    duplicate[at : at + 2] = struct.pack("<H", 256)
    with pytest.raises(TiffMetadataError, match="slide_tag_duplicate"):
        deidentify_tiff_slide(bytes(duplicate))


def test_numeric_allowlist_rejects_text_and_duplicate_keys():
    from openmed.multimodal.slide_deidentification import _description

    raw = b"Aperio|MPP=NAME|AppMag=40SYNTHETIC|Date=20000101|MPP=0.5\0"
    assert _description(raw, True) == b"Aperio\n|MPP=0.5\0"
    with pytest.raises(TiffMetadataError, match="slide_description_duplicate_key"):
        _description(b"Aperio|MPP=0.25|MPP=0.5", True)
    with pytest.raises(ValueError, match="slide_limits_invalid"):
        replace(SlideDeidLimits(), max_blocks=True)


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("deflate", [False, True])
def test_multiple_tiles_and_compressed_payloads_preserved(big, endian, deflate):
    import zlib

    source, _ = slide(big=big, endian=endian, multiple_tiles=True, deflate=deflate)
    result = deidentify_tiff_slide(source)
    blocks = [bytes([value]) * 256 for value in range(30, 34)]
    if deflate:
        blocks = [zlib.compress(block) for block in blocks]
    assert result.directories[0].block_sha256 == tuple(
        hashlib.sha256(b).hexdigest() for b in blocks
    )
    assert result.directories[0].block_count == 4
    cleaned = deidentify_tiff_slide(result.data)
    assert cleaned.directories == tuple(
        d for d in result.directories if d.role == "level"
    )
    assert cleaned.data == result.data
    if not (big and endian == ">"):
        from PIL import Image

        with Image.open(BytesIO(result.data)) as parsed:
            assert parsed.size == (32, 32)
            assert parsed.getpixel((0, 0)) == 30
            assert parsed.getpixel((16, 0)) == 31
            assert parsed.getpixel((0, 16)) == 32
            assert parsed.getpixel((16, 16)) == 33


def test_shared_native_vectors_match_python():
    from pathlib import Path

    rows = json.loads(
        Path("tests/fixtures/multimodal/slide_deid_vectors.json").read_text()
    )
    for row in rows:
        result = deidentify_tiff_slide(bytes.fromhex(row["source_hex"]))
        assert result.output_sha256 == row["output_sha256"]
        assert result.to_audit_report() == row["audit"]


@pytest.mark.parametrize("big", [False, True])
def test_deterministic_hostile_byte_sweep_has_controlled_failures(big):
    source, _ = slide(big=big)
    for offset in range(len(source)):
        changed = bytearray(source)
        changed[offset] ^= 0x80
        try:
            result = deidentify_tiff_slide(bytes(changed))
        except TiffMetadataError as exc:
            assert str(exc) == exc.category and exc.category.startswith("slide_")
        else:
            assert b"SYNTHETIC_OPERATOR_DATE_20000101" not in result.data
            assert b"SYNTHETIC_ORPHAN_IDENTIFIER" not in result.data
            assert all(s not in result.data for s in SENTINELS)


def test_unmarked_secondary_generic_tiled_image_is_ambiguous_and_refused():
    source, _ = slide(
        roles=("level", "level"),
        vendor=False,
        extra_tags={254: (4, 1, struct.pack("<I", 0))},
    )
    with pytest.raises(TiffMetadataError, match="slide_layout_unknown"):
        deidentify_tiff_slide(source)
