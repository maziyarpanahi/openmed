"""Hand-built offline ISO BMFF fixtures and negative controls."""

import json
import struct
from dataclasses import asdict

import pytest

from openmed.multimodal.iso_bmff import BmffPreflightError, preflight_iso_bmff
from openmed.multimodal.media_type import detect_media_type


def box(kind, payload=b"", *, extended=False, to_end=False):
    if extended:
        return struct.pack(">I4sQ", 1, kind, len(payload) + 16) + payload
    return struct.pack(">I4s", 0 if to_end else len(payload) + 8, kind) + payload


def full(kind, payload=b"", version=0, flags=0):
    return box(kind, bytes([version]) + flags.to_bytes(3, "big") + payload)


def ftyp(brand=b"heic", compatible=b"mif1", **kwargs):
    return box(b"ftyp", brand + b"\0" * 4 + compatible, **kwargs)


def heif(*, cycle=False, wide=False, sentinel=b"Synthetic Patient 73", metadata=True):
    width = 4 if wide else 2
    integer = lambda value: value.to_bytes(width, "big")
    entries = []
    # Primary, thumbnail, auxiliary/depth, burst image, EXIF and XMP.
    kinds = [b"hvc1"] * 4 + ([b"Exif", b"mime"] if metadata else [])
    for item, kind in enumerate(kinds, 1):
        payload = integer(item) + b"\0\0" + kind + sentinel + b"\0"
        if kind == b"mime":
            payload += b"application/rdf+xml\0"
        entries.append(full(b"infe", payload, version=3 if wide else 2))
    refs = box(b"thmb", integer(2) + b"\0\1" + integer(1))
    refs += box(b"auxl", integer(3) + b"\0\1" + integer(1))
    if cycle:
        refs += box(b"dimg", integer(1) + b"\0\1" + integer(2))
    # Thumbnail geometry comes first; primary geometry is property number two.
    props = box(
        b"ipco",
        full(b"ispe", struct.pack(">II", 320, 240))
        + full(b"ispe", struct.pack(">II", 4032, 3024)),
    )
    association = struct.pack(">I", 2) + integer(1) + b"\1"
    association += b"\x80\x02" if wide else b"\x82"
    association += integer(2) + b"\1" + (b"\x80\x01" if wide else b"\x81")
    props += full(b"ipma", association, version=int(wide), flags=int(wide))
    payload = full(b"pitm", integer(1), version=int(wide))
    payload += full(b"iinf", len(entries).to_bytes(2, "big") + b"".join(entries))
    payload += box(b"iprp", props)
    payload += full(b"iref", refs, version=int(wide))
    payload += box(b"grpl", full(b"brst", struct.pack(">IIII", 9, 2, 1, 4)))
    return ftyp() + full(b"meta", payload) + box(b"mdat", sentinel * 20)


def m4a(*, codec=b"mp4a", version=0, channels=2, rate=48000, duration=120000):
    time_format = ">QQIQHH" if version else ">II II HH"
    mdhd = full(b"mdhd", struct.pack(time_format, 0, 0, 48000, duration, 0, 0), version)
    hdlr = full(b"hdlr", b"\0" * 4 + b"soun" + b"\0" * 12 + b"Synthetic Name\0")
    entry = b"\0" * 6 + b"\0\1" + b"\0" * 8
    entry += struct.pack(">HHHHI", channels, 16, 0, 0, rate << 16)
    entry += full(b"esds", b"Synthetic codec data")
    stsd = full(b"stsd", struct.pack(">I", 1) + box(codec, entry))
    mdia = box(b"mdia", mdhd + hdlr + box(b"minf", box(b"stbl", stsd)))
    return (
        ftyp(b"M4A ", b"isom")
        + box(b"moov", box(b"trak", mdia))
        + box(b"mdat", b"Synthetic Samples")
    )


@pytest.mark.parametrize("wide", [False, True])
def test_heif_hand_checked_geometry_inventory_and_privacy(wide):
    source = heif(wide=wide)
    result = preflight_iso_bmff(source)
    assert asdict(result) == {
        "width": 4032,
        "height": 3024,
        "item_count": 6,
        "thumbnail_count": 1,
        "auxiliary_count": 1,
        "burst_count": 2,
        "has_exif": True,
        "has_xmp": True,
        "status": "decoder_required",
    }
    assert "Synthetic" not in json.dumps(asdict(result)) + repr(result)
    # Changing names, EXIF values and pixels cannot change the returned report.
    assert result == preflight_iso_bmff(
        heif(wide=wide, sentinel=b"Other private value")
    )


def test_payload_markers_are_not_metadata_presence_evidence():
    result = preflight_iso_bmff(
        heif(metadata=False, sentinel=b"Exifapplication/rdf+xmlSynthetic Payload")
    )
    assert not result.has_exif and not result.has_xmp
    assert result.item_count == 4


@pytest.mark.parametrize(
    "codec,category", [(b"mp4a", "mpeg4_audio"), (b"alac", "apple_lossless")]
)
@pytest.mark.parametrize("version", [0, 1])
def test_m4a_hand_checked_metadata(codec, category, version):
    result = preflight_iso_bmff(m4a(codec=codec, version=version))
    assert asdict(result) == {
        "codec_category": category,
        "channels": 2,
        "sample_rate_hz": 48000.0,
        "duration_seconds": 2.5,
        "status": "decoder_required",
        "notice": "non_diagnostic",
        "reviewer_confirmation_required": True,
    }
    assert "Synthetic" not in repr(result)


@pytest.mark.parametrize(
    "brand,mime",
    [
        (b"heic", "image/heic"),
        (b"mif1", "image/heif"),
        (b"msf1", "image/heif-sequence"),
        (b"hevc", "image/heic-sequence"),
        (b"M4A ", "audio/mp4"),
    ],
)
@pytest.mark.parametrize("extended", [False, True])
def test_detection_complete_file_type_box(brand, mime, extended):
    prefix = ftyp(brand, b"isom", extended=extended)
    assert detect_media_type(prefix) == mime
    assert detect_media_type(prefix[:-1]) is None


@pytest.mark.parametrize(
    "prefix",
    [
        ftyp(b"isom", b"mp42"),
        ftyp(b"heic", b"M4A "),
        ftyp(b"avif", b"mif1"),
        ftyp(b"heic", b"mif1", to_end=True),
        ftyp(b"isom", b"x" * 120 + b"heic"),
    ],
)
def test_unknown_or_ambiguous_brands_are_not_guessed(prefix):
    assert detect_media_type(prefix) is None


@pytest.mark.parametrize(
    "source,options,code",
    [
        (heif(cycle=True), {}, "bmff_item_cycle"),
        (heif(), {"max_depth": 3}, "bmff_depth_limit"),
        (heif(), {"max_boxes": 4}, "bmff_box_count_limit"),
        (heif(), {"max_declared_box_bytes": 100}, "bmff_declared_size_limit"),
        (heif()[:-1], {}, "bmff_truncated"),
        (
            ftyp() + struct.pack(">I4sQ", 1, b"free", (1 << 64) - 1),
            {},
            "bmff_size_overflow",
        ),
        (ftyp() + struct.pack(">I4s", 4, b"free"), {}, "bmff_box_size_invalid"),
        (heif(), {"max_pixels": 100}, "bmff_pixel_limit"),
        (m4a(codec=b"enca"), {}, "bmff_codec_unsupported"),
        (m4a(channels=0), {}, "bmff_metadata_invalid"),
        (m4a(rate=0), {}, "bmff_metadata_invalid"),
        (m4a(duration=(1 << 32) - 1), {}, "bmff_metadata_invalid"),
        (m4a() + box(b"moof"), {}, "bmff_layout_unsupported"),
        (ftyp(), {}, "bmff_metadata_missing_or_duplicate"),
    ],
)
def test_stable_refusals(source, options, code):
    with pytest.raises(BmffPreflightError) as error:
        preflight_iso_bmff(source, **options)
    assert str(error.value) == error.value.category == code
    assert error.value.__context__ is None


@pytest.mark.parametrize("factory", [heif, m4a])
def test_metadata_after_bound_is_not_searched(factory):
    original = factory()
    source = original[:20] + box(b"free", b"S" * 2048) + original[20:]
    with pytest.raises(BmffPreflightError, match="^bmff_metadata_out_of_bounds$"):
        preflight_iso_bmff(source, max_header_bytes=1024)
    assert (
        preflight_iso_bmff(source, max_header_bytes=len(source)).status
        == "decoder_required"
    )


def test_media_payload_is_skipped_even_beyond_byte_bound():
    source = heif()
    payload_start = source.index(b"mdat") + 4
    assert preflight_iso_bmff(source, max_header_bytes=payload_start).has_exif
    with pytest.raises(BmffPreflightError, match="bmff_metadata_out_of_bounds"):
        preflight_iso_bmff(source, max_header_bytes=payload_start - 1)
    assert preflight_iso_bmff(source[:-400] + b"Z" * 400) == preflight_iso_bmff(source)


def test_size_zero_terminal_box_and_extended_metadata_are_supported():
    source = heif()
    offset = source.index(b"mdat") - 4
    source = source[:offset] + box(b"mdat", b"Z", to_end=True)
    assert preflight_iso_bmff(source).width == 4032
    assert preflight_iso_bmff(ftyp(extended=True) + heif()[20:]).height == 3024


@pytest.mark.parametrize(
    "options", [{"max_depth": 65}, {"max_header_bytes": False}, {"max_boxes": 0}]
)
def test_invalid_limits_are_value_free(options):
    with pytest.raises(ValueError):
        preflight_iso_bmff(heif(), **options)


def test_every_truncated_prefix_fails_without_returning_payloads():
    source = heif()
    boundary = source.index(b"mdat") - 4
    for size in range(len(source)):
        if size == boundary:
            assert preflight_iso_bmff(source[:size]).width == 4032
            continue
        with pytest.raises(BmffPreflightError) as error:
            preflight_iso_bmff(source[:size])
        assert "Synthetic" not in str(error.value)
