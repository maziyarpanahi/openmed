from __future__ import annotations

import io
import json
import zlib
from typing import Any

import pytest

from openmed.multimodal.pdf_geometry import (
    PDF_GEOMETRY_SCHEMA_VERSION,
    PDF_REASON_CODES,
    PdfGeometryError,
    PdfGeometryStatus,
    read_pdf_geometry,
)

_SENTINEL = "Synthetic Patient Jane Roe MRN 000123"


def _classic_pdf(
    objects: dict[int, str], *, version: str = "1.7", trailer: str = ""
) -> bytes:
    """Build a synthetic PDF with a classic cross-reference table."""
    output = bytearray(f"%PDF-{version}\n%\xe2\xe3\xcf\xd3\n".encode("latin-1"))
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")
    size = max(objects) + 1
    xref = len(output)
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii")
    for number in range(1, size):
        if number in offsets:
            output += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
        else:
            output += b"0000000000 65535 f \n"
    output += (
        f"trailer\n<< /Size {size} /Root 1 0 R {trailer}>>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(output)


def _stream_object(dictionary: str, payload: bytes) -> bytes:
    return (
        f"<< {dictionary} /Length {len(payload)} >>\nstream\n".encode("latin-1")
        + payload
        + b"\nendstream"
    )


def _object_stream_pdf(
    compressed: dict[int, str],
    direct: dict[int, str],
    *,
    filter_name: str | None = "FlateDecode",
    corrupt: bool = False,
) -> bytes:
    """Build a synthetic PDF 1.5 file with an object stream and an xref stream."""
    numbers = sorted(compressed)
    bodies = [compressed[number].encode("latin-1") for number in numbers]
    offsets: list[int] = []
    position = 0
    for body in bodies:
        offsets.append(position)
        position += len(body) + 1
    index = " ".join(f"{number} {offset}" for number, offset in zip(numbers, offsets))
    header = (index + "\n").encode("ascii")
    payload = header + b"\n".join(bodies) + b"\n"
    encoded = zlib.compress(payload) if filter_name == "FlateDecode" else payload
    if corrupt:
        encoded = b"\x00not-zlib" + encoded[8:]
    filter_entry = f"/Filter /{filter_name}" if filter_name else ""
    stream_number = max([*compressed, *direct]) + 1
    output = bytearray(b"%PDF-1.5\n")
    for number in sorted(direct):
        output += f"{number} 0 obj\n{direct[number]}\nendobj\n".encode("latin-1")
    output += f"{stream_number} 0 obj\n".encode("ascii")
    output += _stream_object(
        f"/Type /ObjStm /N {len(numbers)} /First {len(header)} {filter_entry}",
        encoded,
    )
    output += b"\nendobj\n"
    xref_number = stream_number + 1
    xref_offset = len(output)
    output += f"{xref_number} 0 obj\n".encode("ascii")
    output += _stream_object(
        f"/Type /XRef /Size {xref_number + 1} /Root 1 0 R /W [1 4 2]",
        b"",
    )
    output += f"\nendobj\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return bytes(output)


def _single_page(**page_entries: str) -> dict[int, str]:
    entries = " ".join(f"/{key} {value}" for key, value in page_entries.items())
    return {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: f"<< /Type /Page /Parent 2 0 R {entries} /Contents 4 0 R >>",
        4: _stream_object("", f"BT ({_SENTINEL}) Tj ET".encode("latin-1")).decode(
            "latin-1"
        ),
    }


def test_single_page_letter_geometry_is_readable() -> None:
    report = read_pdf_geometry(_classic_pdf(_single_page(MediaBox="[0 0 612 792]")))

    assert report.status is PdfGeometryStatus.READABLE
    assert report.reason_codes == ()
    assert (report.version_major, report.version_minor) == (1, 7)
    assert report.page_count == report.declared_page_count == 1
    (page,) = report.pages
    assert page.page_index == 0
    assert page.media_box == (0.0, 0.0, 612.0, 792.0)
    assert page.crop_box == page.media_box
    assert page.rotation == 0
    assert (page.width, page.height) == (612.0, 792.0)


def test_mixed_page_sizes_with_inheritance_keep_page_tree_order() -> None:
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [3 0 R 4 0 R 6 0 R] /Count 4 /MediaBox [0 0 612 792] >>",
        3: "<< /Type /Page /Parent 2 0 R >>",
        4: "<< /Type /Pages /Parent 2 0 R /Kids [5 0 R 7 0 R] /Count 2 "
        "/MediaBox [0 0 595.28 841.89] /Rotate 90 >>",
        5: "<< /Type /Page /Parent 4 0 R >>",
        6: "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1224 792] >>",
        7: "<< /Type /Page /Parent 4 0 R /Rotate 0 /CropBox [36 36 559.28 805.89] >>",
    }

    report = read_pdf_geometry(_classic_pdf(objects))

    assert report.status is PdfGeometryStatus.READABLE
    assert report.page_count == 4
    assert [page.media_box for page in report.pages] == [
        (0.0, 0.0, 612.0, 792.0),
        (0.0, 0.0, 595.28, 841.89),
        (0.0, 0.0, 595.28, 841.89),
        (0.0, 0.0, 1224.0, 792.0),
    ]
    assert [page.rotation for page in report.pages] == [0, 90, 0, 0]
    assert [(page.width, page.height) for page in report.pages] == [
        (612.0, 792.0),
        (841.89, 595.28),
        (523.28, 769.89),
        (1224.0, 792.0),
    ]


@pytest.mark.parametrize(
    ("rotate", "expected"),
    [("90", 90), ("180", 180), ("270", 270), ("-90", 270), ("450", 90), ("0", 0)],
)
def test_rotation_is_normalized_clockwise(rotate: str, expected: int) -> None:
    report = read_pdf_geometry(
        _classic_pdf(_single_page(MediaBox="[0 0 612 792]", Rotate=rotate))
    )

    (page,) = report.pages
    assert report.status is PdfGeometryStatus.READABLE
    assert page.rotation == expected
    swapped = expected in (90, 270)
    assert (page.width, page.height) == ((792.0, 612.0) if swapped else (612.0, 792.0))


@pytest.mark.parametrize("rotate", ["45", "90.0", "(90)", "/Ninety"])
def test_invalid_rotation_is_reviewed_and_treated_as_upright(rotate: str) -> None:
    report = read_pdf_geometry(
        _classic_pdf(_single_page(MediaBox="[0 0 612 792]", Rotate=rotate))
    )

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("rotation_invalid",)
    assert report.pages[0].rotation == 0


def test_missing_media_box_is_reviewed_without_geometry() -> None:
    report = read_pdf_geometry(_classic_pdf(_single_page()))

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("media_box_missing",)
    (page,) = report.pages
    assert page.media_box is page.crop_box is page.width is page.height is None


@pytest.mark.parametrize(
    "media_box",
    [
        "[0 0 612]",
        "[0 0 612 792 1]",
        "[0 0 0 792]",
        "[0 0 (612) 792]",
        "612",
        "[0 0 1e9 792]",
    ],
)
def test_invalid_media_box_is_reviewed(media_box: str) -> None:
    report = read_pdf_geometry(_classic_pdf(_single_page(MediaBox=media_box)))

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("media_box_invalid",)
    assert report.pages[0].media_box is None


def test_reversed_box_corners_and_indirect_values_are_normalized() -> None:
    objects = _single_page(MediaBox="5 0 R", CropBox="[600 780 12 12]")
    objects[5] = "[612 792 0 6 0 R]"
    objects[6] = "0"

    report = read_pdf_geometry(_classic_pdf(objects))

    (page,) = report.pages
    assert report.status is PdfGeometryStatus.READABLE
    assert page.media_box == (0.0, 0.0, 612.0, 792.0)
    assert page.crop_box == (12.0, 12.0, 600.0, 780.0)
    assert (page.width, page.height) == (588.0, 768.0)


def test_crop_box_outside_media_box_is_clipped_and_reviewed() -> None:
    report = read_pdf_geometry(
        _classic_pdf(
            _single_page(MediaBox="[0 0 612 792]", CropBox="[-10 100 700 500]")
        )
    )

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("crop_box_outside_media_box",)
    assert report.pages[0].crop_box == (0.0, 100.0, 612.0, 500.0)


@pytest.mark.parametrize("crop_box", ["[700 800 900 900]", "[0 0 1 ]", "[0 0 0 0]"])
def test_unusable_crop_box_falls_back_to_media_box(crop_box: str) -> None:
    report = read_pdf_geometry(
        _classic_pdf(_single_page(MediaBox="[0 0 612 792]", CropBox=crop_box))
    )

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("crop_box_invalid",)
    assert report.pages[0].crop_box == (0.0, 0.0, 612.0, 792.0)


def test_declared_count_mismatch_is_reviewed() -> None:
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[2] = "<< /Type /Pages /Kids [3 0 R] /Count 5 >>"

    report = read_pdf_geometry(_classic_pdf(objects))

    assert report.status is PdfGeometryStatus.REVIEW
    assert report.reason_codes == ("page_count_mismatch",)
    assert (report.page_count, report.declared_page_count) == (1, 5)


def test_catalog_version_can_raise_the_header_version() -> None:
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[1] = "<< /Type /Catalog /Pages 2 0 R /Version /2.0 >>"

    report = read_pdf_geometry(_classic_pdf(objects, version="1.4"))

    assert (report.version_major, report.version_minor) == (2, 0)


def test_incremental_update_uses_the_latest_object_definition() -> None:
    original = _classic_pdf(_single_page(MediaBox="[0 0 612 792]"))
    update = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] >>\nendobj\n"
        b"trailer\n<< /Size 5 /Root 1 0 R /Prev 0 >>\nstartxref\n0\n%%EOF\n"
    )

    report = read_pdf_geometry(original + update)

    assert report.pages[0].media_box == (0.0, 0.0, 842.0, 595.0)


def test_object_streams_are_expanded() -> None:
    pdf = _object_stream_pdf(
        {
            2: "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            3: "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
            4: "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] /Rotate 270 >>",
        },
        {1: "<< /Type /Catalog /Pages 2 0 R >>"},
    )

    report = read_pdf_geometry(pdf)

    assert report.status is PdfGeometryStatus.READABLE
    assert (report.version_major, report.version_minor) == (1, 5)
    assert [(page.width, page.height) for page in report.pages] == [
        (612.0, 792.0),
        (612.0, 792.0),
    ]


def test_uncompressed_object_streams_are_expanded() -> None:
    pdf = _object_stream_pdf(
        {
            1: "<< /Type /Catalog /Pages 2 0 R >>",
            2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: "<< /Type /Page /MediaBox [0 0 100 200] >>",
        },
        {},
        filter_name=None,
    )

    assert read_pdf_geometry(pdf).pages[0].media_box == (0.0, 0.0, 100.0, 200.0)


@pytest.mark.parametrize("variant", [{"filter_name": "LZWDecode"}, {"corrupt": True}])
def test_unreadable_object_streams_are_rejected(variant: dict[str, Any]) -> None:
    pdf = _object_stream_pdf(
        {
            2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: "<< /Type /Page /MediaBox [0 0 612 792] >>",
        },
        {1: "<< /Type /Catalog /Pages 2 0 R >>"},
        **variant,
    )

    report = read_pdf_geometry(pdf)

    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_object_stream_unsupported",)


def test_encrypted_documents_are_rejected_without_pages() -> None:
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[5] = "<< /Filter /Standard /V 2 /R 3 /O <00> /U <00> /P -4 >>"

    report = read_pdf_geometry(_classic_pdf(objects, trailer="/Encrypt 5 0 R "))

    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_encrypted",)
    assert report.pages == ()
    assert report.page_count is None
    assert (report.version_major, report.version_minor) == (1, 7)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "pdf_header_missing"),
        (b"GIF89a not a pdf", "pdf_header_missing"),
        (b" " * 1100 + b"%PDF-1.7\n", "pdf_header_missing"),
        (b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog", "pdf_catalog_missing"),
        (b"%PDF-1.7\ntrailer\n<< /Root 9 0 R >>\n%%EOF", "pdf_catalog_missing"),
        (
            b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R >>\n",
            "pdf_page_tree_invalid",
        ),
    ],
)
def test_malformed_documents_are_rejected(payload: bytes, reason: str) -> None:
    report = read_pdf_geometry(payload)

    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == (reason,)
    assert report.pages == ()


def test_header_within_first_kilobyte_is_accepted() -> None:
    pdf = _classic_pdf(_single_page(MediaBox="[0 0 612 792]"))

    assert read_pdf_geometry(b" " * 512 + pdf).status is PdfGeometryStatus.READABLE


def test_page_tree_cycles_are_rejected() -> None:
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[2] = "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 1 >>"
    objects[5] = "<< /Type /Pages /Kids [2 0 R] /Count 1 >>"

    report = read_pdf_geometry(_classic_pdf(objects))

    assert report.reason_codes == ("pdf_page_tree_invalid",)


def test_non_array_kids_and_unknown_node_types_are_rejected() -> None:
    kids = _single_page(MediaBox="[0 0 612 792]")
    kids[2] = "<< /Type /Pages /Kids 3 /Count 1 >>"
    unknown = _single_page(MediaBox="[0 0 612 792]")
    unknown[3] = "<< /Type /Annot >>"

    assert read_pdf_geometry(_classic_pdf(kids)).reason_codes == (
        "pdf_page_tree_invalid",
    )
    assert read_pdf_geometry(_classic_pdf(unknown)).reason_codes == (
        "pdf_page_tree_invalid",
    )


def test_page_limit_is_enforced() -> None:
    kids = " ".join(f"{number} 0 R" for number in range(3, 8))
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count 5 /MediaBox [0 0 10 10] >>",
        **{number: "<< /Type /Page >>" for number in range(3, 8)},
    }
    pdf = _classic_pdf(objects)

    assert read_pdf_geometry(pdf, max_pages=5).page_count == 5
    report = read_pdf_geometry(pdf, max_pages=4)
    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_page_limit",)


def test_byte_object_and_decompression_limits_are_enforced() -> None:
    pdf = _classic_pdf(_single_page(MediaBox="[0 0 612 792]"))
    object_stream = _object_stream_pdf(
        {
            2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: "<< /Type /Page /MediaBox [0 0 612 792] >>",
        },
        {1: "<< /Type /Catalog /Pages 2 0 R >>"},
    )

    assert (
        read_pdf_geometry(pdf, max_bytes=len(pdf)).status is PdfGeometryStatus.READABLE
    )
    assert read_pdf_geometry(pdf, max_bytes=len(pdf) - 1).reason_codes == (
        "pdf_size_limit",
    )
    assert read_pdf_geometry(pdf, max_objects=4).status is PdfGeometryStatus.READABLE
    assert read_pdf_geometry(pdf, max_objects=3).reason_codes == ("pdf_object_limit",)
    assert read_pdf_geometry(object_stream, max_decompressed_bytes=16).reason_codes == (
        "pdf_decompression_limit",
    )


def test_deep_page_trees_are_rejected() -> None:
    objects: dict[int, str] = {1: "<< /Type /Catalog /Pages 2 0 R >>"}
    for number in range(2, 70):
        objects[number] = f"<< /Type /Pages /Kids [{number + 1} 0 R] /Count 1 >>"
    objects[70] = "<< /Type /Page /MediaBox [0 0 10 10] >>"

    assert read_pdf_geometry(_classic_pdf(objects)).reason_codes == (
        "pdf_page_tree_invalid",
    )


def test_streams_are_read_and_restored_but_not_closed() -> None:
    pdf = _classic_pdf(_single_page(MediaBox="[0 0 612 792]"))
    stream = io.BytesIO(b"prefix" + pdf)
    stream.seek(6)

    report = read_pdf_geometry(stream)

    assert report.status is PdfGeometryStatus.READABLE
    assert stream.tell() == 6
    assert not stream.closed
    assert read_pdf_geometry(bytearray(pdf)) == read_pdf_geometry(memoryview(pdf))


def test_oversized_streams_are_rejected_without_reading_everything() -> None:
    class CountingStream(io.BytesIO):
        requested = 0

        def read(self, size: int | None = -1) -> bytes:
            CountingStream.requested += size if size and size > 0 else 0
            return super().read(size)

    stream = CountingStream(b"%PDF-1.7\n" + b"0" * 10_000)

    report = read_pdf_geometry(stream, max_bytes=100)

    assert report.reason_codes == ("pdf_size_limit",)
    assert CountingStream.requested <= 101


def test_outputs_contain_no_text_or_names() -> None:
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[1] = (
        f"<< /Type /Catalog /Pages 2 0 R /Info << /Title ({_SENTINEL}) >> "
        f"/Names << /Dests <4a616e65> >> >>"
    )
    pdf = _classic_pdf(objects, trailer=f"/ID [({_SENTINEL}) ({_SENTINEL})] ")

    report = read_pdf_geometry(pdf)
    rendered = report.to_json() + repr(report)

    assert report.status is PdfGeometryStatus.READABLE
    assert _SENTINEL not in rendered
    assert "Catalog" not in rendered
    assert "Jane" not in rendered


def test_report_serialization_is_deterministic() -> None:
    report = read_pdf_geometry(
        _classic_pdf(
            _single_page(MediaBox="[0 0 612 792]", Rotate="90", CropBox="[0 0 700 700]")
        )
    )

    assert list(report.to_dict()) == [
        "schema_version",
        "status",
        "reason_codes",
        "version_major",
        "version_minor",
        "page_count",
        "declared_page_count",
        "pages",
    ]
    assert report.to_json() == (
        '{"declared_page_count":1,"page_count":1,"pages":[{"crop_box":'
        '[0.0,0.0,612.0,700.0],"height":612.0,"media_box":[0.0,0.0,612.0,792.0],'
        '"page_index":0,"reason_codes":["crop_box_outside_media_box"],'
        '"rotation":90,"width":700.0}],"reason_codes":'
        '["crop_box_outside_media_box"],'
        f'"schema_version":"{PDF_GEOMETRY_SCHEMA_VERSION}",'
        '"status":"review","version_major":1,"version_minor":7}'
    )
    assert json.loads(report.to_json()) == report.to_dict()
    assert set(report.reason_codes) <= set(PDF_REASON_CODES)


@pytest.mark.parametrize(
    ("kwargs", "category"),
    [
        ({"max_bytes": 0}, "max_bytes_invalid"),
        ({"max_pages": True}, "max_pages_invalid"),
        ({"max_objects": -1}, "max_objects_invalid"),
        ({"max_decompressed_bytes": 1.5}, "max_decompressed_bytes_invalid"),
    ],
)
def test_invalid_limits_raise_value_free_errors(
    kwargs: dict[str, Any], category: str
) -> None:
    with pytest.raises(PdfGeometryError) as exc_info:
        read_pdf_geometry(b"%PDF-1.7", **kwargs)

    assert exc_info.value.category == category
    assert str(exc_info.value) == category


@pytest.mark.parametrize("source", [_SENTINEL, 42, None])
def test_invalid_sources_raise_value_free_errors(source: Any) -> None:
    with pytest.raises(PdfGeometryError) as exc_info:
        read_pdf_geometry(source)

    assert exc_info.value.category == "source_invalid"
    assert _SENTINEL not in str(exc_info.value)
