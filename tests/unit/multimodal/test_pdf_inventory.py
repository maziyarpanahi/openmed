"""Offline safety controls for counts-only PDF inventory."""

import io
import json

import pytest

from openmed.multimodal.abstention import AbstentionReason
from openmed.multimodal.pdf_geometry import PdfGeometryError, PdfGeometryStatus
from openmed.multimodal.pdf_inventory import PdfContentProfile, read_pdf_inventory
from tests.fixtures.pdf_inventory import (
    SENTINEL,
    _classic_pdf,
    _object_stream_pdf,
    _single_page,
    _stream_object,
    category_pdf,
    incremental_pdf,
    plain_pdf,
)

CATEGORIES = (
    (
        "annotations",
        "pdf_annotations",
        "annotation_subtypes",
        {"Text": 1, "other": 1},
        False,
    ),
    ("forms", "pdf_form_values", "field_value_count", 1, False),
    ("attachments", "pdf_embedded_files", "embedded_file_count", 1, True),
    ("xfa", "pdf_xfa", "xfa_packet_count", 2, False),
    (
        "optional_content",
        "pdf_optional_content",
        "optional_content_group_count",
        1,
        False,
    ),
    ("javascript", "pdf_javascript", "javascript_action_count", 1, True),
    ("open_action", "pdf_open_action", "open_action_count", 1, True),
    ("launch_action", "pdf_launch_action", "launch_action_count", 1, True),
)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("category,code,count,expected,strict", CATEGORIES)
def test_categories_and_profiles(category, code, count, expected, strict, compressed):
    source = category_pdf(category, compressed=compressed)
    report = read_pdf_inventory(source)
    assert report.reason_codes == (code,)
    assert report.status is (
        PdfGeometryStatus.REJECTED if strict else PdfGeometryStatus.REVIEW
    )
    assert report.inventory.to_dict()[count] == expected
    assert report.abstention.reason is (
        AbstentionReason.UNSUPPORTED_MEDIA
        if strict
        else AbstentionReason.PHI_UNCERTAINTY
    )
    reviewed = read_pdf_inventory(source, profile=PdfContentProfile.REVIEW)
    assert reviewed.status is PdfGeometryStatus.REVIEW
    assert reviewed.inventory == report.inventory
    # JSON, ordinary diagnostics and repeat reads never reveal submitted strings.
    for rendered in (report.to_json(), repr(report), str(report.to_dict())):
        assert SENTINEL not in rendered
        assert "changed-" not in rendered
    assert report.to_json() == read_pdf_inventory(source).to_json()


def test_plain_text_has_no_hidden_categories():
    report = read_pdf_inventory(plain_pdf())
    assert report.status is PdfGeometryStatus.READABLE
    assert report.reason_codes == () and report.abstention is None
    counts = report.inventory.to_dict()
    assert counts.pop("revision_count") == 1
    assert counts.pop("annotation_subtypes") == {}
    assert set(counts.values()) == {0}


def test_incremental_field_change_inventories_old_and_current_values():
    report = read_pdf_inventory(incremental_pdf())
    assert report.inventory.field_count == report.inventory.field_value_count == 2
    assert report.inventory.revision_count == 2
    assert report.reason_codes == ("pdf_form_values", "pdf_incremental_revisions")
    assert SENTINEL not in report.to_json()


def test_removed_field_definition_is_still_inventoried():
    # The latest form field reference resolves to null; retained prior PHI must
    # not yield a readable verdict even when current geometry is valid.
    report = read_pdf_inventory(incremental_pdf(remove_field=True))
    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_inventory_invalid",)
    assert SENTINEL not in repr(report)


@pytest.mark.parametrize("kind", ["fields", "names", "reference", "page"])
def test_cyclic_structures_reject_with_bounded_codes(kind):
    objects = _single_page(MediaBox="[0 0 612 792]")
    if kind == "page":
        objects[2] = "<< /Type /Pages /Kids [2 0 R] /Count 1 >>"
        expected = "pdf_page_tree_invalid"
    else:
        if kind == "fields":
            objects[1] = (
                "<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [5 0 R] >> >>"
            )
            objects[5] = "<< /T (synthetic) /Kids [5 0 R] >>"
        elif kind == "names":
            objects[1] = (
                "<< /Type /Catalog /Pages 2 0 R /Names << /EmbeddedFiles 5 0 R >> >>"
            )
            objects[5] = "<< /Kids [5 0 R] >>"
        else:
            objects[3] = objects[3].replace("/Contents", "/Annots 5 0 R /Contents")
            objects[5] = "6 0 R"
            objects[6] = "5 0 R"
        expected = "pdf_inventory_invalid"
    report = read_pdf_inventory(_classic_pdf(objects))
    assert report.status is PdfGeometryStatus.REJECTED
    assert report.inventory is None
    assert report.reason_codes == (expected,)


@pytest.mark.parametrize("compressed", [False, True])
def test_object_and_byte_limits(compressed):
    source = category_pdf("forms", compressed=compressed)
    for limits, code in (
        ({"max_bytes": len(source) - 1}, "pdf_size_limit"),
        ({"max_objects": 1}, "pdf_object_limit"),
    ):
        report = read_pdf_inventory(source, **limits)
        assert report.reason_codes == (code,)
        assert report.abstention.reason is AbstentionReason.RESOURCE_LIMIT


@pytest.mark.parametrize("filter_name", [None, "FlateDecode"])
def test_decompression_budget_includes_uncompressed_object_streams(filter_name):
    source = _object_stream_pdf(
        {5: "<< /S /JavaScript /JS (synthetic) >>"},
        _single_page(MediaBox="[0 0 612 792]"),
        filter_name=filter_name,
    )
    assert read_pdf_inventory(source, max_decompressed_bytes=1).reason_codes == (
        "pdf_decompression_limit",
    )


def test_unsupported_object_stream_cannot_hide_behind_readable_direct_pages():
    source = _object_stream_pdf(
        {5: "<< /S /JavaScript /JS (synthetic) >>"},
        _single_page(MediaBox="[0 0 612 792]"),
        filter_name="LZWDecode",
    )
    assert read_pdf_inventory(source).reason_codes == ("pdf_object_stream_unsupported",)


def test_fake_trailers_and_objects_in_streams_and_strings_are_not_counted():
    objects = _single_page(MediaBox="[0 0 612 792]")
    marker = b"trailer << /Root 1 0 R >> 8 0 obj << /S /JavaScript >> endobj"
    objects[4] = _stream_object("", marker).decode()
    objects[5] = "(trailer << /Root 1 0 R >> 8 0 obj << /S /Launch >> endobj)"
    report = read_pdf_inventory(_classic_pdf(objects))
    assert report.status is PdfGeometryStatus.READABLE
    assert report.inventory.revision_count == 1


def test_seekable_stream_is_restored_and_not_closed():
    source = io.BytesIO(b"prefix" + category_pdf("attachments"))
    source.seek(6)
    report = read_pdf_inventory(source)
    assert source.tell() == 6 and not source.closed
    assert report.reason_codes == ("pdf_embedded_files",)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits_are_value_free(limit):
    with pytest.raises(PdfGeometryError, match="max_objects_invalid"):
        read_pdf_inventory(plain_pdf(), max_objects=limit)


def test_private_source_failure_has_only_a_controlled_message():
    class Broken:
        def read(self, size):
            raise OSError(SENTINEL)

    with pytest.raises(PdfGeometryError) as caught:
        read_pdf_inventory(Broken())
    assert str(caught.value) == "source_read_error"
    assert SENTINEL not in str(caught.value)


def test_escaped_pdf_names_use_controlled_buckets():
    source = plain_pdf(Annots="[<< /Subtype /Te#78t /Contents (synthetic) >>]")
    report = read_pdf_inventory(source)
    assert json.loads(report.to_json())["inventory"]["annotation_subtypes"] == {
        "Text": 1
    }


def test_hybrid_xref_trailer_is_not_an_extra_revision():
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[5] = _stream_object("/Type /XRef /Root 1 0 R /W [1 4 2]", b"").decode()
    report = read_pdf_inventory(_classic_pdf(objects, trailer="/XRefStm 0 "))
    assert report.status is PdfGeometryStatus.READABLE
    assert report.inventory.revision_count == 1


def test_public_inventory_cannot_echo_arbitrary_subtype_or_reason():
    from openmed.multimodal import PdfContentInventory, PdfInventoryReport

    with pytest.raises(PdfGeometryError, match="pdf_inventory_subtypes_invalid"):
        PdfContentInventory(annotation_subtypes=((SENTINEL, 1),))
    with pytest.raises(PdfGeometryError, match="pdf_inventory_codes_invalid"):
        PdfInventoryReport(PdfGeometryStatus.REVIEW, (SENTINEL,), None, None)


def test_huge_numeric_tokens_fail_with_controlled_code():
    objects = _single_page(MediaBox="[0 0 612 792]")
    objects[5] = "<< /V " + "9" * 5000 + " >>"
    report = read_pdf_inventory(_classic_pdf(objects))
    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_inventory_invalid",)


def test_every_annotation_subtype_is_bucketed_without_source_payloads():
    from openmed.multimodal import PDF_ANNOTATION_SUBTYPES

    entries = " ".join(
        f"<< /Subtype /{code} /Contents ({SENTINEL}) >>"
        for code in PDF_ANNOTATION_SUBTYPES[:-1]
    )
    report = read_pdf_inventory(plain_pdf(Annots=f"[{entries}]"))
    assert report.inventory.annotation_subtypes == tuple(
        (code, 1) for code in PDF_ANNOTATION_SUBTYPES[:-1]
    )
    assert SENTINEL not in report.to_json()


def test_stream_contract_violation_is_rejected_without_payload_details():
    class Oversized:
        def read(self, size):
            return b"x" * (size + 1)

    with pytest.raises(PdfGeometryError, match="source_invalid"):
        read_pdf_inventory(Oversized(), max_bytes=10)


def test_annotation_author_title_does_not_become_an_acroform_field():
    source = plain_pdf(
        Annots=f"[<< /Subtype /Text /T ({SENTINEL}) /Contents ({SENTINEL}) >>]"
    )
    report = read_pdf_inventory(source)
    assert report.reason_codes == ("pdf_annotations",)
    assert report.inventory.field_count == report.inventory.field_value_count == 0


def test_file_attachment_annotation_is_strictly_rejected_without_extracting_it():
    source = plain_pdf(
        Annots=f"[<< /Subtype /FileAttachment /FS << /F ({SENTINEL}) >> >>]"
    )
    report = read_pdf_inventory(source)
    assert report.status is PdfGeometryStatus.REJECTED
    assert report.reason_codes == ("pdf_annotations", "pdf_embedded_files")
    assert SENTINEL not in report.to_json()
