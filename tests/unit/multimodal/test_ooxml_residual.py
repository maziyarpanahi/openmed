"""Synthetic package inspection, malformed controls and value-free refusals."""

import io
import zipfile

import pytest

from openmed.multimodal.ooxml_residual import (
    OoxmlResidualError,
    _sanitize,
    _xlsx_coverage,
    inspect_ooxml,
    verify_ooxml,
)

SENTINEL = "SYNTHETIC_PRIVATE_患者_3810"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def package(parts):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    return output.getvalue()


@pytest.mark.parametrize(
    "part,xml,category",
    [
        (
            "word/document.xml",
            f'<w:document xmlns:w="{W}"><w:del w:author="{SENTINEL}"><w:delText>{SENTINEL}</w:delText></w:del></w:document>',
            "tracked_changes",
        ),
        (
            "word/comments.xml",
            f'<comments author="{SENTINEL}">{SENTINEL}</comments>',
            "comments_notes",
        ),
        (
            "ppt/notesSlides/notesSlide1.xml",
            f"<notes>{SENTINEL}</notes>",
            "comments_notes",
        ),
        ("word/header1.xml", f"<header>{SENTINEL}</header>", "headers_footers"),
        ("word/footer1.xml", f"<footer>{SENTINEL}</footer>", "headers_footers"),
        ("docProps/core.xml", f'<properties author="{SENTINEL}"/>', "core_properties"),
        ("docProps/app.xml", f"<properties>{SENTINEL}</properties>", "app_properties"),
        (
            "docProps/custom.xml",
            f'<properties><property name="{SENTINEL}"/></properties>',
            "custom_properties",
        ),
        ("customXml/item1.xml", f"<data>{SENTINEL}</data>", "custom_xml"),
        (
            "xl/workbook.xml",
            f'<workbook xmlns="{S}"><definedName name="{SENTINEL}">{SENTINEL}</definedName></workbook>',
            "defined_names",
        ),
        (
            "xl/workbook.xml",
            f'<workbook xmlns="{S}"><sheet name="{SENTINEL}" state="veryHidden"/></workbook>',
            "hidden_sheets",
        ),
        (
            "xl/pivotCache/pivotCacheRecords1.xml",
            f"<records>{SENTINEL}</records>",
            "pivot_caches",
        ),
        (
            "xl/externalLinks/externalLink1.xml",
            f"<external>{SENTINEL}</external>",
            "external_links",
        ),
        ("word/embeddings/object.bin", SENTINEL, "embedded_objects"),
        (
            "word/document.xml",
            f'<document><docPr descr="{SENTINEL}" title="{SENTINEL}"/></document>',
            "alt_text",
        ),
        ("word/private-part.xml", f'<data author="{SENTINEL}"/>', "uncovered_part"),
        (
            "word/document.xml",
            f"<document><textbox>{SENTINEL}</textbox></document>",
            "uncovered_text",
        ),
        (
            "word/_rels/document.xml.rels",
            f'<Relationships><Relationship TargetMode="External" Target="file:///{SENTINEL}"/></Relationships>',
            "external_links",
        ),
    ],
)
def test_hidden_categories_and_diagnostics(part, xml, category):
    data = package({part: xml})
    findings = inspect_ooxml(data)
    assert category in {item.category for item in findings}
    with pytest.raises(OoxmlResidualError) as refusal:
        verify_ooxml(data)
    report = [item.to_dict() for item in refusal.value.findings]
    assert all(set(item) == {"category", "count", "digest"} for item in report)
    assert all(item["count"] > 0 and len(item["digest"]) == 64 for item in report)
    assert SENTINEL not in str(report) + str(refusal.value) + repr(refusal.value)
    assert part not in str(report) + str(refusal.value)


@pytest.mark.parametrize(
    "data,category",
    [
        (b"not a zip", "invalid_package"),
        (package({"word/document.xml": f"<broken>{SENTINEL}"}), "invalid_xml"),
        (
            package(
                {"word/document.xml": '<!DOCTYPE d [<!ENTITY a "private">]><d>&a;</d>'}
            ),
            "invalid_xml",
        ),
        (
            package({"word/document.xml": f"<document><!--{SENTINEL}--></document>"}),
            "uncovered_text",
        ),
        (
            package(
                {"word/document.xml": f"<document><?private {SENTINEL}?></document>"}
            ),
            "uncovered_text",
        ),
    ],
)
def test_malformed_and_non_element_payloads_fail_closed(data, category):
    assert category in {item.category for item in inspect_ooxml(data)}


def test_coverage_exempts_only_addressed_text():
    data = package(
        {
            "word/document.xml": "<document><t>visible</t><hidden>private</hidden></document>"
        }
    )
    findings = inspect_ooxml(data, covered_text={"word/document.xml": {(0,)}})
    assert len(findings) == 1
    assert findings[0].category == "uncovered_text"
    assert findings[0].count == 1


def test_known_scrub_removes_sentinel_and_internal_references():
    data = package(
        {
            "docProps/core.xml": f"<properties>{SENTINEL}</properties>",
            "customXml/item1.xml": f"<data>{SENTINEL}</data>",
            "docProps/thumbnail.jpeg": SENTINEL,
            "word/document.xml": f'<document><docPr descr="{SENTINEL}"/></document>',
            "word/_rels/document.xml.rels": '<Relationships><Relationship Target="../customXml/item1.xml"/></Relationships>',
            "[Content_Types].xml": '<Types><Override PartName="/customXml/item1.xml"/></Types>',
        }
    )
    safe = _sanitize(data)
    verify_ooxml(safe)
    with zipfile.ZipFile(io.BytesIO(safe)) as archive:
        assert all(
            SENTINEL.encode() not in archive.read(name) for name in archive.namelist()
        )
        assert "customXml/item1.xml" not in archive.namelist()
        assert b"customXml" not in archive.read("word/_rels/document.xml.rels")
        assert b"customXml" not in archive.read("[Content_Types].xml")


def test_malformed_xlsx_coverage_has_typed_safe_error():
    with pytest.raises(OoxmlResidualError, match="invalid_package") as error:
        _xlsx_coverage(package({"private.xml": SENTINEL}))
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "xml",
    [
        f"<!--{SENTINEL}--><document/>",
        f"<?private {SENTINEL}?><document/>",
        f'<w:document xmlns:w="{W}"><w:fldSimple w:instr="{SENTINEL}"/></w:document>',
        f'<w:document xmlns:w="{W}"><w:docVar w:name="{SENTINEL}"/></w:document>',
    ],
)
def test_non_run_payloads_are_refused(xml):
    findings = inspect_ooxml(package({"word/document.xml": xml}))
    assert "uncovered_text" in {item.category for item in findings}


def test_directory_entries_do_not_escape_inspection():
    findings = inspect_ooxml(package({"private/": SENTINEL}))
    assert findings[0].category == "uncovered_part"


def test_xlsx_unused_shared_strings_and_phonetic_text_are_uncovered():
    data = package(
        {
            "xl/workbook.xml": f'<workbook xmlns="{S}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet r:id="r1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
            "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{S}"><sheetData><row><c t="s"><v>0</v></c></row></sheetData></worksheet>',
            "xl/sharedStrings.xml": f'<sst xmlns="{S}"><si><t>covered</t><rPh><t>{SENTINEL}</t></rPh></si><si><t>{SENTINEL}</t></si></sst>',
        }
    )
    findings = inspect_ooxml(data, covered_text=_xlsx_coverage(data))
    assert len(findings) == 1 and findings[0].count == 2
    assert findings[0].category == "uncovered_text"


def test_cleared_properties_drop_custom_tags_and_unused_namespace_values():
    data = package(
        {
            "docProps/custom.xml": f'<private xmlns="{SENTINEL}" xmlns:unused="{SENTINEL}"/>'
        }
    )
    safe = _sanitize(data)
    verify_ooxml(safe)
    with zipfile.ZipFile(io.BytesIO(safe)) as archive:
        assert SENTINEL.encode() not in archive.read("docProps/custom.xml")


def test_printer_setting_nodes_and_relationships_are_removed():
    data = package(
        {
            "word/settings.xml": f'<settings xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:printerSettings r:id="printer"/></settings>',
            "word/printerSettings/printerSettings1.bin": SENTINEL,
            "word/_rels/settings.xml.rels": '<Relationships><Relationship Id="printer" Target="printerSettings/printerSettings1.bin"/></Relationships>',
        }
    )
    safe = _sanitize(data)
    verify_ooxml(safe)
    with zipfile.ZipFile(io.BytesIO(safe)) as archive:
        assert b"printerSettings" not in archive.read("word/settings.xml")
        assert b"printerSettings" not in archive.read("word/_rels/settings.xml.rels")


def test_reused_output_stream_has_no_old_prefix_or_trailing_payload():
    from openmed.multimodal.ooxml_residual import _publish

    stream = io.BytesIO((SENTINEL * 10000).encode())
    stream.seek(5)
    _publish(package({"docProps/core.xml": "<properties/>"}), stream, {})
    assert SENTINEL.encode() not in stream.getvalue()
    verify_ooxml(stream.getvalue())


def test_nonseekable_output_stream_is_refused_before_writing():
    from openmed.multimodal.ooxml_residual import _publish

    class Sink:
        def seekable(self):
            return False

        def write(self, data):
            pytest.fail("must refuse before writing")

    with pytest.raises(OoxmlResidualError, match="invalid_destination"):
        _publish(package({"docProps/core.xml": "<properties/>"}), Sink(), {})
