"""Real optional parsers retain text and provenance after ZIP admission."""

import zipfile
from contextlib import contextmanager

import pytest

from openmed.multimodal import documents_docx, epub, odt, pptx, xlsx

pytestmark = pytest.mark.integration


@contextmanager
def unchecked_zip(source, policy=None):
    """Negative control: reproduce the former package handoff."""
    with zipfile.ZipFile(source) as archive:
        yield archive


def create_document(kind, path):
    if kind == "docx":
        document = pytest.importorskip("docx").Document()
        paragraph = document.add_paragraph()
        paragraph.add_run("Synthetic ")
        paragraph.add_run("Jane Roe")
        document.save(path)
    elif kind == "pptx":
        library = pytest.importorskip("pptx")
        document = library.Presentation()
        slide = document.slides.add_slide(document.slide_layouts[6])
        frame = slide.shapes.add_textbox(0, 0, 1000000, 1000000).text_frame
        frame.paragraphs[0].add_run().text = "Synthetic "
        frame.paragraphs[0].add_run().text = "Jane Roe"
        document.save(path)
    elif kind == "xlsx":
        library = pytest.importorskip("openpyxl")
        document = library.Workbook()
        document.active.append(["notes", "age"])
        document.active.append(["Synthetic Jane Roe", 42])
        document.save(path)
        document.close()
    else:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if kind == "odt":
                archive.writestr(
                    "content.xml",
                    """
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:text><text:p>Synthetic <text:span>Jane Roe</text:span>
 </text:p></office:text></office:body></office:document-content>""",
                )
            else:
                archive.writestr(
                    "META-INF/container.xml",
                    """
<container><rootfiles><rootfile full-path="package.opf"
 media-type="application/oebps-package+xml"/></rootfiles></container>""",
                )
                archive.writestr(
                    "package.opf",
                    """
<package><manifest><item id="chapter" href="chapter.xhtml"
 media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="chapter"/></spine></package>""",
                )
                archive.writestr(
                    "chapter.xhtml",
                    "<html><body><p>Synthetic <b>Jane Roe</b></p></body></html>",
                )


@pytest.mark.parametrize("kind", ["docx", "pptx", "epub", "odt", "xlsx"])
def test_text_offsets_and_coordinates_identical_to_unchecked_baseline(
    tmp_path,
    monkeypatch,
    kind,
):
    source = tmp_path / f"synthetic.{kind}"
    create_document(kind, source)
    module = documents_docx if kind == "docx" else globals()[kind]
    if kind == "xlsx":
        library = pytest.importorskip("openpyxl")

        def extract():
            return xlsx.redact_xlsx(
                source, tmp_path / "output.xlsx", cell_deidentifier=lambda text: text
            )

        guarded = extract()
        workbook = library.load_workbook(guarded.output_path)
        guarded_cells = list(workbook.active.values)
        workbook.close()
        monkeypatch.setattr(module, "admitted_zip", unchecked_zip)
        baseline = extract()
        workbook = library.load_workbook(baseline.output_path)
        baseline_cells = list(workbook.active.values)
        workbook.close()
        assert guarded_cells == baseline_cells
        assert guarded.to_document() == baseline.to_document()
        assert guarded.redaction_report == baseline.redaction_report
    else:
        extract = getattr(module, f"extract_{kind}")
        guarded = extract(source)
        monkeypatch.setattr(module, "admitted_zip", unchecked_zip)
        baseline = extract(source)
        assert guarded == baseline
        assert "Synthetic Jane Roe" in guarded.text
        assert guarded.spans
        for span in guarded.spans:
            assert guarded.text_for(span) == baseline.text[span.start : span.end]
