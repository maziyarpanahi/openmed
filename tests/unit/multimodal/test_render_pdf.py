"""Regression tests for deterministic, layout-preserving PDF redaction."""

from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import pytest

from openmed.multimodal import (
    PdfLayoutFidelityError,
    PdfRedactionRegion,
    PdfRedactionResult,
    RedactedTextRemovalError,
    assert_redacted_text_removed,
    extract_pdf,
    measure_pdf_layout_fidelity,
    project_text_spans,
    render_redacted_pdf,
    verify_redacted_pdf,
    verify_redacted_text_removed,
    write_redacted_pdf,
)


def _load_fixture_builder():
    path = Path(__file__).parent / "fixtures" / "redaction_pdfs.py"
    spec = importlib.util.spec_from_file_location("render_redaction_pdfs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_name(path: Path):
    document = extract_pdf(path)
    start = document.text.index("John")
    end = document.text.index("Doe") + len("Doe")
    regions = project_text_spans(document, [(start, end)])
    assert regions
    return regions


def test_render_removes_source_text_and_preserves_non_phi_layout(tmp_path):
    pdfplumber = pytest.importorskip("pdfplumber")
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())
    regions = _project_name(source)

    result = render_redacted_pdf(source, output, regions)

    assert isinstance(result, PdfRedactionResult)
    assert result.passed
    assert result.page_count == 1
    assert result.region_count == 1
    assert result.region_fidelity.passed
    assert result.text_removal.passed
    assert result.layout_fidelity.passed
    assert result.layout_fidelity.pages[0].outside_changed_fraction == 0.0
    assert result.region_fidelity.regions[0].bbox[0] < regions[0].bbox[0]
    assert result.layout_fidelity.to_dict()["limits"] == {
        "max_pages": 100,
        "max_page_pixels": 40_000_000,
        "max_total_pixels": 100_000_000,
    }
    assert output.is_file()

    with pdfplumber.open(output) as pdf:
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        extracted = page.extract_text() or ""
        assert "Patient" in extracted
        assert "MRN 12345" in extracted
        assert "John" not in extracted
        assert "Doe" not in extracted
        assert page.width == pytest.approx(612.0)
        assert page.height == pytest.approx(792.0)
        assert len(page.images) == 1
        assert any(rect.get("fill") for rect in page.rects)

    # The rebuilt PDF never copies the original text operators or plaintext.
    output_bytes = output.read_bytes()
    assert b"John" not in output_bytes
    assert b"Doe" not in output_bytes


def test_render_is_byte_deterministic_and_report_is_phi_safe(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "Synthetic_Patient_John_Doe.pdf"
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    source.write_bytes(fx.original_pdf_bytes())
    regions = _project_name(source)

    first_result = render_redacted_pdf(source, first, regions)
    second_result = write_redacted_pdf(source, second, reversed(regions))

    assert first.read_bytes() == second.read_bytes()
    assert first_result.output_sha256 == second_result.output_sha256
    payload = json.dumps(first_result.to_dict(), sort_keys=True)
    assert "John" not in payload
    assert "Doe" not in payload
    assert "Synthetic_Patient" not in payload
    assert "first.pdf" not in payload


def test_render_does_not_copy_source_metadata_or_auxiliary_catalog_data(tmp_path):
    pikepdf = pytest.importorskip("pikepdf")
    fx = _load_fixture_builder()
    plain = tmp_path / "plain.pdf"
    source = tmp_path / "source-with-synthetic-metadata.pdf"
    output = tmp_path / "redacted.pdf"
    plain.write_bytes(fx.original_pdf_bytes())
    with pikepdf.open(plain) as pdf:
        pdf.docinfo["/Title"] = "Synthetic Patient John Doe"
        pdf.Root.Names = pikepdf.Dictionary()
        pdf.save(source)

    result = render_redacted_pdf(source, output, _project_name(source))

    assert result.passed
    with pikepdf.open(output) as pdf:
        assert not dict(pdf.docinfo)
        assert "/Metadata" not in pdf.Root
        assert "/Names" not in pdf.Root
    assert b"Synthetic Patient John Doe" not in output.read_bytes()


def test_multipage_render_preserves_pagination_and_page_geometry(tmp_path):
    pdfplumber = pytest.importorskip("pdfplumber")
    fx = _load_fixture_builder()
    source = tmp_path / "multipage.pdf"
    output = tmp_path / "multipage-redacted.pdf"
    source.write_bytes(fx.multipage_pdf_bytes())

    result = render_redacted_pdf(source, output, _project_name(source))

    assert result.passed
    assert result.page_count == 2
    assert result.layout_fidelity.pagination_preserved
    assert len(result.layout_fidelity.pages) == 2
    with pdfplumber.open(output) as pdf:
        assert [(float(page.width), float(page.height)) for page in pdf.pages] == [
            (612.0, 792.0),
            (420.0, 595.0),
        ]
        assert "Follow up notes remain stable" in (pdf.pages[1].extract_text() or "")


def test_layout_measurement_rejects_non_redacted_content_shift(tmp_path):
    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    shifted = tmp_path / "shifted.pdf"
    original.write_bytes(fx.original_pdf_bytes())
    shifted.write_bytes(fx.shifted_non_phi_pdf_bytes())
    regions = _project_name(original)

    report = measure_pdf_layout_fidelity(
        original,
        shifted,
        regions,
        render_dpi=72,
        max_outside_changed_fraction=0.0001,
    )

    assert not report.passed
    assert report.failing_pages
    page = report.pages[0]
    assert page.size_preserved
    assert page.outside_changed_pixel_count > 0
    assert page.outside_changed_fraction > report.max_outside_changed_fraction
    with pytest.raises(PdfLayoutFidelityError):
        measure_pdf_layout_fidelity(
            original,
            shifted,
            regions,
            render_dpi=72,
            max_outside_changed_fraction=0.0001,
            strict=True,
        )


def test_global_text_verifier_catches_source_text_moved_elsewhere(tmp_path):
    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    moved = tmp_path / "moved.pdf"
    original.write_bytes(fx.original_pdf_bytes())
    moved.write_bytes(fx.moved_leak_pdf_bytes())
    regions = _project_name(original)

    report = verify_redacted_text_removed(original, moved, regions)

    assert not report.passed
    assert report.regions[0].residual_word_count == 2
    payload = json.dumps(report.to_dict())
    assert "John" not in payload
    assert "Doe" not in payload
    with pytest.raises(RedactedTextRemovalError):
        assert_redacted_text_removed(original, moved, regions)


def test_global_text_verifier_catches_separated_selected_words(tmp_path):
    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    moved = tmp_path / "separated.pdf"
    original.write_bytes(fx.original_pdf_bytes())
    moved.write_bytes(fx.separated_moved_leak_pdf_bytes())

    report = verify_redacted_text_removed(original, moved, _project_name(original))

    assert not report.passed
    assert report.regions[0].residual_word_count == 2


def test_global_text_verifier_allows_unrelated_identical_words(tmp_path):
    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    clean = tmp_path / "clean.pdf"
    original.write_bytes(fx.duplicate_token_original_pdf_bytes())
    clean.write_bytes(fx.duplicate_token_clean_redaction_pdf_bytes())

    report = verify_redacted_text_removed(original, clean, _project_name(original))

    assert report.passed
    assert report.regions[0].residual_word_count == 0


def test_region_labels_are_hashed_in_serialized_evidence(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())
    projected = _project_name(source)[0]
    marker = "SYNTHETIC_PATIENT_LABEL"
    region = PdfRedactionRegion(projected.page, projected.bbox, marker)
    labeled_region = {
        "page": projected.page,
        "bbox": projected.bbox,
        "label": marker,
    }

    result = render_redacted_pdf(source, output, [region])
    region_payload = region.to_dict()
    result_payload = json.dumps(result.to_dict(), sort_keys=True)
    fidelity_payload = json.dumps(
        verify_redacted_pdf(source, output, [labeled_region]).to_dict()
    )
    removal_payload = json.dumps(
        verify_redacted_text_removed(source, output, [labeled_region]).to_dict()
    )

    assert "label" not in region_payload
    assert "label_sha256" in region_payload
    assert marker not in json.dumps(region_payload)
    assert marker not in result_payload
    assert marker not in fidelity_payload
    assert marker not in removal_payload
    assert "label_sha256" in fidelity_payload
    assert "label_sha256" in removal_payload


def test_render_rejects_type3_fonts_before_publication(tmp_path):
    pikepdf = pytest.importorskip("pikepdf")
    fx = _load_fixture_builder()
    plain = tmp_path / "plain.pdf"
    source = tmp_path / "type3.pdf"
    output = tmp_path / "redacted.pdf"
    plain.write_bytes(fx.original_pdf_bytes())
    with pikepdf.open(plain) as pdf:
        pdf.pages[0].Resources.Font.F1.Subtype = pikepdf.Name("/Type3")
        pdf.save(source)

    with pytest.raises(ValueError, match="Type 3 fonts"):
        render_redacted_pdf(source, output, _project_name(plain))

    assert not output.exists()


def test_render_enforces_page_pixel_budget(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())

    with pytest.raises(ValueError, match="max_page_pixels=1000"):
        render_redacted_pdf(
            source,
            output,
            _project_name(source),
            max_page_pixels=1000,
        )

    assert not output.exists()


def test_render_enforces_page_count_budget(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "multipage.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.multipage_pdf_bytes())

    with pytest.raises(ValueError, match="max_pages=1"):
        render_redacted_pdf(
            source,
            output,
            _project_name(source),
            max_pages=1,
        )

    assert not output.exists()


def test_render_enforces_total_pixel_budget(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())

    with pytest.raises(ValueError, match="max_total_pixels=1000"):
        render_redacted_pdf(
            source,
            output,
            _project_name(source),
            max_total_pixels=1000,
        )

    assert not output.exists()


def test_render_enforces_region_count_budget(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())
    regions = [
        _project_name(source)[0],
        {"page": 0, "bbox": (300.0, 300.0, 320.0, 320.0)},
    ]

    with pytest.raises(ValueError, match="max_regions=1"):
        render_redacted_pdf(source, output, regions, max_regions=1)

    assert not output.exists()


def test_render_errors_omit_source_paths_and_backend_causes(tmp_path):
    source = tmp_path / "Synthetic_Patient_John_Doe.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(b"not a valid PDF")

    with pytest.raises(RuntimeError) as exc_info:
        render_redacted_pdf(
            source,
            output,
            [{"page": 0, "bbox": (1.0, 1.0, 2.0, 2.0)}],
        )

    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert source.name not in str(exc_info.value)
    assert source.name not in formatted
    assert exc_info.value.__cause__ is None
    assert not output.exists()


@pytest.mark.parametrize(
    "region, message",
    [
        ({"page": 1, "bbox": (1, 1, 2, 2)}, "outside the 1-page PDF"),
        ({"page": 0, "bbox": (-1, 1, 2, 2)}, "outside page bounds"),
        ({"page": 0, "bbox": (2, 1, 2, 3)}, "positive width and height"),
    ],
)
def test_render_rejects_invalid_regions_without_creating_output(
    tmp_path, region, message
):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    source.write_bytes(fx.original_pdf_bytes())

    with pytest.raises(ValueError, match=message):
        render_redacted_pdf(source, output, [region])

    assert not output.exists()


def test_render_refuses_accidental_overwrite(tmp_path):
    fx = _load_fixture_builder()
    source = tmp_path / "source.pdf"
    output = tmp_path / "existing.pdf"
    source.write_bytes(fx.original_pdf_bytes())
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        render_redacted_pdf(source, output, _project_name(source))

    assert output.read_bytes() == b"keep me"


def test_text_removal_fails_closed_when_source_region_has_no_text(tmp_path):
    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    clean = tmp_path / "clean.pdf"
    original.write_bytes(fx.original_pdf_bytes())
    clean.write_bytes(fx.clean_redaction_pdf_bytes())
    empty_region = [{"page": 0, "bbox": (300.0, 300.0, 320.0, 320.0)}]

    report = verify_redacted_text_removed(original, clean, empty_region)

    assert not report.passed
    assert not report.regions[0].source_text_found
    assert report.regions[0].source_sha256 == ()
