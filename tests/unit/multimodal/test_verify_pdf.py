"""Tests for redacted-PDF text-layer leakage and visual fidelity verification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.multimodal import (
    PdfFidelityReport,
    RedactionFidelityError,
    verify_redacted_pdf,
)

# A redaction region over a synthetic name; coordinates are pdfplumber top-based.
REGION = {"page": 0, "bbox": (100.0, 60.0, 180.0, 74.0), "label": "PERSON"}


def _word(text, x0, top, x1, bottom):
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _rect(x0, top, x1, bottom, *, fill=True):
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom, "fill": fill}


class _FakePage:
    def __init__(self, words=(), rects=()):
        self._words = list(words)
        self.rects = list(rects)

    def extract_words(self, **kwargs):
        return list(self._words)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _install_fake_pdfplumber(monkeypatch, layouts):
    """Install a fake ``pdfplumber`` whose ``open`` dispatches on the path."""

    def _open(path):
        return _FakePdf(layouts[str(path)])

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=_open))


def test_detects_residual_text_under_box(monkeypatch):
    # Box drawn, but "John Doe" is still selectable underneath -> leak.
    redacted = [
        _FakePage(
            words=[_word("John", 110, 62, 140, 72), _word("Doe", 145, 62, 175, 72)],
            rects=[_rect(100, 58, 190, 80)],
        )
    ]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION])

    assert isinstance(report, PdfFidelityReport)
    region = report.regions[0]
    assert region.residual_text_found
    assert region.residual_word_count == 2
    assert region.redaction_box_present  # a box is present, yet text leaked
    assert not region.passed
    assert not report.passed
    assert report.residual_text_regions


def test_detects_unchanged_region_no_box(monkeypatch):
    # No visible box was drawn over the region -> fails closed.
    redacted = [_FakePage(words=[], rects=[])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION])

    region = report.regions[0]
    assert not region.residual_text_found
    assert not region.redaction_box_present
    assert not region.visual_ok
    assert not region.passed
    assert not report.passed


def test_correct_redaction_passes(monkeypatch):
    # Text scrubbed AND an opaque box drawn -> passes.
    redacted = [_FakePage(words=[], rects=[_rect(100, 58, 190, 80)])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION])

    region = report.regions[0]
    assert not region.residual_text_found
    assert region.redaction_box_present
    # The fake page has no to_image(), so the rasterizer raises a non-IO error
    # and the visual check falls back to the vector (box-presence) path.
    assert region.visual_method == "vector"
    assert region.pixels_changed is None
    assert region.passed
    assert report.passed


def test_stroke_only_border_is_not_a_redaction_box(monkeypatch):
    # A black table border reports fill=False but a non_stroking_color; it must
    # NOT count as a redaction box (regression for the false-positive fix).
    border = _rect(90, 55, 200, 85, fill=False)
    border["stroke"] = True
    border["non_stroking_color"] = 0
    redacted = [_FakePage(words=[], rects=[border])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION])
    region = report.regions[0]
    assert not region.redaction_box_present  # a border is not a redaction
    assert not region.passed
    assert not report.passed


def test_missing_original_fails_closed_when_rasterizing(monkeypatch):
    # A raster backend selected but a PDF that cannot be read must not silently
    # report a clean pass; it surfaces an error (regression for the fails-open fix).
    redacted = [_FakePage(words=[], rects=[_rect(100, 58, 190, 80)])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    def raise_ioerror(path, page, bbox):
        raise FileNotFoundError(str(path))

    with pytest.raises(FileNotFoundError):
        verify_redacted_pdf(
            "original.pdf", "redacted.pdf", [REGION], rasterizer=raise_ioerror
        )


def test_page_bbox_tuple_region_is_accepted(monkeypatch):
    redacted = [
        _FakePage(
            words=[_word("John", 110, 62, 175, 72)], rects=[_rect(100, 58, 190, 80)]
        )
    ]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    # (page, bbox) as a tuple must be treated as a direct region, not a char span.
    report = verify_redacted_pdf(
        "original.pdf", "redacted.pdf", [(0, (100.0, 60.0, 180.0, 74.0))]
    )
    assert len(report.regions) == 1
    assert report.regions[0].residual_text_found
    assert not report.passed


def test_pixel_path_requires_region_to_change(monkeypatch):
    redacted = [_FakePage(words=[], rects=[_rect(100, 58, 190, 80)])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    def changed(path, page, bbox):
        return b"redacted" if str(path) == "redacted.pdf" else b"original"

    changed_report = verify_redacted_pdf(
        "original.pdf", "redacted.pdf", [REGION], rasterizer=changed
    )
    region = changed_report.regions[0]
    assert region.visual_method == "pixel"
    assert region.pixels_changed is True
    assert region.passed

    def unchanged(path, page, bbox):
        return b"identical"

    unchanged_report = verify_redacted_pdf(
        "original.pdf", "redacted.pdf", [REGION], rasterizer=unchanged
    )
    region = unchanged_report.regions[0]
    assert region.pixels_changed is False
    assert not region.passed  # box present but pixels never changed
    assert not unchanged_report.passed


def test_strict_raises_on_leak(monkeypatch):
    redacted = [
        _FakePage(
            words=[_word("John", 110, 62, 175, 72)],
            rects=[_rect(100, 58, 190, 80)],
        )
    ]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    with pytest.raises(RedactionFidelityError) as excinfo:
        verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION], strict=True)
    assert excinfo.value.report.residual_text_regions


def test_report_is_phi_safe(monkeypatch):
    redacted = [
        _FakePage(
            words=[_word("John", 110, 62, 175, 72)],
            rects=[_rect(100, 58, 190, 80)],
        )
    ]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": redacted})

    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [REGION])
    payload = report.to_dict()

    assert payload["check"] == "redacted_pdf_fidelity"
    assert payload["passed"] is False
    assert payload["region_count"] == 1
    assert payload["regions"][0]["residual_sha256"]  # residual recorded as a hash
    # The report must never carry plaintext identifiers.
    assert "John" not in json.dumps(payload)


def test_char_span_input_resolves_against_original(monkeypatch):
    original = [
        _FakePage(
            words=[
                _word("Patient", 72, 60, 110, 72),
                _word("John", 115, 60, 150, 72),
                _word("Doe", 152, 60, 180, 72),
            ]
        )
    ]
    redacted = [
        _FakePage(
            words=[_word("John", 115, 60, 150, 72), _word("Doe", 152, 60, 180, 72)],
            rects=[_rect(110, 56, 185, 76)],
        )
    ]
    _install_fake_pdfplumber(
        monkeypatch, {"original.pdf": original, "redacted.pdf": redacted}
    )

    # Text is "Patient John Doe"; the "John Doe" character span is 8..16.
    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [(8, 16)])

    region = report.regions[0]
    # A bare (start, end) char span carries no label, so it projects to None.
    assert region.label is None
    assert region.residual_text_found  # the name survived under the box
    assert not report.passed


def test_empty_spans_fails_closed(monkeypatch):
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": [_FakePage()]})
    report = verify_redacted_pdf("original.pdf", "redacted.pdf", [])
    assert not report.passed  # nothing verified -> not a pass


def _write_spans(tmp_path, payload):
    path = tmp_path / "spans.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_exit_codes_and_output(monkeypatch, tmp_path, capsys):
    from openmed.cli.verify_pdf import run_from_args

    leaky = [
        _FakePage(
            words=[_word("John", 110, 62, 175, 72)], rects=[_rect(100, 58, 190, 80)]
        )
    ]
    clean = [_FakePage(words=[], rects=[_rect(100, 58, 190, 80)])]

    # Leak -> exit 1, and the report JSON is written to --output.
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": leaky})
    spans = _write_spans(tmp_path, [{"page": 0, "bbox": [100, 60, 180, 74]}])
    out = tmp_path / "report.json"
    args = SimpleNamespace(
        original=Path("original.pdf"),
        redacted=Path("redacted.pdf"),
        spans=spans,
        output=out,
    )
    assert run_from_args(args) == 1
    written = json.loads(out.read_text())
    assert written["passed"] is False
    assert "John" not in out.read_text()  # report file stays PHI-safe

    # Clean -> exit 0.
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": clean})
    args = SimpleNamespace(
        original=Path("original.pdf"),
        redacted=Path("redacted.pdf"),
        spans=spans,
        output=None,
    )
    assert run_from_args(args) == 0


def test_cli_bad_spans_file_exits_2(tmp_path):
    from openmed.cli.verify_pdf import run_from_args

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    args = SimpleNamespace(
        original=Path("o.pdf"), redacted=Path("r.pdf"), spans=bad, output=None
    )
    assert run_from_args(args) == 2


def test_cli_spans_object_wrapper(monkeypatch, tmp_path):
    from openmed.cli.verify_pdf import run_from_args

    clean = [_FakePage(words=[], rects=[_rect(100, 58, 190, 80)])]
    _install_fake_pdfplumber(monkeypatch, {"redacted.pdf": clean})
    # spans provided as {"regions": [...]} instead of a bare list.
    spans = _write_spans(
        tmp_path, {"regions": [{"page": 0, "bbox": [100, 60, 180, 74]}]}
    )
    args = SimpleNamespace(
        original=Path("original.pdf"),
        redacted=Path("redacted.pdf"),
        spans=spans,
        output=None,
    )
    assert run_from_args(args) == 0


def _load_fixture_builder():
    path = Path(__file__).parent / "fixtures" / "redaction_pdfs.py"
    spec = importlib.util.spec_from_file_location("redaction_pdfs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pdfplumber_end_to_end(tmp_path):
    pytest.importorskip("pdfplumber")
    from openmed.multimodal import extract_pdf, project_text_spans

    fx = _load_fixture_builder()
    original = tmp_path / "original.pdf"
    leaky = tmp_path / "leaky_redaction.pdf"
    clean = tmp_path / "clean_redaction.pdf"
    original.write_bytes(fx.original_pdf_bytes())
    leaky.write_bytes(fx.leaky_redaction_pdf_bytes())
    clean.write_bytes(fx.clean_redaction_pdf_bytes())

    document = extract_pdf(original)
    start = document.text.index("John")
    end = document.text.index("Doe") + len("Doe")
    regions = project_text_spans(document, [(start, end)])
    assert regions

    leaky_report = verify_redacted_pdf(original, leaky, regions)
    assert not leaky_report.passed
    assert leaky_report.residual_text_regions

    clean_report = verify_redacted_pdf(original, clean, regions)
    assert clean_report.passed

    # A "redacted" file identical to the original means nothing was applied.
    unchanged_report = verify_redacted_pdf(original, original, regions)
    assert not unchanged_report.passed
