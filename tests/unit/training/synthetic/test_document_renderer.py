"""Actual synthetic raster geometry, corruption and reproduction controls."""

import json
import math
import re
import socket
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image, ImageChops, ImageFont

from openmed.multimodal.evidence.fused_graph import NodeKind
from openmed.training.synthetic import document_renderer as renderer
from openmed.training.synthetic.burned_in import BurnedInFont
from openmed.training.synthetic.document_renderer import (
    DocumentLayout,
    DocumentRenderError,
    DocumentRenderLimits,
    DocumentRenderSpec,
    ScanArtifacts,
    WordRole,
    WordVisibility,
    render_synthetic_document,
    reproduce_synthetic_document,
)
from openmed.training.synthetic.note_grammar import (
    GRAMMARS,
    render_synthetic_note,
    synthetic_note_facts,
)


@pytest.fixture(scope="module")
def clean():
    return render_synthetic_document(
        spec=DocumentRenderSpec(seed=42, layout=DocumentLayout.LETTER)
    )


@pytest.mark.parametrize("layout", list(DocumentLayout))
@pytest.mark.parametrize("grammar", sorted(GRAMMARS))
def test_layouts_preserve_every_generated_word_and_fact(layout, grammar):
    document = render_synthetic_document(
        spec=DocumentRenderSpec(seed=7, layout=layout, grammar=grammar)
    )
    owned = {key for _, keys in GRAMMARS[grammar] for key in keys}
    note = render_synthetic_note(
        grammar,
        tuple(fact for fact in synthetic_note_facts(seed=7) if fact.key in owned),
    )
    words = [word for page in document.pages for word in page.words]
    assert note.text in document.text
    assert [word.reading_order for word in words] == list(range(len(words)))
    assert sorted(word.span for word in words) == [
        match.span() for match in re.finditer(r"\S+", document.text)
    ]
    for span in note.gold_spans:
        assert any(
            document.text[gold.start : gold.end] == span.text
            and gold.fact_ref == span.metadata["fact_id"]
            and gold.label == span.label
            for gold in document.spans
        )
    for page in document.pages:
        assert page.handwriting is not None
        assert any(word.role is WordRole.STAMP for word in page.words)
        assert any(
            word.role is WordRole.PHI_OVERLAY and word.is_phi for word in page.words
        )
        assert all(word.visibility is WordVisibility.VISIBLE for word in page.words)
        assert all(word.clean_box == word.visible_box for word in page.words)
        assert len(page.graph.nodes) == 1 + 2 * len(page.words)


def test_clean_masks_are_tight_and_match_actual_painted_word_pixels(clean):
    page = clean.pages[0]
    with Image.open(BytesIO(page.png)) as image:
        for word in page.words:
            with Image.open(BytesIO(word.clean_mask_png)) as mask:
                x0, y0, x1, y1 = word.clean_box
                assert mask.size == (x1 - x0, y1 - y0)
                assert mask.getbbox() == (0, 0, *mask.size)
                color = (150, 35, 45) if word.role is WordRole.STAMP else (25, 34, 43)
                with Image.new("RGB", mask.size, "white") as expected:
                    expected.paste(color, (0, 0, *mask.size), mask)
                    with image.crop(word.clean_box) as actual:
                        assert ImageChops.difference(actual, expected).getbbox() is None


def test_seed_and_runtime_receipts_reproduce_actual_payloads(clean):
    again = reproduce_synthetic_document(clean)
    assert again == clean
    changed = render_synthetic_document(spec=replace(clean.spec, seed=43))
    assert changed.text != clean.text
    assert changed.pages[0].png != clean.pages[0].png
    assert changed.digest != clean.digest


def test_visual_templates_are_distinct_even_at_same_size():
    images = {
        render_synthetic_document(
            spec=DocumentRenderSpec(seed=42, layout=layout, page_size=(768, 1024))
        )
        .pages[0]
        .png
        for layout in DocumentLayout
    }
    assert len(images) == len(DocumentLayout)


def test_label_pagination_orders_each_page_before_the_next():
    document = render_synthetic_document(
        spec=DocumentRenderSpec(seed=42, layout=DocumentLayout.LABEL)
    )
    assert len(document.pages) > 1
    for page, following in zip(document.pages, document.pages[1:]):
        assert page.words[-1].role is WordRole.FOOTER
        assert page.words[-1].reading_order + 1 == following.words[0].reading_order
    body = [
        word
        for page in document.pages
        for word in page.words
        if word.role is WordRole.BODY
    ]
    assert [word.span for word in body] == sorted(word.span for word in body)


@pytest.mark.parametrize(
    ("artifacts", "operation"),
    [
        (ScanArtifacts(skew_degrees=4), "skew"),
        (ScanArtifacts(blur_radius=1.2), "blur"),
        (ScanArtifacts(scan_strength=16), "scan"),
        (ScanArtifacts(occlusion=(0, 0, 0.5, 0.5)), "occlude"),
        (ScanArtifacts(jpeg_quality=15), "jpeg"),
    ],
)
def test_each_artifact_executes_and_binds_changed_pixels(clean, artifacts, operation):
    changed = render_synthetic_document(spec=replace(clean.spec, artifacts=artifacts))
    assert changed.pages[0].clean_png == clean.pages[0].clean_png
    assert changed.pages[0].png != clean.pages[0].png
    original = next(
        stage for stage in clean.pages[0].stages if stage.operation == operation
    )
    actual = next(
        stage for stage in changed.pages[0].stages if stage.operation == operation
    )
    assert actual.output_digest != original.output_digest
    assert actual.parameters_digest != original.parameters_digest
    assert reproduce_synthetic_document(changed) == changed


def test_skew_and_occlusion_move_geometry_without_deleting_phi(clean):
    artifacts = ScanArtifacts(skew_degrees=8, blur_radius=1, occlusion=(0, 0, 0.2, 0.4))
    changed = render_synthetic_document(spec=replace(clean.spec, artifacts=artifacts))
    assert changed.spans == clean.spans
    assert len(changed.pages[0].words) == len(clean.pages[0].words)
    assert any(
        word.visibility is WordVisibility.PARTIAL for word in changed.pages[0].words
    )
    assert any(word.clipped for word in changed.pages[0].words)
    for word in changed.pages[0].words:
        x, y = word.clean_box[:2]
        assert word.transformed_polygon[0] == (
            x + math.tan(math.radians(8)) * (y - 512),
            float(y),
        )
        if word.visible_box:
            x0, y0, x1, y1 = word.visible_box
            with Image.open(BytesIO(word.visible_mask_png)) as mask:
                assert mask.size == (x1 - x0, y1 - y0)
                assert mask.getbbox() == (0, 0, *mask.size)


def test_total_occlusion_retains_hidden_gold_but_not_final_ocr(clean):
    changed = render_synthetic_document(
        spec=replace(clean.spec, artifacts=ScanArtifacts(occlusion=(0, 0, 1, 1)))
    )
    assert changed.spans == clean.spans
    page = changed.pages[0]
    assert all(word.visibility is WordVisibility.OCCLUDED for word in page.words)
    assert all(
        word.visible_box is None and word.visible_fraction == 0 for word in page.words
    )
    assert any(word.is_phi for word in page.words)
    assert len(page.graph.nodes) == 1
    assert any(node.kind is NodeKind.OCR for node in page.clean_graph.nodes)
    with Image.open(BytesIO(page.png)) as image:
        assert image.getextrema() == ((255, 255),) * 3


def test_optional_marks_can_be_disabled_without_omitting_body():
    document = render_synthetic_document(
        spec=DocumentRenderSpec(
            seed=1, stamp=False, phi_overlay=False, handwriting_placeholder=False
        )
    )
    assert all(page.handwriting is None for page in document.pages)
    assert all(
        word.role in (WordRole.BODY, WordRole.FOOTER)
        for page in document.pages
        for word in page.words
    )


def test_large_font_reserves_footer_space_and_keeps_placeholder_off_words():
    document = render_synthetic_document(
        spec=DocumentRenderSpec(
            seed=42,
            layout=DocumentLayout.LABEL,
            page_size=(320, 500),
            font_size=28,
        )
    )
    for page in document.pages:
        px0, py0, px1, py1 = page.handwriting.clean_box
        for word in page.words:
            x0, y0, x1, y1 = word.clean_box
            assert x1 <= px0 or px1 <= x0 or y1 <= py0 or py1 <= y0


@pytest.mark.parametrize(
    "limits",
    [
        DocumentRenderLimits(max_pages=1),
        DocumentRenderLimits(max_words=2),
        DocumentRenderLimits(max_page_pixels=1),
        DocumentRenderLimits(max_mask_work_pixels=1),
        DocumentRenderLimits(max_png_bytes=1),
    ],
)
def test_resource_budgets_refuse_whole_result(limits):
    with pytest.raises(
        DocumentRenderError, match="^synthetic_document_render_rejected$"
    ):
        render_synthetic_document(
            spec=DocumentRenderSpec(seed=42, layout=DocumentLayout.LABEL), limits=limits
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 0, "layout": "private"},
        {"seed": 0, "grammar": "private"},
        {"seed": 0, "page_size": (1, 1)},
        {"seed": 0, "font_size": 100},
        {"seed": 0, "stamp": 1},
    ],
)
def test_invalid_specs_use_fixed_errors(kwargs):
    with pytest.raises(
        DocumentRenderError, match="^synthetic_document_render_rejected$"
    ):
        DocumentRenderSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skew_degrees": float("nan")},
        {"blur_radius": 4},
        {"scan_strength": True},
        {"occlusion": (0.5, 0, 0.1, 1)},
        {"jpeg_quality": 0},
    ],
)
def test_invalid_artifacts_reject(kwargs):
    with pytest.raises(DocumentRenderError):
        ScanArtifacts(**kwargs)


def test_public_reports_omit_values_images_and_font_paths(clean):
    report = clean.to_json()
    assert json.loads(report)["clinical_validation"] is False
    for span in clean.spans:
        assert clean.text[span.start : span.end] not in report
        assert clean.text[span.start : span.end] not in repr(clean)
    assert "data:image" not in report
    assert "font_path" not in report


def test_missing_custom_font_never_falls_back_or_leaks_backend_context(tmp_path):
    secret_path = tmp_path / "private-sensitive-name.ttf"
    with pytest.raises(DocumentRenderError) as error:
        render_synthetic_document(
            spec=DocumentRenderSpec(seed=1),
            font=BurnedInFont(path=secret_path, license="OFL-1.1"),
        )
    assert str(error.value) == "synthetic_document_render_rejected"
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_explicit_local_font_uses_same_bytes_and_leaves_files_unchanged(
    tmp_path, clean
):
    path = tmp_path / "local-font.ttf"
    path.write_bytes(ImageFont.load_default(size=18).font_bytes)
    font = BurnedInFont(path=path, license="CC0-1.0")
    custom = render_synthetic_document(spec=clean.spec, font=font)
    assert custom == clean
    assert reproduce_synthetic_document(custom, font=font) == custom
    assert list(tmp_path.iterdir()) == [path]


def test_glyph_rejection_is_not_implicitly_replaced(monkeypatch):
    def reject(*args, **kwargs):
        raise RuntimeError("private backend details")

    monkeypatch.setattr(renderer, "_check_glyph_coverage", reject)
    with pytest.raises(DocumentRenderError) as error:
        render_synthetic_document(spec=DocumentRenderSpec(seed=1))
    assert error.value.__context__ is None
    assert "private" not in str(error.value)


def test_report_revalidates_mutated_metadata_before_serializing(clean):
    word = replace(clean.pages[0].words[0], transformed_polygon=(("private", 0),) * 4)
    with pytest.raises(DocumentRenderError):
        word.to_dict()
    page = replace(clean.pages[0], words=(word,) + clean.pages[0].words[1:])
    with pytest.raises(DocumentRenderError):
        replace(clean, pages=(page,)).to_json()
    with pytest.raises(DocumentRenderError):
        replace(clean, font_license="private").to_json()


@pytest.mark.parametrize("field", ["font_digest", "runtime_digest", "text"])
def test_reproduction_rejects_stale_runtime_font_or_content(clean, field):
    changed = replace(
        clean,
        **{
            field: "0" * 64
            if field != "text"
            else clean.text.replace("SYNTHETIC", "SYNTHETIX")
        },
    )
    with pytest.raises(DocumentRenderError):
        reproduce_synthetic_document(changed)


def test_edited_pixel_receipt_and_geometry_cannot_pass_reproduction(clean):
    page = clean.pages[0]
    with Image.open(BytesIO(page.png)) as image:
        image.putpixel((0, 0), (0, 0, 0))
        stream = BytesIO()
        image.save(stream, format="PNG")
    changed = replace(clean, pages=(replace(page, png=stream.getvalue()),))
    with pytest.raises(DocumentRenderError):
        reproduce_synthetic_document(changed)
    word = replace(page.words[0], visible_fraction=0.5)
    with pytest.raises(DocumentRenderError):
        replace(clean, pages=(replace(page, words=(word,) + page.words[1:]),)).to_json()


def test_combined_corruptions_are_offline_and_deterministic(monkeypatch, tmp_path):
    def reject_network(*args, **kwargs):
        raise AssertionError("unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.chdir(tmp_path)
    spec = DocumentRenderSpec(
        seed=23,
        layout=DocumentLayout.LABEL,
        artifacts=ScanArtifacts(
            skew_degrees=-4,
            blur_radius=1.3,
            scan_strength=8,
            occlusion=(0.1, 0.1, 0.2, 0.5),
            jpeg_quality=25,
        ),
    )
    document = render_synthetic_document(spec=spec)
    assert reproduce_synthetic_document(document) == document
    assert all(page.clean_png != page.png for page in document.pages)
    assert list(tmp_path.iterdir()) == []


def test_executable_public_guide():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    source = (root / "docs/training/synthetic/document-rendering.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, "document-rendering-guide", "exec"), {})
