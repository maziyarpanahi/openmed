"""Synthetic coordinated redaction, source binding and fail-closed export controls."""

import json
import re
import socket
from dataclasses import replace
from io import BytesIO

import pytest

from openmed.core.schemas.span import OpenMedSpan
from openmed.multimodal.ecg.annotation_privacy import (
    AnnotationField,
    AnnotationPage,
    AnnotationPolicy,
    AnnotationPrivacyError,
    AnnotationReview,
    EcgAnnotationSource,
    LocalAnnotationDetector,
    LocalAnnotationOcr,
    PageRegion,
    PrivacyReason,
    RegionKind,
    SurfaceKind,
    export_annotation_privacy,
    prepare_annotation_privacy,
)
from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind,
    LeadFindingCode,
    LeadLayout,
    LeadObservation,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform
from openmed.multimodal.ocr import OcrResult, OcrWord
from openmed.multimodal.ocr_tesseract_stream import StreamingTesseractEngine

Image = pytest.importorskip("PIL.Image")


def ref(index):
    return f"{index:032x}"


def span(start, end):
    return OpenMedSpan(
        doc_id="synthetic",
        start=start,
        end=end,
        text_hash="hmac-sha256:" + "0" * 64,
        entity_type="name",
        canonical_label="PERSON",
    )


def detect(text):
    return tuple(
        span(m.start(), m.end())
        for m in re.finditer("SYNTHETIC_PERSON|人物例|الاسم_التجريبي", text)
    )


def detector(call=detect, **changes):
    return replace(
        LocalAnnotationDetector(
            detect=call,
            model_digest="b" * 64,
            qualification_digest="c" * 64,
            local_only=True,
        ),
        **changes,
    )


def policy(**changes):
    return replace(
        AnnotationPolicy(
            version="synthetic-v1", evidence_digest="d" * 64, language="en"
        ),
        **changes,
    )


def regions():
    return (
        PageRegion(
            region_ref=ref(1),
            kind=RegionKind.ANNOTATION,
            surface=SurfaceKind.HEADER,
            bounds=(0, 0, 120, 30),
        ),
        PageRegion(
            region_ref=ref(2),
            kind=RegionKind.TRACE,
            bounds=(0, 30, 120, 70),
            slot=(0, 0),
        ),
        PageRegion(
            region_ref=ref(3), kind=RegionKind.BACKGROUND, bounds=(0, 70, 120, 80)
        ),
    )


def raster():
    image = Image.new("RGB", (120, 80), "white")
    # Synthetic colored tokens let the fixture recognizer inspect actual pixels.
    image.paste((200, 100, 50), (5, 5, 30, 15))
    image.paste((50, 100, 200), (60, 5, 100, 15))
    image.paste((0, 100, 0), (5, 49, 115, 51))
    image.info["comment"] = "SYNTHETIC_PERSON"
    return image


def source(*, rendered=True, fields=None):
    wave = EcgWaveform(
        (EcgLead.from_samples("I", [0, 0.1, 0, -0.1] * 25, unit="mV"),),
        100,
        EcgProvenance("array", "a" * 64),
    )
    return EcgAnnotationSource(
        source_digest="a" * 64,
        extraction_digest="e" * 64,
        filename="SYNTHETIC_PERSON-record.png",
        fields=fields
        if fields is not None
        else (
            AnnotationField(SurfaceKind.HEADER, "patient", "SYNTHETIC_PERSON"),
            AnnotationField(
                SurfaceKind.CALIBRATION, "calibration", "10 mm/mV; 25 mm/s"
            ),
            AnnotationField(SurfaceKind.ANNOTATION, "note", "Synthetic regular trace"),
        ),
        pages=(AnnotationPage(raster(), regions()),) if rendered else (),
        layout=LeadLayout.digital(("I",)),
        observations=(
            LeadObservation(
                label="I",
                row=0,
                start_seconds=0,
                duration_seconds=1,
                sample_rate_hz=100,
                sample_count=100,
            ),
        ),
        acquisition=AcquisitionKind.IMAGE_DERIVED
        if rendered
        else AcquisitionKind.DIGITAL,
        waveform=wave,
    )


def review(plan, **changes):
    return replace(
        AnnotationReview(
            plan_digest=plan.digest,
            reviewer_ref=ref(50),
            evidence_digest="f" * 64,
            reviewed_field_refs=tuple(item.field_ref for item in plan.fields),
            reviewed_region_refs=plan.region_refs,
        ),
        **changes,
    )


@pytest.fixture
def ocr(monkeypatch):
    def recognize(self, image, *, languages=None):
        assert languages == ("en",)
        words = []
        if image.getpixel((10, 10)) == (200, 100, 50):
            words.append(OcrWord("SYNTHETIC_PERSON", (5, 5, 30, 15), 0.99))
        if image.getpixel((65, 10)) == (50, 100, 200):
            words.append(OcrWord("10mm/mV", (60, 5, 100, 15), 0.99))
        return OcrResult(tuple(words))

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", recognize)
    return LocalAnnotationOcr(
        engine=StreamingTesseractEngine(),
        runtime_digest="1" * 64,
        qualification_digest="2" * 64,
    )


def run(item=None, *, ocr=None, rule=None, limits=None):
    item = source(rendered=ocr is not None) if item is None else item
    rule, limits = rule or detector(), limits or policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    return plan, export_annotation_privacy(item, limits, rule, plan, review(plan), ocr)


def assert_error(reason, action):
    with pytest.raises(AnnotationPrivacyError) as caught:
        action()
    assert caught.value.reason is reason and str(caught.value) == reason.value
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_actual_structured_and_pixel_redaction_preserves_calibration_and_waveform(ocr):
    item = source()
    original = item.pages[0].image.tobytes()
    plan, result = run(item, ocr=ocr)
    assert plan.findings == () and len(plan.pixels) == 1
    assert (
        result.waveform is item.waveform and item.pages[0].image.tobytes() == original
    )
    data = json.loads(result.structured_json)
    assert data["filename"] == "ecg-redacted" and data["page_names"] == [
        "ecg-page-0000.png"
    ]
    values = [field["value"] for field in data["fields"]]
    assert "[REDACTED]" in values and "10 mm/mV; 25 mm/s" in values
    assert "Synthetic regular trace" in values and "I" in values
    assert "SYNTHETIC_PERSON" not in result.structured_json + result.provenance_json
    with Image.open(BytesIO(result.png_pages[0])) as image:
        assert image.crop((3, 3, 32, 17)).getextrema() == ((0, 0),) * 3
        assert image.getpixel((65, 10)) == (50, 100, 200)
        assert (
            image.crop((0, 30, 120, 70)).tobytes()
            == item.pages[0].image.crop((0, 30, 120, 70)).tobytes()
        )
        assert image.info == {}
    evidence = json.loads(result.provenance_json)
    assert evidence["pixel_verification"] == "exact-decoded-raster"
    assert evidence["regions"][0][1]["slot"] == [0, 0]
    assert evidence["plan"]["pixels"][0]["region_refs"] == [ref(1)]
    assert not evidence["waveform_modified"]


@pytest.mark.parametrize(
    "kind",
    [
        SurfaceKind.HEADER,
        SurfaceKind.ANNOTATION,
        SurfaceKind.CALIBRATION,
        SurfaceKind.FOOTER,
        SurfaceKind.METADATA,
    ],
)
@pytest.mark.parametrize("value", ["SYNTHETIC_PERSON", "人物例", "الاسم_التجريبي"])
def test_every_structured_surface_and_field_name_is_redacted(kind, value):
    item = source(rendered=False, fields=(AnnotationField(kind, value, value),))
    _, result = run(item)
    field = json.loads(result.structured_json)["fields"][1]
    assert field["name"] == field["value"] == "[REDACTED]"
    assert (
        value
        not in result.structured_json + result.provenance_json + result.to_report_json()
    )


def test_sweep_covers_whole_identifier_despite_partial_model_span():
    def partial(text):
        match = re.search("123", text)
        return (span(match.start(), match.end()),) if match else ()

    item = source(
        rendered=False,
        fields=(AnnotationField(SurfaceKind.HEADER, "MRN", "123456789"),),
    )
    _, result = run(item, rule=detector(partial))
    value = json.loads(result.structured_json)["fields"][1]["value"]
    assert "123" not in value and "456789" not in value and "[REDACTED]" in value


def test_unknown_filename_is_always_replaced_without_default_model_loading():
    item = replace(
        source(rendered=False, fields=()), filename="unrecognized-private-name.edf"
    )
    _, result = run(item, rule=detector(lambda text: ()))
    assert "unrecognized" not in result.structured_json + result.provenance_json


def test_review_must_cover_every_current_field_and_region(ocr):
    item, rule, limits = source(), detector(), policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    for changed in (
        review(plan, reviewed_field_refs=()),
        review(plan, reviewed_region_refs=()),
        review(plan, plan_digest="0" * 64),
    ):
        assert_error(
            PrivacyReason.REVIEW,
            lambda: export_annotation_privacy(item, limits, rule, plan, changed, ocr),
        )


@pytest.mark.parametrize(
    "change",
    ["pixel", "field", "filename", "region", "waveform", "policy", "detector", "ocr"],
)
def test_review_cannot_be_reused_after_source_or_runtime_change(ocr, change):
    item, rule, limits = source(), detector(), policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    if change == "pixel":
        item.pages[0].image.putpixel((80, 50), (1, 2, 3))
    elif change == "field":
        item = replace(
            item,
            fields=item.fields + (AnnotationField(SurfaceKind.FOOTER, "new", "note"),),
        )
    elif change == "filename":
        item = replace(item, filename="different.png")
    elif change == "region":
        page = item.pages[0]
        item = replace(
            item,
            pages=(
                replace(
                    page,
                    regions=(replace(page.regions[0], region_ref=ref(90)),)
                    + page.regions[1:],
                ),
            ),
        )
    elif change == "waveform":
        item = replace(
            item,
            waveform=replace(
                item.waveform,
                leads=(EcgLead.from_samples("I", [0.2] * 100, unit="mV"),),
            ),
        )
    elif change == "policy":
        limits = replace(limits, evidence_digest="9" * 64)
    elif change == "detector":
        rule = replace(rule, model_digest="9" * 64)
    else:
        ocr.engine.timeout_seconds += 1
    assert_error(
        PrivacyReason.STALE,
        lambda: export_annotation_privacy(item, limits, rule, plan, review(plan), ocr),
    )


@pytest.mark.parametrize(
    "case", ["hole", "overlap", "outside", "missing-trace", "duplicate-reference"]
)
def test_page_partition_is_verified_before_any_detector_call(ocr, case):
    item, parts = source(), regions()
    if case == "hole":
        parts = parts[:2] + (replace(parts[2], bounds=(0, 71, 120, 80)),)
    elif case == "overlap":
        parts = parts[:2] + (replace(parts[2], bounds=(0, 69, 120, 80)),)
    elif case == "outside":
        parts = parts[:2] + (replace(parts[2], bounds=(0, 70, 121, 80)),)
    elif case == "missing-trace":
        parts = (
            parts[0],
            replace(parts[1], kind=RegionKind.BACKGROUND, slot=None),
            parts[2],
        )
    else:
        item = replace(item, pages=item.pages * 2)
    if case != "duplicate-reference":
        item = replace(item, pages=(replace(item.pages[0], regions=parts),))

    def forbidden(text):
        pytest.fail("uncovered image reached detector")

    assert_error(
        PrivacyReason.COVERAGE,
        lambda: prepare_annotation_privacy(item, policy(), detector(forbidden), ocr),
    )


@pytest.mark.parametrize("box", [(4, 35, 40, 45), (4, 72, 40, 78)])
def test_text_on_trace_or_background_requires_coverage_review(monkeypatch, ocr, box):
    monkeypatch.setattr(
        StreamingTesseractEngine,
        "recognize",
        lambda *a, **k: OcrResult((OcrWord("Unrecognized", box, 0.1),)),
    )
    item, rule, limits = source(), detector(), policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    assert PrivacyReason.COVERAGE in plan.findings
    assert_error(
        PrivacyReason.COVERAGE,
        lambda: export_annotation_privacy(item, limits, rule, plan, review(plan), ocr),
    )


def test_padding_cannot_erase_adjacent_calibration_or_trace(monkeypatch, ocr):
    for box in ((31, 5, 100, 15), (60, 28, 100, 32)):
        words = (
            OcrWord("SYNTHETIC_PERSON", (5, 5, 30, 15), 0.99),
            OcrWord("10mm/mV", box, 0.99),
        )
        monkeypatch.setattr(
            StreamingTesseractEngine,
            "recognize",
            lambda *a, words=words, **k: OcrResult(words),
        )
        assert (
            PrivacyReason.COVERAGE
            in prepare_annotation_privacy(source(), policy(), detector(), ocr).findings
        )


def test_post_render_residual_prevents_export(monkeypatch, ocr):
    original = StreamingTesseractEngine.recognize

    def recognize(self, image, **kwargs):
        if image.getpixel((10, 10)) == (0, 0, 0):
            return OcrResult((OcrWord("MRN:123456789", (60, 5, 100, 15), 0.99),))
        return original(self, image, **kwargs)

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", recognize)
    item, rule, limits = source(), detector(), policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    assert_error(
        PrivacyReason.RESIDUAL,
        lambda: export_annotation_privacy(item, limits, rule, plan, review(plan), ocr),
    )


def test_rewritten_values_are_checked_not_only_planned_offsets():
    def fragmented(text):
        target = "SYNTHETIC_PERSON" if "SYNTHETIC_PERSON" in text else "PERSON"
        index = text.find(target)
        return (
            (
                span(
                    index, index + (10 if target == "SYNTHETIC_PERSON" else len(target))
                ),
            )
            if index >= 0
            else ()
        )

    item, rule, limits = source(rendered=False), detector(fragmented), policy()
    plan = prepare_annotation_privacy(item, limits, rule)
    assert_error(
        PrivacyReason.RESIDUAL,
        lambda: export_annotation_privacy(item, limits, rule, plan, review(plan)),
    )


def test_unknown_or_misordered_leads_remain_held(ocr):
    for label in ("SYNTHETIC_PERSON", "II"):
        item = source()
        item = replace(item, observations=(replace(item.observations[0], label=label),))
        plan = prepare_annotation_privacy(item, policy(), detector(), ocr)
        assert PrivacyReason.LAYOUT in plan.findings
        assert_error(
            PrivacyReason.LAYOUT,
            lambda: export_annotation_privacy(
                item, policy(), detector(), plan, review(plan), ocr
            ),
        )


def test_privacy_does_not_claim_unknown_image_timing_is_resolved(ocr):
    item = replace(
        source(), waveform=None, observations=(LeadObservation(label="I", row=0),)
    )
    plan, result = run(item, ocr=ocr)
    assert LeadFindingCode.TIMING_UNKNOWN in plan.lead_findings
    assert plan.findings == () and result.waveform is None
    assert "timing_unverified" in result.provenance_json


@pytest.mark.parametrize("failure", ["detector", "ocr"])
def test_backend_error_has_no_sensitive_context_or_output(
    ocr, monkeypatch, capsys, failure
):
    def fail(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PERSON /private/source.png")

    rule = detector(fail) if failure == "detector" else detector()
    if failure == "ocr":
        monkeypatch.setattr(StreamingTesseractEngine, "recognize", fail)
    assert_error(
        PrivacyReason.BACKEND,
        lambda: prepare_annotation_privacy(source(), policy(), rule, ocr),
    )
    assert "SYNTHETIC_PERSON" not in str(capsys.readouterr())


def test_model_outputs_are_typed_and_original_coordinate_based():
    for invalid in (
        None,
        [span(0, 100_000)],
        [object()],
        [replace(span(0, 1), start=True)],
    ):
        rule = detector(lambda text, result=invalid: result)
        assert_error(
            PrivacyReason.DECLARATION,
            lambda: prepare_annotation_privacy(source(rendered=False), policy(), rule),
        )


def test_network_provider_and_absent_ocr_are_refused():
    assert_error(PrivacyReason.OFFLINE, lambda: detector(local_only=False))

    def network(text):
        return ()

    network.allows_network = True
    assert_error(PrivacyReason.OFFLINE, lambda: detector(network))
    assert_error(
        PrivacyReason.COVERAGE,
        lambda: prepare_annotation_privacy(source(), policy(), detector()),
    )
    assert_error(
        PrivacyReason.OFFLINE,
        lambda: LocalAnnotationOcr(
            engine=object(), runtime_digest="a" * 64, qualification_digest="b" * 64
        ),
    )


def test_changed_ocr_configuration_during_execution_is_refused(monkeypatch, ocr):
    original = StreamingTesseractEngine.recognize

    def mutate(self, image, **kwargs):
        self.timeout_seconds += 1
        return original(self, image, **kwargs)

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", mutate)
    assert_error(
        PrivacyReason.STALE,
        lambda: prepare_annotation_privacy(source(), policy(), detector(), ocr),
    )


def test_source_mutation_during_detection_is_refused(ocr):
    item = source()

    def mutate(text):
        item.pages[0].image.putpixel((0, 0), (10, 20, 30))
        return detect(text)

    assert_error(
        PrivacyReason.STALE,
        lambda: prepare_annotation_privacy(item, policy(), detector(mutate), ocr),
    )


def test_limits_cover_source_and_actual_serialized_export(ocr):
    item = source()
    for limits in (
        policy(maximum_text_chars=5),
        policy(maximum_page_pixels=1),
        policy(maximum_total_pixels=1),
    ):
        assert_error(
            PrivacyReason.LIMIT,
            lambda: prepare_annotation_privacy(item, limits, detector(), ocr),
        )
    item, limits = source(rendered=False), policy(maximum_output_bytes=1)
    plan = prepare_annotation_privacy(item, limits, detector())
    assert_error(
        PrivacyReason.LIMIT,
        lambda: export_annotation_privacy(item, limits, detector(), plan, review(plan)),
    )


def test_reports_are_value_free_replay_deterministic_and_no_network(monkeypatch, ocr):
    def forbidden(*args, **kwargs):
        pytest.fail("annotation pipeline attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    item = source()
    plan, first = run(item, ocr=ocr)
    again, second = run(item, ocr=ocr)
    assert plan == again and first == second
    report = (
        plan.to_report_json()
        + first.to_report_json()
        + repr(plan)
        + repr(first)
        + repr(item)
    )
    for value in ("SYNTHETIC_PERSON", "10 mm/mV", "Synthetic regular trace"):
        assert value not in report
    assert "bounds" not in plan.to_report_json()


def test_cross_word_identifier_projects_to_each_complete_original_word(
    monkeypatch, ocr
):
    item = source(fields=())
    item.pages[0].image.paste((80, 150, 190), (35, 5, 55, 15))

    def recognize(self, image, **kwargs):
        words = []
        if image.getpixel((10, 10)) != (0, 0, 0):
            words.append(OcrWord("Casey", (5, 5, 30, 15), 0.99))
        if image.getpixel((40, 10)) != (0, 0, 0):
            words.append(OcrWord("Example", (35, 5, 55, 15), 0.99))
        return OcrResult(tuple(words))

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", recognize)

    def combined(text):
        index = text.find("Casey Example")
        return (span(index, index + 13),) if index >= 0 else ()

    plan, result = run(item, ocr=ocr, rule=detector(combined))
    assert [item.span for item in plan.pixels] == [(0, 5), (6, 13)]
    with Image.open(BytesIO(result.png_pages[0])) as image:
        assert image.getpixel((10, 10)) == image.getpixel((40, 10)) == (0, 0, 0)
        assert image.getpixel((32, 10)) == (255, 255, 255)


def test_later_page_failure_withholds_the_entire_export(monkeypatch, ocr):
    item = source()
    second = raster()
    second.putpixel((40, 50), (9, 8, 7))
    second_regions = tuple(
        replace(
            part,
            region_ref=ref(index + 10),
            slot=(1, 0) if part.kind is RegionKind.TRACE else None,
        )
        for index, part in enumerate(regions())
    )
    wave = replace(
        item.waveform,
        leads=item.waveform.leads
        + (EcgLead.from_samples("II", item.waveform.leads[0].samples_mv, unit="mV"),),
    )
    item = replace(
        item,
        waveform=wave,
        layout=LeadLayout.digital(("I", "II")),
        observations=item.observations
        + (replace(item.observations[0], label="II", row=1),),
        pages=item.pages + (AnnotationPage(second, second_regions),),
    )
    original = StreamingTesseractEngine.recognize
    verified_pages = []

    def recognize(self, image, **kwargs):
        if image.getpixel((10, 10)) == (0, 0, 0):
            verified_pages.append(image.getpixel((40, 50)))
            if image.getpixel((40, 50)) == (9, 8, 7):
                return OcrResult((OcrWord("SYNTHETIC_PERSON", (5, 5, 30, 15), 0.99),))
        return original(self, image, **kwargs)

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", recognize)
    plan = prepare_annotation_privacy(item, policy(), detector(), ocr)
    assert_error(
        PrivacyReason.RESIDUAL,
        lambda: export_annotation_privacy(
            item, policy(), detector(), plan, review(plan), ocr
        ),
    )
    assert verified_pages == [(0, 100, 0), (9, 8, 7)]


@pytest.mark.parametrize(
    "word",
    [
        OcrWord("sentinel", (1, 1, 5, 5), 0.9, page=1),
        OcrWord("sentinel", (1, 1, 200, 5), 0.9),
        OcrWord("sentinel", (1, 1, 5, 5), float("nan")),
        OcrWord("", (1, 1, 5, 5), 0.9),
    ],
)
def test_invalid_ocr_geometry_or_confidence_never_means_no_identifiers(
    monkeypatch, ocr, word
):
    monkeypatch.setattr(
        StreamingTesseractEngine, "recognize", lambda *a, **k: OcrResult((word,))
    )
    assert_error(
        PrivacyReason.DECLARATION,
        lambda: prepare_annotation_privacy(source(), policy(), detector(), ocr),
    )


@pytest.mark.parametrize(
    "value",
    [
        "MRN: 123456789",
        "MRN: １２３４５６７８９",
        "synthetic.person@example.com",
        "202-555-0146",
    ],
)
def test_critical_identifier_sweep_does_not_depend_on_model_hits(value):
    item = source(
        rendered=False, fields=(AnnotationField(SurfaceKind.HEADER, "contact", value),)
    )
    _, result = run(item, rule=detector(lambda text: ()))
    changed = json.loads(result.structured_json)["fields"][1]["value"]
    assert changed != value and "[REDACTED]" in changed


def test_streaming_adapter_uses_memory_input_and_parsed_tsv(monkeypatch):
    from pathlib import Path

    from openmed.multimodal import ocr_tesseract_stream as native

    payloads = []

    def execute(command, payload, **kwargs):
        assert command[1:3] == ["stdin", "stdout"]
        assert "tessedit_write_images=0" in command
        payloads.append(payload)
        rows = []
        with Image.open(BytesIO(payload)) as image:
            assert image.info == {}
            if image.getpixel((10, 10)) == (200, 100, 50):
                rows.append([5, 1, 1, 1, 1, 1, 5, 5, 25, 10, 99, "SYNTHETIC_PERSON"])
            rows.append([5, 1, 1, 1, 1, 2, 60, 5, 40, 10, 99, "10mm/mV"])
        return (
            "\t".join(native._TSV_HEADER)
            + "\n"
            + "\n".join("\t".join(map(str, row)) for row in rows)
            + "\n"
        ).encode()

    def no_write(*args, **kwargs):
        pytest.fail("annotation export wrote a source or output file")

    monkeypatch.setattr(native.shutil, "which", lambda executable: "/trusted/tesseract")
    monkeypatch.setattr(native, "_execute", execute)
    monkeypatch.setattr(Path, "write_bytes", no_write)
    monkeypatch.setattr(Path, "write_text", no_write)
    binding = LocalAnnotationOcr(
        engine=StreamingTesseractEngine(),
        runtime_digest="1" * 64,
        qualification_digest="2" * 64,
    )
    _, result = run(ocr=binding)
    assert len(payloads) == 3 and payloads[0] == payloads[1] != payloads[2]
    assert result.png_pages[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_header_only_detection_redacts_matching_rendered_text(ocr):
    def contextual(text):
        return detect(text) if text.startswith("patient\n") else ()

    plan, result = run(ocr=ocr, rule=detector(contextual))
    assert len(plan.pixels) == 1
    with Image.open(BytesIO(result.png_pages[0])) as image:
        assert image.getpixel((10, 10)) == (0, 0, 0)
        assert image.getpixel((65, 10)) == (50, 100, 200)


def test_image_only_detection_propagates_back_to_fields(ocr):
    def image_context(text):
        return detect(text) if "10mm/mV" in text else ()

    _, result = run(ocr=ocr, rule=detector(image_context))
    fields = json.loads(result.structured_json)["fields"]
    assert fields[1]["value"] == "[REDACTED]"
    assert "SYNTHETIC_PERSON" not in result.structured_json


def test_known_identifier_residual_refused_when_detector_misses_output(
    monkeypatch, ocr
):
    original = StreamingTesseractEngine.recognize

    def recognize(self, image, **kwargs):
        if image.getpixel((10, 10)) == (0, 0, 0):
            return OcrResult((OcrWord("synthetic_person", (60, 5, 100, 15), 0.99),))
        return original(self, image, **kwargs)

    monkeypatch.setattr(StreamingTesseractEngine, "recognize", recognize)
    rule = detector(lambda text: detect(text) if text.startswith("patient\n") else ())
    item, limits = source(), policy()
    plan = prepare_annotation_privacy(item, limits, rule, ocr)
    assert_error(
        PrivacyReason.RESIDUAL,
        lambda: export_annotation_privacy(item, limits, rule, plan, review(plan), ocr),
    )


def test_known_name_does_not_erase_parts_of_clinical_words():
    item = source(
        rendered=False,
        fields=(
            AnnotationField(SurfaceKind.HEADER, "patient", "Art"),
            AnnotationField(SurfaceKind.ANNOTATION, "note", "Artery; art; artifact"),
        ),
    )
    rule = detector(lambda text: (span(8, 11),) if text == "patient\nArt" else ())
    _, result = run(item, rule=rule)
    assert (
        json.loads(result.structured_json)["fields"][2]["value"]
        == "Artery; [REDACTED]; artifact"
    )


def test_context_spanning_detection_propagates_value_without_header_key():
    item = source(
        rendered=False,
        fields=(
            AnnotationField(SurfaceKind.HEADER, "patient", "Casey Example"),
            AnnotationField(SurfaceKind.ANNOTATION, "note", "patient Casey Example"),
        ),
    )
    rule = detector(
        lambda text: (span(0, len(text)),) if text == "patient\nCasey Example" else ()
    )
    _, result = run(item, rule=rule)
    assert (
        json.loads(result.structured_json)["fields"][2]["value"] == "patient [REDACTED]"
    )


def test_identifier_matching_canonical_lead_is_a_hold_not_a_label_rewrite():
    item = source(
        rendered=False, fields=(AnnotationField(SurfaceKind.HEADER, "patient", "I"),)
    )
    rule = detector(lambda text: (span(8, 9),) if text == "patient\nI" else ())
    plan = prepare_annotation_privacy(item, policy(), rule)
    assert PrivacyReason.LAYOUT in plan.findings
    assert_error(
        PrivacyReason.LAYOUT,
        lambda: export_annotation_privacy(item, policy(), rule, plan, review(plan)),
    )


def test_rendered_lead_label_region_cannot_be_silently_redacted(ocr):
    item = source()
    page = item.pages[0]
    item = replace(
        item,
        pages=(
            replace(
                page,
                regions=(replace(page.regions[0], surface=SurfaceKind.LEAD_LABEL),)
                + page.regions[1:],
            ),
        ),
    )
    plan = prepare_annotation_privacy(item, policy(), detector(), ocr)
    assert PrivacyReason.LAYOUT in plan.findings
    assert_error(
        PrivacyReason.LAYOUT,
        lambda: export_annotation_privacy(
            item, policy(), detector(), plan, review(plan), ocr
        ),
    )


def test_cross_surface_matching_has_bounded_rule_count():
    item = source(
        rendered=False,
        fields=tuple(
            AnnotationField(SurfaceKind.HEADER, "patient", f"Example-{i:04d}")
            for i in range(513)
        ),
    )
    rule = detector(
        lambda text: (span(8, len(text)),) if text.startswith("patient\n") else ()
    )
    assert_error(
        PrivacyReason.LIMIT, lambda: prepare_annotation_privacy(item, policy(), rule)
    )


def test_propagation_preserves_original_offsets_across_whitespace_variants():
    item = source(
        rendered=False,
        fields=(
            AnnotationField(SurfaceKind.HEADER, "patient", "Casey\nExample"),
            AnnotationField(
                SurfaceKind.ANNOTATION, "note", "Seen: casey  example; unchanged"
            ),
        ),
    )
    rule = detector(
        lambda text: (span(8, len(text)),) if text == "patient\nCasey\nExample" else ()
    )
    _, result = run(item, rule=rule)
    assert (
        json.loads(result.structured_json)["fields"][2]["value"]
        == "Seen: [REDACTED]; unchanged"
    )


@pytest.mark.parametrize(
    "labelled,bare",
    [
        ("MRN: １２３４５６７８９", "123456789"),
        ("MRN: 123456789", "١٢٣٤٥٦٧٨٩"),
        ("MRN: 123456789", "123\u200b456789"),
    ],
)
def test_confirmed_record_number_propagates_without_label_across_scripts(
    labelled, bare
):
    item = source(
        rendered=False,
        fields=(
            AnnotationField(SurfaceKind.HEADER, "record", labelled),
            AnnotationField(SurfaceKind.ANNOTATION, "note", f"before {bare} after"),
        ),
    )
    _, result = run(item, rule=detector(lambda text: ()))
    assert (
        json.loads(result.structured_json)["fields"][2]["value"]
        == "before [REDACTED] after"
    )
