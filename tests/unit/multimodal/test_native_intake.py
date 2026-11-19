"""Reference native-intake mapping and the vectors shared with Swift and Kotlin."""

import json
import struct
from pathlib import Path

import pytest

from openmed.multimodal.apple_contract import MultimodalPacket
from openmed.multimodal.evidence.fused_graph import SpatialUnit
from openmed.multimodal.native_intake import (
    BOX_TOLERANCE,
    IntakeLimits,
    IntakeOutcome,
    IntakePageGeometry,
    IntakePageInput,
    IntakeReason,
    NativeIntakeError,
    PageStatus,
    Recognition,
    RecognizedRegion,
    intake_node_ref,
    map_document,
    map_recognized_box,
    refuse_document,
)
from openmed.multimodal.page_rotation import PageTransform
from tests.fixtures.multimodal import document_intake

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/parity/document_intake_v1.json"
)
DIGEST = "d" * 64


def corpus():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def cases():
    return {case["id"]: case for case in corpus()["cases"]}


def f64(value):
    return struct.unpack(">d", bytes.fromhex(value))[0]


def rendered(text, regions=(), *, ocr=(100, 100), **geometry):
    return IntakePageInput(
        geometry=IntakePageGeometry(**{"width": 100, "height": 100, **geometry}),
        recognition=Recognition(
            width=ocr[0],
            height=ocr[1],
            text=text,
            regions=tuple(
                RecognizedRegion(span=span, box=box) for span, box in regions
            ),
        ),
    )


def test_committed_vectors_match_the_reference_mapper():
    assert FIXTURE.read_text(encoding="utf-8") == document_intake.render()
    data = corpus()
    assert data["synthetic_only"] is True
    assert data["outcomes"] == [outcome.value for outcome in IntakeOutcome]
    assert data["reasons"] == [reason.value for reason in IntakeReason]
    assert data["page_statuses"] == [status.value for status in PageStatus]
    assert data["box_tolerance"] == BOX_TOLERANCE
    covered = {cover for case in data["cases"] for cover in case["covers"]}
    assert {"multipage", "rotated", "cropped", "mixed-script"} <= covered
    assert {"partial", "refused"} <= covered


@pytest.mark.parametrize("case", corpus()["cases"], ids=lambda case: case["id"])
def test_mapped_regions_recover_the_authored_original_boxes(case):
    report = document_intake.run(case)
    truths = [
        region["truth"]
        for row, record in zip(case["pages"], report.pages)
        if record.status is PageStatus.MAPPED
        for region in row["ocr"]["regions"]
    ]
    assert len(truths) == len(report.regions)
    for truth, region, row in zip(truths, report.regions, case["expected"]["regions"]):
        assert region.box == tuple(f64(value) for value in row["box_f64"])
        assert max(abs(a - b) for a, b in zip(region.box, truth)) <= case["tolerance"]
        assert report.text[region.span[0] : region.span[1]]
        payload = MultimodalPacket(region.evidence.wire).payload()
        assert payload["span"] == list(region.span)
        assert payload["kind"] == "ocr_span"


def test_spans_map_back_to_regions_across_pages_and_scripts():
    rows = cases()
    report = document_intake.run(rows["multipage_pdf"])
    crossing = report.regions_for_span(30, 41)
    assert [region.page for region in crossing] == [0, 1]
    assert report.text[30:41] == "01-02\n\nPlan"
    assert report.regions_for_span(35, 37) == ()

    mixed = document_intake.run(rows["mixed_script_image"])
    emoji = mixed.regions_for_span(15, 16)
    assert [mixed.text[r.span[0] : r.span[1]] for r in emoji] == ["😀"]
    assert emoji[0].box == (200.0, 0.0, 220.0, 40.0)
    accented = mixed.regions_for_span(17, 22)[0]
    assert mixed.text[accented.span[0] : accented.span[1]] == "José"
    # Unicode scalars, not UTF-16 code units or grapheme clusters.
    assert len(mixed.text) == 22
    assert len(mixed.text.encode("utf-16-le")) // 2 == 23

    rotated = document_intake.run(rows["cropped_rotated_image"])
    page = rotated.pages[0]
    assert page.crop_evidence is not None
    assert page.crop_evidence.payload()["box_f64"] is not None
    region = rotated.regions[0].evidence.payload()
    assert region["rotation"] == 90
    assert [f64(value) for value in region["display_box_f64"]] == list(
        PageTransform((1000, 800), 90).bbox((150, 100, 230, 130))
    )


def test_partial_and_refused_outcomes_are_explicit():
    rows = cases()
    missing = document_intake.run(rows["missing_and_failed_pages"])
    assert missing.outcome is IntakeOutcome.PARTIAL
    assert [page.status for page in missing.pages] == [
        PageStatus.MAPPED,
        PageStatus.HELD,
        PageStatus.HELD,
        PageStatus.MAPPED,
    ]
    lost = document_intake.run(rows["permission_lost"])
    assert len(lost.pages) == 2 and lost.text == "Before"
    assert (
        document_intake.run(rows["unrotated_render"]).outcome is IntakeOutcome.REFUSED
    )
    assert document_intake.run(rows["empty_document"]).reasons == (
        IntakeReason.EMPTY_DOCUMENT,
    )
    for refusal in ("permission_denied", "document_locked"):
        report = document_intake.run(rows[refusal])
        assert report.outcome is IntakeOutcome.REFUSED and not report.pages
    with pytest.raises(NativeIntakeError):
        refuse_document(IntakeReason.PAGE_MISSING)


def test_page_limit_placeholder_matches_a_truncating_native_source():
    stop = IntakePageInput(unavailable=IntakeReason.PAGE_LIMIT)
    pages = [rendered("A"), rendered("B"), stop]
    report = map_document(DIGEST, pages, limits=IntakeLimits(max_pages=2))
    assert report.outcome is IntakeOutcome.PARTIAL
    assert report.reasons == (IntakeReason.PAGE_LIMIT,)
    assert report.text == "A\n\nB"
    early = map_document(DIGEST, [rendered("A"), stop])
    assert early.reasons == (IntakeReason.PAGE_LIMIT,) and len(early.pages) == 1


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_each_quarter_turn_round_trips_through_a_crop(rotation):
    crop = (20.0, 10.0, 180.0, 90.0)
    geometry = IntakePageGeometry(
        width=200, height=100, unit=SpatialUnit.POINT, rotation=rotation, crop=crop
    )
    truth = (40.0, 50.0, 80.0, 80.0)
    local = PageTransform((160, 80), rotation)
    upright = local.bbox((20.0, 40.0, 60.0, 70.0))
    size = (int(local.target_size.width * 2), int(local.target_size.height * 2))
    box = tuple(value * 2 for value in upright)
    assert map_recognized_box(geometry, *size, box) == truth


def test_ocr_box_overshoot_is_clamped_only_within_tolerance():
    geometry = IntakePageGeometry(width=100, height=100)
    assert map_recognized_box(geometry, 100, 100, (-1, 10, 101, 20)) == (
        0.0,
        10.0,
        100.0,
        20.0,
    )
    for box in [(-2, 10, 50, 20), (10, 10, 102, 20), (10, 10, 10, 20)]:
        with pytest.raises(NativeIntakeError) as error:
            map_recognized_box(geometry, 100, 100, box)
        assert error.value.reason is IntakeReason.REGION_OUT_OF_BOUNDS
        assert str(error.value) == "region_out_of_bounds"


def test_render_rounding_is_tolerated_but_swapped_dimensions_are_not():
    geometry = IntakePageGeometry(width=612, height=791.7, rotation=0)
    # Each dimension rounded to the nearest pixel at 2x.
    assert map_recognized_box(geometry, 1224, 1583, (0, 0, 1224, 1583))[2] == 612.0
    for width, height in [(1584, 1224), (1224, 1500)]:
        with pytest.raises(NativeIntakeError) as error:
            map_recognized_box(geometry, width, height, (0, 0, 10, 10))
        assert error.value.reason is IntakeReason.AMBIGUOUS_TRANSFORM


@pytest.mark.parametrize(
    "geometry",
    [
        {"rotation": 45},
        {"rotation": -90},
        {"crop": (0, 0, 0, 100)},
        {"crop": (-1, 0, 50, 100)},
        {"crop": (0, 0, 50, float("inf"))},
        {"width": 0},
        {"width": float("nan")},
        {"width": 10**400},
    ],
)
def test_ambiguous_geometry_holds_the_page_without_echoing_values(geometry):
    report = map_document(
        DIGEST, [rendered("private-synthetic", **geometry), rendered("kept")]
    )
    assert report.outcome is IntakeOutcome.PARTIAL
    assert report.pages[0].reason is IntakeReason.AMBIGUOUS_TRANSFORM
    assert report.text == "kept"
    assert "private-synthetic" not in repr(report)
    assert "private-synthetic" not in json.dumps(report.to_dict())


def test_region_limits_and_invalid_spans_hold_the_page():
    limits = IntakeLimits(max_page_scalars=3)
    assert map_document(DIGEST, [rendered("four")], limits=limits).reasons == (
        IntakeReason.RESOURCE_LIMIT,
    )
    for span in [(2, 2), (3, 2), (0, 9)]:
        report = map_document(DIGEST, [rendered("text", [(span, (1, 1, 5, 5))])])
        assert report.reasons == (IntakeReason.INVALID_RECOGNITION,)


def test_every_region_indexes_the_text_and_records_stay_value_free():
    report = document_intake.run(cases()["multipage_pdf"])
    summary = json.dumps(report.to_dict())
    assert "Synthetic" not in summary and "Jane" not in summary
    assert "Synthetic" not in repr(report)
    assert "Jane" not in repr(rendered("Jane"))
    ordinals: dict[int, int] = {}
    source = document_intake.digest("multipage_pdf")
    for region in report.regions:
        ordinal = ordinals.setdefault(region.page, 0)
        ordinals[region.page] += 1
        label = f"region:{ordinal}"
        assert region.node_ref == intake_node_ref(source, region.page, label)
    assert report.pages[1].node_ref == intake_node_ref(source, 1, "page")
    with pytest.raises(NativeIntakeError):
        report.regions_for_span(0, len(report.text) + 1)
    with pytest.raises(NativeIntakeError):
        report.regions_for_span(3, 2)


@pytest.mark.parametrize(
    "build",
    [
        lambda: IntakeLimits(max_pages=0),
        lambda: IntakeLimits(max_pages=True),
        lambda: IntakePageGeometry(width="1", height=1),
        lambda: IntakePageGeometry(width=1, height=1, crop=[0, 0, 1, 1]),
        lambda: RecognizedRegion(span=(0, 1.0), box=(0, 0, 1, 1)),
        lambda: Recognition(width=1, height=1, text=b"x"),
        lambda: IntakePageInput(),
        lambda: IntakePageInput(unavailable=IntakeReason.PERMISSION_DENIED),
        lambda: IntakePageInput(unavailable="page_missing"),
        lambda: map_document("not-a-digest", [rendered("x")]),
        lambda: intake_node_ref(DIGEST, -1, "page"),
    ],
)
def test_invalid_inputs_raise_a_value_free_error(build):
    with pytest.raises(NativeIntakeError) as error:
        build()
    assert str(error.value) == "invalid native document intake input"
