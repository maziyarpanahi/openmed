"""Synthetic affine/crop coverage, rational rounding and source-provenance controls."""

import json
import socket
from dataclasses import replace
from fractions import Fraction

import pytest

from openmed.multimodal.evidence import (
    EvidenceNode,
    NodeKind,
    PageGeometry,
    SpatialUnit,
)
from openmed.multimodal.page_rotation import PageRotation, PageSize, PageTransform
from openmed.multimodal.privacy import (
    Affine,
    CoordinateSpace,
    MaskFailure,
    MaskLineageError,
    Rectangle,
    TransformStep,
    propagate_masks,
)


def space(index=0, width=100, height=80, *, unit=SpatialUnit.PIXEL, page=0):
    return CoordinateSpace(
        space_ref=f"{index:032x}",
        source_digest="a" * 64,
        geometry=PageGeometry(page=page, width=width, height=height, unit=unit),
    )


def mask(source, region=(20, 20, 40, 30), *, index=1, kind=NodeKind.PIXELS):
    return EvidenceNode(
        node_id=f"{index:032x}",
        kind=kind,
        source_digest=source.source_digest,
        geometry=source.geometry,
        region=region,
        span=(0, 4) if kind is NodeKind.OCR else None,
    )


def stage(source, target, affine=Affine(), coverage=None, margin=0):
    return TransformStep(
        source=source,
        target=target,
        affine=affine,
        resampling_margin=margin,
        coverage=coverage,
    )


@pytest.mark.parametrize("rotation", list(PageRotation))
def test_quarter_turns_match_existing_page_rotation_contract(rotation):
    source = space()
    old = PageTransform(PageSize(100, 80), rotation)
    target = space(1, old.target_size.width, old.target_size.height)
    node = mask(source)
    result = propagate_masks(
        source, (node,), (TransformStep.rotate(source, target, rotation),)
    )
    assert result.accepted and result.masks[0].raster_box == old.bbox(node.region)
    assert node.region == (20, 20, 40, 30)


def test_crop_rotation_and_scale_compose_and_invert_source_coordinates():
    source, cropped, rotated, scaled = (
        space(),
        space(1, 80, 60),
        space(2, 60, 80),
        space(3, 120, 240),
    )
    node = mask(source, (30, 20, 40, 30), kind=NodeKind.OCR)
    steps = (
        TransformStep.crop(source, cropped, Rectangle(10, 10, 90, 70)),
        TransformStep.rotate(cropped, rotated, PageRotation.ROTATE_90),
        TransformStep.scale(rotated, scaled, resampling_margin=0),
    )
    result = propagate_masks(source, (node,), steps)
    assert result.accepted and result.masks[0].raster_box == (80, 60, 100, 90)
    for point in Rectangle(*node.region).polygon:
        assert result.composed.inverse().point(result.composed.point(point)) == point
    report = json.loads(result.to_json())
    assert report["sources"][0]["span"] == [0, 4]
    assert result.source_nodes[0] is node and node.geometry == source.geometry


def test_shear_and_inverse_do_not_repeatedly_inflate_bounding_boxes():
    source, sheared, restored = (
        space(0, 100, 100),
        space(1, 200, 100),
        space(2, 100, 100),
    )
    node = mask(source, (20, 20, 30, 30))
    forward = Affine(b=1)
    result = propagate_masks(
        source,
        (node,),
        (
            replace(stage(source, sheared, forward), rasterize=False),
            stage(sheared, restored, forward.inverse()),
        ),
    )
    assert result.accepted and result.composed == Affine()
    assert result.masks[0].polygon == Rectangle(*node.region).polygon
    assert result.masks[0].raster_box == (20, 20, 30, 30)


def test_fractional_scaling_rounds_outwards_without_losing_a_pixel():
    source, target = space(0, 9, 9), space(1, 6, 6)
    result = propagate_masks(
        source,
        (mask(source, (1, 1, 4, 4)),),
        (TransformStep.scale(source, target, resampling_margin=0),),
    )
    assert result.accepted and result.masks[0].raster_box == (0, 0, 3, 3)
    assert result.masks[0].polygon == Rectangle(0, 0, 3, 3).polygon
    geometric = propagate_masks(
        source,
        (mask(source, (1, 1, 4, 4)),),
        (
            replace(
                TransformStep.scale(source, target, resampling_margin=0),
                rasterize=False,
            ),
        ),
    )
    assert geometric.masks[0].polygon[0] == (Fraction(2, 3), Fraction(2, 3))
    assert geometric.masks[0].polygon[2] == (Fraction(8, 3), Fraction(8, 3))
    assert geometric.masks[0].raster_box is None


def test_pdf_points_render_to_pixels_with_explicit_unit_transition():
    source, target = space(0, 72, 72, unit=SpatialUnit.POINT), space(1, 300, 300)
    result = propagate_masks(
        source,
        (mask(source, (12, 12, 24, 24)),),
        (TransformStep.scale(source, target, resampling_margin=0),),
    )
    assert result.accepted and result.masks[0].raster_box == (50, 50, 100, 100)
    assert result.source.geometry.unit is SpatialUnit.POINT
    assert result.target.geometry.unit is SpatialUnit.PIXEL


@pytest.mark.parametrize("region", [(5, 20, 15, 30), (0, 0, 5, 5), (85, 20, 95, 30)])
def test_partial_or_complete_crop_loss_rejects_every_mask(region):
    source, target = space(), space(1, 80, 60)
    first, second = mask(source), mask(source, region, index=2)
    result = propagate_masks(
        source,
        (first, second),
        (TransformStep.crop(source, target, Rectangle(10, 10, 90, 70)),),
    )
    assert not result.accepted and result.failure is MaskFailure.COVERAGE_LOSS
    assert result.masks == () and result.composed is None and result.failed_step == 0


def test_coverage_union_holes_are_not_hidden_by_the_outer_extent():
    source, target = space(0, 10, 10), space(1, 10, 10)
    coverage = (
        Rectangle(0, 0, 4, 10),
        Rectangle(6, 0, 10, 10),
        Rectangle(4, 0, 6, 4),
        Rectangle(4, 6, 6, 10),
    )
    result = propagate_masks(
        source,
        (mask(source, (3, 3, 7, 7)),),
        (stage(source, target, coverage=coverage),),
    )
    assert result.failure is MaskFailure.COVERAGE_LOSS and result.masks == ()


def test_overlapping_coverage_is_a_union_not_double_counted_area():
    source, target = space(0, 10, 10), space(1, 10, 10)
    node = mask(source, (0, 0, 10, 10))
    half = Rectangle(0, 0, 5, 10)
    assert (
        propagate_masks(
            source, (node,), (stage(source, target, coverage=(half, half)),)
        ).failure
        is MaskFailure.COVERAGE_LOSS
    )
    complete = (Rectangle(0, 0, 7, 10), Rectangle(3, 0, 10, 10))
    assert propagate_masks(
        source, (node,), (stage(source, target, coverage=complete),)
    ).accepted


def test_zero_coverage_and_singular_transform_fail_closed():
    source, target = space(), space(1)
    assert (
        propagate_masks(
            source, (mask(source),), (stage(source, target, coverage=()),)
        ).failure
        is MaskFailure.COVERAGE_LOSS
    )
    singular = Affine(1, 2, 0, 2, 4, 0)
    result = propagate_masks(
        source, (mask(source),), (stage(source, target, singular),)
    )
    assert result.failure is MaskFailure.SINGULAR and result.masks == ()
    with pytest.raises(MaskLineageError):
        singular.inverse()


def test_small_nonsingular_transform_has_nonempty_outward_raster_mask():
    source, target = space(), space(1)
    affine = Affine(a=Fraction(1, 10**12), e=Fraction(1, 10**12))
    result = propagate_masks(source, (mask(source),), (stage(source, target, affine),))
    assert result.accepted and result.masks[0].raster_box == (0, 0, 1, 1)


def test_resampling_margin_propagates_and_cannot_be_lost_in_a_later_crop():
    source, scaled, cropped = space(0, 40, 40), space(1, 80, 80), space(2, 22, 22)
    node = mask(source, (10, 10, 20, 20))
    scale = TransformStep.scale(source, scaled, resampling_margin=2)
    first = propagate_masks(source, (node,), (scale,))
    assert first.masks[0].raster_box == (18, 18, 42, 42)
    result = propagate_masks(
        source,
        (node,),
        (scale, TransformStep.crop(scaled, cropped, Rectangle(19, 19, 41, 41))),
    )
    assert result.failure is MaskFailure.COVERAGE_LOSS and result.failed_step == 1


@pytest.mark.parametrize(
    "changed", ["source_digest", "geometry", "space", "page", "reuse"]
)
def test_source_or_coordinate_identity_mismatch_never_produces_masks(changed):
    source, target = space(), space(1)
    node, previous = mask(source), source
    expected = MaskFailure.SPACE_MISMATCH
    if changed == "source_digest":
        node = replace(node, source_digest="b" * 64)
        expected = MaskFailure.SOURCE_MISMATCH
    elif changed == "geometry":
        node = replace(node, geometry=space(0, 101, 80).geometry)
        expected = MaskFailure.SOURCE_MISMATCH
    elif changed == "space":
        previous = space(9)
    elif changed == "page":
        target = space(1, page=1)
    else:
        target = source
    result = propagate_masks(source, (node,), (stage(previous, target),))
    assert result.failure is expected and result.masks == ()


def test_arithmetic_and_coverage_work_are_bounded_with_no_partial_output():
    source, second, third = space(), space(1), space(2)
    tiny = Affine(a=Fraction(1, 2**300), e=1)
    result = propagate_masks(
        source,
        (mask(source),),
        (stage(source, second, tiny), stage(second, third, tiny)),
    )
    assert result.failure is MaskFailure.ARITHMETIC_LIMIT and result.masks == ()
    result = propagate_masks(
        source,
        (mask(source), mask(source, index=2)),
        (stage(source, second),),
        max_clip_tests=1,
    )
    assert result.failure is MaskFailure.RESOURCE_LIMIT and result.masks == ()


def test_complex_coverage_union_exceeds_cell_budget_before_mask_intersections():
    source, target = space(0, 1000, 1000), space(1, 1000, 1000)
    coverage = tuple(Rectangle(i * 3, 0, i * 3 + 1, 1000) for i in range(32)) + tuple(
        Rectangle(0, i * 3, 1000, i * 3 + 1) for i in range(32)
    )
    result = propagate_masks(
        source, (mask(source),), (stage(source, target, coverage=coverage),)
    )
    assert result.failure is MaskFailure.RESOURCE_LIMIT


def test_point_only_lineage_never_claims_a_raster_mask():
    source = space(unit=SpatialUnit.POINT)
    result = propagate_masks(source, (mask(source),), ())
    assert result.accepted and result.masks[0].raster_box is None


def test_materialized_intermediate_pixels_stay_protected_after_inverse_transform():
    source, sheared, restored = (
        space(0, 100, 100),
        space(1, 200, 100),
        space(2, 100, 100),
    )
    node = mask(source, (20, 20, 30, 30))
    forward = Affine(b=1)
    result = propagate_masks(
        source,
        (node,),
        (
            stage(source, sheared, forward),
            stage(sheared, restored, forward.inverse()),
        ),
    )
    assert result.accepted and result.composed == Affine()
    assert result.masks[0].raster_box == (10, 20, 40, 30)


def test_reflection_preserves_coverage_and_inverts_exactly():
    source, target = space(), space(1)
    affine = Affine(a=-1, c=100)
    result = propagate_masks(source, (mask(source),), (stage(source, target, affine),))
    assert result.accepted and result.masks[0].raster_box == (60, 20, 80, 30)
    assert result.composed.inverse().point((60, 20)) == (40, 20)


def test_fractional_pixel_crop_is_not_misdeclared_as_lossless():
    with pytest.raises(MaskLineageError):
        TransformStep.crop(space(), space(1, 10, 10), Rectangle(0.5, 0.5, 10.5, 10.5))


@pytest.mark.parametrize(
    "bad", [True, float("nan"), float("inf"), "SYNTHETIC-PRIVATE", 2**600]
)
def test_invalid_numbers_raise_fixed_value_free_errors(bad):
    with pytest.raises(MaskLineageError, match="^invalid redaction mask lineage$"):
        Affine(a=bad)


def test_deterministic_report_preserves_source_coordinates_without_network(
    monkeypatch, caplog
):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network access")

    monkeypatch.setattr(socket, "socket", deny)
    source, target = space(), space(1, 200, 160)
    node = mask(source, kind=NodeKind.OCR)
    step = TransformStep.scale(source, target, resampling_margin=0)
    reports = [propagate_masks(source, (node,), (step,)).to_json() for _ in range(2)]
    assert reports[0] == reports[1] and caplog.text == ""
    assert json.loads(reports[0])["sources"][0]["region"]["left"] == [20, 1]
    assert node.region == (20, 20, 40, 30)
