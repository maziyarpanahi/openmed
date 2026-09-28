"""Focused synthetic tests for lossless OCR page-space rotation."""

from __future__ import annotations

import math

import pytest

from openmed.multimodal import (
    AmbiguousOrientationError,
    ExtractedDocument,
    GeometryValidationError,
    InvalidPageDimensionsError,
    OcrResult,
    OcrWord,
    OutOfBoundsError,
    PageDimensions,
    PageRotation,
    PageSize,
    PageTransform,
    SourceSpan,
    transform_bbox,
    transform_document,
    transform_ocr_result,
    transform_point,
)
from openmed.multimodal.box_normalization import PageSize as NormalizationPageSize


def test_public_page_size_keeps_normalization_contract():
    assert PageSize is NormalizationPageSize
    assert PageDimensions is not PageSize


@pytest.mark.parametrize(
    ("rotation", "point", "expected", "target_size"),
    [
        (0, (10.0, 20.0), (10.0, 20.0), (100.0, 200.0)),
        (90, (10.0, 20.0), (180.0, 10.0), (200.0, 100.0)),
        (180, (10.0, 20.0), (90.0, 180.0), (100.0, 200.0)),
        (270, (10.0, 20.0), (20.0, 90.0), (200.0, 100.0)),
    ],
)
def test_transform_point_supports_each_clockwise_quarter_turn(
    rotation, point, expected, target_size
):
    transform = PageTransform((100, 200), rotation)

    assert transform.point(point) == expected
    assert transform.target_size.as_tuple() == target_size


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, (10.0, 20.0, 40.0, 80.0)),
        (90, (120.0, 10.0, 180.0, 40.0)),
        (180, (60.0, 120.0, 90.0, 180.0)),
        (270, (20.0, 60.0, 80.0, 90.0)),
    ],
)
def test_transform_bbox_preserves_axis_aligned_geometry(rotation, expected):
    assert transform_bbox((10, 20, 40, 80), (100, 200), rotation) == expected


@pytest.mark.parametrize("rotation", list(PageRotation))
def test_each_transform_round_trips_points_and_boxes(rotation):
    transform = PageTransform(PageDimensions(100, 200), rotation)
    point = (11.5, 22.25)
    box = (11.5, 22.25, 44.75, 88.5)

    assert transform.inverse().point(transform.point(point)) == point
    assert transform.inverse().bbox(transform.bbox(box)) == box


def test_ocr_result_preserves_order_metadata_and_source_fields():
    result = OcrResult(
        words=(
            OcrWord("synthetic-a", (10, 20, 40, 80), 0.91, page=3),
            OcrWord("synthetic-b", (50, 100, 90, 150), 0.87, page=3),
        ),
        metadata={"source_ref": "synthetic-page-3"},
    )

    transformed = transform_ocr_result(result, (100, 200), 90)

    assert [word.text for word in transformed.words] == [
        "synthetic-a",
        "synthetic-b",
    ]
    assert [word.page for word in transformed.words] == [3, 3]
    assert [word.confidence for word in transformed.words] == [0.91, 0.87]
    assert [word.bbox for word in transformed.words] == [
        (120.0, 10.0, 180.0, 40.0),
        (50.0, 50.0, 100.0, 90.0),
    ]
    assert transformed.metadata == result.metadata
    assert result.words[0].bbox == (10, 20, 40, 80)


def test_document_transform_preserves_offsets_pages_and_provenance():
    document = ExtractedDocument(
        text="synthetic text",
        spans=(
            SourceSpan(
                start=0,
                end=8,
                page=2,
                bbox=(10, 20, 40, 80),
                metadata={"source_ref": "synthetic-span-1"},
            ),
            SourceSpan(start=8, end=13, page=2, metadata={"kind": "gap"}),
        ),
        metadata={"source_ref": "synthetic-document"},
    )

    transformed = transform_document(document, width=100, height=200, rotation=90)

    assert transformed.text == document.text
    assert transformed.metadata == document.metadata
    assert transformed.spans[0].start == 0
    assert transformed.spans[0].end == 8
    assert transformed.spans[0].page == 2
    assert transformed.spans[0].bbox == (120.0, 10.0, 180.0, 40.0)
    assert transformed.spans[0].metadata == document.spans[0].metadata
    assert transformed.spans[1] is document.spans[1]


@pytest.mark.parametrize(
    "rotation",
    [None, 45, -90, "portrait", True, math.nan],
)
def test_rejects_ambiguous_orientation(rotation):
    with pytest.raises(AmbiguousOrientationError, match="rotation"):
        transform_point((10, 20), (100, 200), rotation)


def test_rejects_conflicting_orientation_aliases():
    with pytest.raises(AmbiguousOrientationError, match="both"):
        transform_point((10, 20), (100, 200), 90, orientation=270)


@pytest.mark.parametrize(
    "page_size",
    [
        (0, 200),
        (100, -1),
        (math.nan, 200),
        (100, math.inf),
        (100,),
        {"width": 100},
    ],
)
def test_rejects_invalid_page_dimensions(page_size):
    with pytest.raises(InvalidPageDimensionsError, match="page"):
        transform_point((10, 20), page_size, 0)


@pytest.mark.parametrize(
    "point",
    [(-1, 20), (10, 201), (math.nan, 20), (10, math.inf)],
)
def test_rejects_out_of_bounds_or_non_finite_points(point):
    with pytest.raises((OutOfBoundsError, GeometryValidationError)):
        transform_point(point, (100, 200), 0)


@pytest.mark.parametrize(
    "box",
    [
        (-1, 20, 40, 80),
        (10, 20, 101, 80),
        (40, 20, 10, 80),
        (10, 20, 40, 20),
    ],
)
def test_rejects_invalid_or_out_of_bounds_boxes(box):
    with pytest.raises(GeometryValidationError):
        transform_bbox(box, (100, 200), 0)


def test_mapping_coordinates_are_supported_without_guessing_orientation():
    assert transform_point({"x": 10, "y": 20}, (100, 200), 90) == (180.0, 10.0)
    assert transform_bbox(
        {"left": 10, "top": 20, "right": 40, "bottom": 80},
        (100, 200),
        90,
    ) == (120.0, 10.0, 180.0, 40.0)


@pytest.mark.parametrize(
    "value",
    [
        {"bbox": (1, 2, 3, 4), "x0": 9},
        {"left": 1, "top": 2, "right": 3, "bottom": 4, "x0": 9},
        {"bbox": {"1": 2, "2": 3, "3": 4, "4": 5}},
    ],
)
def test_partial_or_nested_coordinate_representations_rejected(value):
    with pytest.raises(GeometryValidationError):
        transform_bbox(value, (100, 200), 0)


def test_conversion_failure_has_no_raw_context():
    with pytest.raises(GeometryValidationError) as caught:
        transform_point(("synthetic-sensitive-value", 2), (100, 200), 0)
    assert caught.value.__context__ is None


def test_iterator_failure_has_no_raw_context():
    from openmed.multimodal import transform_ocr_words

    def broken():
        raise RuntimeError("synthetic-sensitive-value")
        yield

    with pytest.raises(GeometryValidationError) as caught:
        transform_ocr_words(broken(), (100, 200), 0)
    assert caught.value.__context__ is None
    assert "synthetic-sensitive-value" not in str(caught.value)


def test_word_collection_is_bounded():
    from openmed.multimodal import transform_ocr_words

    word = OcrWord("synthetic", (1, 2, 3, 4), 0.9)
    with pytest.raises(GeometryValidationError):
        transform_ocr_words([word] * 4097, (100, 200), 0)


def test_coordinate_iterator_stops_at_required_count():
    def points():
        yield 1
        yield 2
        yield 3
        raise AssertionError("must not exhaust unbounded input")

    with pytest.raises(GeometryValidationError) as caught:
        transform_point(points(), (100, 200), 0)
    assert caught.value.__context__ is None


def test_mutated_typed_dimensions_are_revalidated():
    dimensions = PageDimensions(100, 200)
    object.__setattr__(dimensions, "width", -1)
    with pytest.raises(InvalidPageDimensionsError):
        PageTransform(dimensions, 0)


def test_mutated_transform_rotation_is_revalidated():
    transformer = PageTransform((100, 200), 0)
    object.__setattr__(transformer, "rotation", 45)
    with pytest.raises(AmbiguousOrientationError):
        transformer.point((1, 2))
