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
    PageRotation,
    PageSize,
    PageTransform,
    SourceSpan,
    transform_bbox,
    transform_document,
    transform_ocr_result,
    transform_point,
)


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
    transform = PageTransform(PageSize(100, 200), rotation)
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
