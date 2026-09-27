"""Focused synthetic tests for OCR box-coordinate normalization."""

from __future__ import annotations

import math

import pytest

from openmed.multimodal import (
    AmbiguousBoxError,
    BoxValidationError,
    CoordinateOrigin,
    CoordinateUnit,
    normalize_box,
    normalize_boxes,
    normalize_coordinates,
)


def test_pixel_box_is_normalized_and_keeps_opaque_source_reference():
    result = normalize_box(
        (100, 50, 300, 150),
        unit="pixel",
        page_size=(1000, 500),
        source_ref="page-0-word-1",
    )

    assert result.bbox == (0.1, 0.1, 0.3, 0.3)
    assert result.coordinates == result.page_bbox == result.bbox
    assert result.source_unit is CoordinateUnit.PIXEL
    assert result.source_origin is CoordinateOrigin.TOP_LEFT
    assert result.source_ref == "page-0-word-1"
    assert result.to_dict()["source_ref"] == "page-0-word-1"
    assert "page-0-word-1" not in repr(result)


def test_point_box_converts_bottom_left_origin_to_top_left_page_space():
    result = normalize_box(
        (60, 600, 180, 720),
        unit="point",
        page_size={"width": 600, "height": 800},
        origin="bottom-left",
        page=2,
        source_ref="page-2-word-3",
    )

    assert result.bbox == (0.1, 0.1, 0.3, 0.25)
    assert result.page == 2
    assert result.source_unit is CoordinateUnit.POINT
    assert result.source_origin is CoordinateOrigin.BOTTOM_LEFT


def test_normalized_box_stays_in_unit_page_space():
    result = normalize_box(
        {
            "bbox": (0.2, 0.25, 0.8, 0.75),
            "unit": "normalized",
            "origin": "top-left",
            "source_ref": "page-0-word-4",
        }
    )

    assert result.bbox == (0.2, 0.25, 0.8, 0.75)
    assert normalize_box(result) is result
    assert normalize_coordinates(result) == result.bbox


def test_normalized_bottom_left_box_flips_y_without_page_rescaling():
    result = normalize_box(
        (0.1, 0.2, 0.3, 0.4),
        unit="normalized",
        origin="bottom-left",
        page_size=(600, 800),
    )

    assert result.bbox == (0.1, 0.6, 0.3, 0.8)


def test_mapping_xywh_is_explicit_and_batch_order_is_stable():
    boxes = normalize_boxes(
        (
            {
                "x": 100,
                "y": 50,
                "width": 200,
                "height": 100,
                "unit": "px",
                "page_width": 1000,
                "page_height": 500,
                "source_ref": "page-0-word-1",
            },
            {
                "bbox": (0.4, 0.2, 0.6, 0.4),
                "unit": "normalized",
                "source_ref": "page-0-word-2",
            },
        )
    )

    assert [box.bbox for box in boxes] == [
        (0.1, 0.1, 0.3, 0.3),
        (0.4, 0.2, 0.6, 0.4),
    ]
    assert [box.source_ref for box in boxes] == [
        "page-0-word-1",
        "page-0-word-2",
    ]


@pytest.mark.parametrize(
    ("box", "kwargs", "message"),
    [
        ((10, 20, 10, 40), {"unit": "pixel", "page_size": (100, 100)}, "inverted"),
        ((10, math.nan, 30, 40), {"unit": "pixel", "page_size": (100, 100)}, "NaN"),
        ((-1, 20, 30, 40), {"unit": "pixel", "page_size": (100, 100)}, "out-of-bounds"),
        ((10, 20, 30, 40), {"unit": "pixel"}, "page_size"),
    ],
)
def test_rejects_invalid_boxes_without_echoing_input(box, kwargs, message):
    with pytest.raises(
        (BoxValidationError, AmbiguousBoxError), match=message
    ) as excinfo:
        normalize_box(box, source_ref="opaque-source-1", **kwargs)

    assert "opaque-source-1" not in str(excinfo.value)


@pytest.mark.parametrize(
    "box",
    [
        (0.1, 0.2, 0.3, 0.4),
        {"bbox": (0.1, 0.2, 0.3, 0.4)},
        {"bbox": (0.1, 0.2, 0.3, 0.4), "box": (0.1, 0.2, 0.3, 0.4)},
        {
            "bbox": (0.1, 0.2, 0.3, 0.4),
            "x0": 0.1,
            "y0": 0.2,
            "x1": 0.3,
            "y1": 0.4,
            "unit": "normalized",
        },
    ],
)
def test_rejects_ambiguous_coordinate_interpretations(box):
    with pytest.raises(AmbiguousBoxError, match="ambiguous|required"):
        normalize_box(box)


def test_rejects_conflicting_embedded_and_explicit_options():
    with pytest.raises(AmbiguousBoxError, match="conflicting"):
        normalize_box(
            {
                "bbox": (0.1, 0.2, 0.3, 0.4),
                "unit": "normalized",
                "source_ref": "page-0-word-1",
            },
            unit="pixel",
            page_size=(100, 100),
            source_ref="page-0-word-2",
        )


def test_rejects_polygon_like_sequences_instead_of_guessing():
    with pytest.raises(AmbiguousBoxError, match="ambiguous"):
        normalize_box(
            ((0, 0), (1, 0), (1, 1), (0, 1)),
            unit="normalized",
        )


def test_result_serialization_is_coordinate_only_and_deterministic():
    first = normalize_box(
        (20, 10, 80, 40),
        unit="pixel",
        page_width=100,
        page_height=100,
        source_ref="page-0-word-8",
    )
    second = normalize_box(
        (20, 10, 80, 40),
        unit="pixel",
        page_width=100,
        page_height=100,
        source_ref="page-0-word-8",
    )

    assert first == second
    assert first.to_dict() == {
        "bbox": [0.2, 0.1, 0.8, 0.4],
        "coordinate_system": "normalized-page",
        "page": 0,
        "source_origin": "top-left",
        "source_ref": "page-0-word-8",
        "source_unit": "pixel",
    }
