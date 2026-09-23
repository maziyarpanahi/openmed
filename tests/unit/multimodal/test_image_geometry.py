"""Synthetic unit tests for allocation-safe image geometry derivation."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.image_geometry import (
    IMAGE_GEOMETRY_SCHEMA_VERSION,
    MAX_IMAGE_DIMENSION,
    ImageGeometryError,
    Orientation,
    derive_image_geometry,
)

_UINT32_MAX = (1 << 31) - 1


@pytest.mark.parametrize(
    "width,height,pixels,ratio,orientation",
    [
        (4, 4, 16, (1, 1), Orientation.SQUARE),
        (3, 4, 12, (3, 4), Orientation.PORTRAIT),
        (4, 3, 12, (4, 3), Orientation.LANDSCAPE),
        (1, 1, 1, (1, 1), Orientation.SQUARE),
        (1920, 1080, 2_073_600, (16, 9), Orientation.LANDSCAPE),
        (1080, 1920, 2_073_600, (9, 16), Orientation.PORTRAIT),
        (7, 5, 35, (7, 5), Orientation.LANDSCAPE),
    ],
)
def test_hand_calculations(width, height, pixels, ratio, orientation) -> None:
    result = derive_image_geometry(width, height)
    assert result.width == width
    assert result.height == height
    assert result.pixel_count == pixels
    assert (result.aspect_ratio_width, result.aspect_ratio_height) == ratio
    assert result.orientation is orientation
    assert result.bytes_per_pixel is None
    assert result.estimated_bytes is None


def test_largest_representable_dimensions_pass_hand_calculation() -> None:
    result = derive_image_geometry(
        _UINT32_MAX,
        _UINT32_MAX,
        bytes_per_pixel=1,
        max_pixels=_UINT32_MAX * _UINT32_MAX,
    )
    assert result.pixel_count == _UINT32_MAX * _UINT32_MAX
    assert result.estimated_bytes == _UINT32_MAX * _UINT32_MAX
    assert (result.aspect_ratio_width, result.aspect_ratio_height) == (1, 1)
    assert result.orientation is Orientation.SQUARE


def test_memory_estimate_matches_hand_calculation() -> None:
    result = derive_image_geometry(640, 480, bytes_per_pixel=3)
    assert result.pixel_count == 307_200
    assert result.estimated_bytes == 921_600
    assert result.bytes_per_pixel == 3


@pytest.mark.parametrize(
    "value,category_name",
    [
        (None, "missing"),
        (True, "boolean"),
        (False, "boolean"),
        (float("nan"), "not_finite"),
        (float("inf"), "not_finite"),
        (float("-inf"), "not_finite"),
        (4.0, "not_integer"),
        ("10", "not_integer"),
        ([], "not_integer"),
        (0, "not_positive"),
        (-5, "not_positive"),
        (_UINT32_MAX + 1, "overflow"),
    ],
)
def test_invalid_widths_fail_closed(value, category_name) -> None:
    with pytest.raises(
        ImageGeometryError, match=f"^image_geometry_width_{category_name}$"
    ) as raised:
        derive_image_geometry(value, 4)
    assert raised.value.category == f"image_geometry_width_{category_name}"
    assert str(raised.value) == f"image_geometry_width_{category_name}"


@pytest.mark.parametrize(
    "value,category_name",
    [
        (None, "missing"),
        (True, "boolean"),
        (float("nan"), "not_finite"),
        (0, "not_positive"),
        (_UINT32_MAX + 1, "overflow"),
    ],
)
def test_invalid_heights_fail_closed(value, category_name) -> None:
    with pytest.raises(
        ImageGeometryError, match=f"^image_geometry_height_{category_name}$"
    ):
        derive_image_geometry(4, value)


def test_pixel_limit_is_inclusive_and_checked_by_division() -> None:
    assert derive_image_geometry(100, 10, max_pixels=1000).pixel_count == 1000
    with pytest.raises(
        ImageGeometryError, match="^image_geometry_pixel_limit_exceeded$"
    ):
        derive_image_geometry(101, 10, max_pixels=1000)
    # A width larger than the whole limit can never materialize the product.
    with pytest.raises(
        ImageGeometryError, match="^image_geometry_pixel_limit_exceeded$"
    ):
        derive_image_geometry(101, 1, max_pixels=100)


def test_byte_estimate_is_overflow_checked() -> None:
    max_square = _UINT32_MAX * _UINT32_MAX
    # (2**63 - 1) // max_square == 2, so two bytes per pixel still fits.
    assert (
        derive_image_geometry(
            _UINT32_MAX,
            _UINT32_MAX,
            bytes_per_pixel=2,
            max_pixels=max_square,
        ).estimated_bytes
        == 2 * max_square
    )
    with pytest.raises(
        ImageGeometryError, match="^image_geometry_byte_estimate_overflow$"
    ):
        derive_image_geometry(
            _UINT32_MAX,
            _UINT32_MAX,
            bytes_per_pixel=3,
            max_pixels=max_square,
        )
    assert (
        derive_image_geometry(
            _UINT32_MAX,
            _UINT32_MAX,
            bytes_per_pixel=1,
            max_pixels=max_square,
        ).estimated_bytes
        == max_square
    )


@pytest.mark.parametrize(
    "bytes_per_pixel",
    [0, -1, True, 2.5, "3", [], 1025],
)
def test_invalid_bytes_per_pixel_fails_with_constant_message(
    bytes_per_pixel,
) -> None:
    with pytest.raises(ValueError, match="^bytes_per_pixel"):
        derive_image_geometry(4, 4, bytes_per_pixel=bytes_per_pixel)


@pytest.mark.parametrize("max_pixels", [None, True, 0, -1, 2.5, "10", []])
def test_invalid_max_pixels_fails_with_constant_message(max_pixels) -> None:
    with pytest.raises(ValueError, match="^max_pixels"):
        derive_image_geometry(4, 4, max_pixels=max_pixels)


def test_max_pixels_has_a_hard_range() -> None:
    with pytest.raises(ValueError, match="^max_pixels is out of range$"):
        derive_image_geometry(4, 4, max_pixels=(1 << 63))
    assert derive_image_geometry(4, 4, max_pixels=(1 << 63) - 1).pixel_count == 16


def test_dimension_bound_is_inclusive() -> None:
    result = derive_image_geometry(_UINT32_MAX, 1, max_pixels=(1 << 63) - 1)
    assert result.width == MAX_IMAGE_DIMENSION
    assert result.pixel_count == _UINT32_MAX


def test_error_is_a_value_error_without_values() -> None:
    error = ImageGeometryError("image_geometry_width_missing")
    assert isinstance(error, ValueError)
    assert error.category == "image_geometry_width_missing"
    assert str(error) == "image_geometry_width_missing"


def test_serialization_is_deterministic_and_minimal() -> None:
    result = derive_image_geometry(1920, 1080, bytes_per_pixel=4)
    assert result.schema_version == IMAGE_GEOMETRY_SCHEMA_VERSION
    data = result.to_dict()
    assert list(data) == [
        "schema_version",
        "width",
        "height",
        "pixel_count",
        "aspect_ratio_width",
        "aspect_ratio_height",
        "orientation",
        "bytes_per_pixel",
        "estimated_bytes",
    ]
    expected = (
        '{"schema_version":1,"width":1920,"height":1080,'
        '"pixel_count":2073600,"aspect_ratio_width":16,'
        '"aspect_ratio_height":9,"orientation":"landscape",'
        '"bytes_per_pixel":4,"estimated_bytes":8294400}'
    )
    assert result.to_json() == expected
    assert json.loads(result.to_json()) == data


def test_optional_keys_are_absent_without_an_estimate() -> None:
    data = derive_image_geometry(4, 4).to_dict()
    assert "bytes_per_pixel" not in data
    assert "estimated_bytes" not in data


def test_reordered_inputs_produce_identical_output() -> None:
    first = derive_image_geometry(1920, 1080, bytes_per_pixel=4)
    second = derive_image_geometry(1920, 1080, bytes_per_pixel=4)
    assert first.to_json() == second.to_json()
