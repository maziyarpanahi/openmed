from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.multimodal.orientation_preflight import (
    MAX_ORIENTATION_VALUES,
    ORIENTATION_PREFLIGHT_SCHEMA_VERSION,
    ORIENTATION_REASON_CODES,
    ImageTransform,
    OrientationPreflightError,
    OrientationStatus,
    check_image_orientation,
)

Grid = list[list[str]]

# EXIF 2.32 Orientation: which visual side the stored 0th row and 0th column
# represent.
_EXIF_ROW_COLUMN_SIDES = {
    1: ("top", "left"),
    2: ("top", "right"),
    3: ("bottom", "right"),
    4: ("bottom", "left"),
    5: ("left", "top"),
    6: ("right", "top"),
    7: ("right", "bottom"),
    8: ("left", "bottom"),
}


def _upright() -> Grid:
    return [["a", "b", "c"], ["d", "e", "f"]]


def _stored_for(orientation: int, upright: Grid) -> Grid:
    """Build stored pixels whose EXIF orientation displays ``upright``."""
    height, width = len(upright), len(upright[0])
    row_side, column_side = _EXIF_ROW_COLUMN_SIDES[orientation]
    if row_side in ("top", "bottom"):
        rows = range(height) if row_side == "top" else range(height - 1, -1, -1)
        columns = range(width) if column_side == "left" else range(width - 1, -1, -1)
        return [[upright[row][column] for column in columns] for row in rows]
    # Stored rows run along visual columns.
    stored_rows = range(width) if row_side == "left" else range(width - 1, -1, -1)
    stored_columns = (
        range(height) if column_side == "top" else range(height - 1, -1, -1)
    )
    return [[upright[row][column] for row in stored_columns] for column in stored_rows]


def _apply(transform: ImageTransform, grid: Grid) -> Grid:
    result = [list(reversed(row)) for row in grid] if transform.mirrored else grid
    for _ in range(transform.rotation_degrees // 90):
        result = [list(row) for row in zip(*result[::-1])]  # 90 degrees clockwise
    return result


@pytest.mark.parametrize("orientation", range(1, 9))
def test_each_exif_transform_displays_stored_pixels_upright(orientation: int) -> None:
    upright = _upright()
    stored = _stored_for(orientation, upright)
    transform = ImageTransform.from_exif_orientation(orientation)

    assert _apply(transform, stored) == upright
    assert transform.exif_orientation == orientation
    assert transform.swaps_dimensions is (orientation >= 5)
    assert transform.is_identity is (orientation == 1)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_matching_declarations_are_aligned_for_all_eight_orientations(
    orientation: int,
) -> None:
    transform = ImageTransform.from_exif_orientation(orientation)

    report = check_image_orientation(
        [orientation],
        declared_transform=transform,
        stored_size=(40, 30),
        declared_size=(30, 40) if orientation >= 5 else (40, 30),
    )

    assert report.status is OrientationStatus.ALIGNED
    assert report.reason_codes == ()
    assert report.orientation == orientation
    assert report.required_transform == transform
    assert (report.oriented_width, report.oriented_height) == (
        (30, 40) if orientation >= 5 else (40, 30)
    )


@pytest.mark.parametrize("orientation", range(2, 9))
def test_undeclared_transforms_are_required(orientation: int) -> None:
    report = check_image_orientation([orientation])

    assert report.status is OrientationStatus.TRANSFORM_REQUIRED
    assert report.reason_codes == ("transform_required",)
    assert report.required_transform == ImageTransform.from_exif_orientation(
        orientation
    )
    assert report.declared_transform is None


def test_upright_orientation_without_declaration_is_aligned() -> None:
    report = check_image_orientation((1,))

    assert report.status is OrientationStatus.ALIGNED
    assert report.reason_codes == ()
    assert report.required_transform == ImageTransform()


def test_identity_declaration_over_rotated_metadata_was_not_applied() -> None:
    report = check_image_orientation([6], declared_transform=ImageTransform())

    assert report.status is OrientationStatus.TRANSFORM_REQUIRED
    assert report.reason_codes == ("transform_not_applied",)


@pytest.mark.parametrize(
    ("orientation", "declared"),
    [
        (6, ImageTransform(rotation_degrees=270)),
        (1, ImageTransform(rotation_degrees=90)),
        (3, ImageTransform(rotation_degrees=180, mirrored=True)),
    ],
)
def test_contradicting_declarations_are_ambiguous(
    orientation: int, declared: ImageTransform
) -> None:
    report = check_image_orientation([orientation], declared_transform=declared)

    assert report.status is OrientationStatus.AMBIGUOUS
    assert report.reason_codes == ("transform_conflict",)
    assert report.orientation == orientation


def test_missing_orientation_without_declaration_is_aligned() -> None:
    report = check_image_orientation([], stored_size=(640, 480))

    assert report.status is OrientationStatus.ALIGNED
    assert report.reason_codes == ("orientation_missing",)
    assert report.orientation is None
    assert report.required_transform is None
    assert (report.oriented_width, report.oriented_height) == (640, 480)


def test_missing_orientation_with_identity_declaration_is_aligned() -> None:
    report = check_image_orientation([], declared_transform=ImageTransform())

    assert report.status is OrientationStatus.ALIGNED
    assert report.reason_codes == ("orientation_missing",)


def test_missing_orientation_with_rotation_declaration_is_ambiguous() -> None:
    report = check_image_orientation(
        [], declared_transform=ImageTransform(rotation_degrees=90)
    )

    assert report.status is OrientationStatus.AMBIGUOUS
    assert report.reason_codes == (
        "transform_without_orientation",
        "orientation_missing",
    )


def test_duplicate_equal_values_are_recorded_but_aligned() -> None:
    report = check_image_orientation(
        [8, 8], declared_transform=ImageTransform(rotation_degrees=270)
    )

    assert report.status is OrientationStatus.ALIGNED
    assert report.reason_codes == ("orientation_duplicate",)
    assert report.orientation == 8


def test_duplicate_equal_values_still_require_undeclared_transform() -> None:
    report = check_image_orientation([3, 3])

    assert report.status is OrientationStatus.TRANSFORM_REQUIRED
    assert report.reason_codes == ("transform_required", "orientation_duplicate")


def test_conflicting_values_are_ambiguous_without_an_orientation() -> None:
    report = check_image_orientation(
        [1, 6], declared_transform=ImageTransform(rotation_degrees=90)
    )

    assert report.status is OrientationStatus.AMBIGUOUS
    assert report.reason_codes == ("orientation_conflict",)
    assert report.orientation is None
    assert report.required_transform is None


@pytest.mark.parametrize("values", [[0], [9], [-1], [1, 65535], [2**31]])
def test_out_of_range_values_are_invalid(values: list[int]) -> None:
    report = check_image_orientation(values)

    assert report.status is OrientationStatus.INVALID
    assert report.reason_codes == ("orientation_value_invalid",)
    assert report.orientation is None


def test_value_count_limit_is_invalid() -> None:
    at_limit = check_image_orientation([1] * MAX_ORIENTATION_VALUES)
    over_limit = check_image_orientation([1] * (MAX_ORIENTATION_VALUES + 1))

    assert at_limit.status is OrientationStatus.ALIGNED
    assert over_limit.status is OrientationStatus.INVALID
    assert over_limit.reason_codes == ("orientation_value_limit",)


def test_declared_size_that_ignores_rotation_is_inconsistent() -> None:
    report = check_image_orientation(
        [6],
        declared_transform=ImageTransform(rotation_degrees=90),
        stored_size=(4000, 3000),
        declared_size=(4000, 3000),
    )

    assert report.status is OrientationStatus.AMBIGUOUS
    assert report.reason_codes == ("dimensions_inconsistent",)
    assert (report.oriented_width, report.oriented_height) == (3000, 4000)


def test_required_transform_is_used_for_dimensions_when_undeclared() -> None:
    report = check_image_orientation(
        [8], stored_size=(4000, 3000), declared_size=(3000, 4000)
    )

    assert report.reason_codes == ("transform_required",)
    assert (report.oriented_width, report.oriented_height) == (3000, 4000)


@pytest.mark.parametrize(
    "sizes",
    [
        {"stored_size": (0, 10)},
        {"stored_size": (10, 1_000_001)},
        {"stored_size": (10, 10), "declared_size": (-1, 10)},
    ],
)
def test_non_positive_or_oversized_dimensions_are_invalid(
    sizes: dict[str, Any],
) -> None:
    report = check_image_orientation([1], **sizes)

    assert report.status is OrientationStatus.INVALID
    assert "dimensions_invalid" in report.reason_codes


def test_most_severe_status_wins_and_reasons_follow_declared_order() -> None:
    report = check_image_orientation(
        [6, 6],
        declared_transform=ImageTransform(rotation_degrees=270),
        stored_size=(0, 10),
    )

    assert report.status is OrientationStatus.INVALID
    assert report.reason_codes == (
        "dimensions_invalid",
        "transform_conflict",
        "orientation_duplicate",
    )
    positions = [ORIENTATION_REASON_CODES.index(code) for code in report.reason_codes]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    ("kwargs", "category"),
    [
        ({"orientation_values": 6}, "orientation_values_invalid"),
        ({"orientation_values": "6"}, "orientation_values_invalid"),
        ({"orientation_values": [True]}, "orientation_values_invalid"),
        ({"orientation_values": ["Synthetic Patient"]}, "orientation_values_invalid"),
        ({"orientation_values": [6.0]}, "orientation_values_invalid"),
        (
            {"orientation_values": [6], "declared_transform": {"rotation": 90}},
            "declared_transform_invalid",
        ),
        (
            {"orientation_values": [6], "stored_size": [10, 10]},
            "stored_size_invalid",
        ),
        (
            {"orientation_values": [6], "declared_size": (10, "10")},
            "declared_size_invalid",
        ),
    ],
)
def test_argument_type_errors_are_value_free(
    kwargs: dict[str, Any], category: str
) -> None:
    with pytest.raises(OrientationPreflightError) as exc_info:
        check_image_orientation(**kwargs)

    assert exc_info.value.category == category
    assert str(exc_info.value) == category
    assert "Synthetic Patient" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("kwargs", "category"),
    [
        ({"rotation_degrees": 45}, "transform_rotation_invalid"),
        ({"rotation_degrees": 360}, "transform_rotation_invalid"),
        ({"rotation_degrees": True}, "transform_rotation_invalid"),
        ({"mirrored": 1}, "transform_mirror_invalid"),
    ],
)
def test_transform_validation(kwargs: dict[str, Any], category: str) -> None:
    with pytest.raises(OrientationPreflightError) as exc_info:
        ImageTransform(**kwargs)

    assert exc_info.value.category == category


@pytest.mark.parametrize("value", [0, 9, True, "6"])
def test_from_exif_orientation_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(OrientationPreflightError, match="orientation_value_invalid"):
        ImageTransform.from_exif_orientation(value)


def test_report_serialization_is_deterministic() -> None:
    report = check_image_orientation(
        [5, 5],
        declared_transform=ImageTransform(rotation_degrees=270, mirrored=True),
        stored_size=(200, 100),
        declared_size=(100, 200),
    )

    assert list(report.to_dict()) == [
        "schema_version",
        "status",
        "reason_codes",
        "orientation",
        "required_transform",
        "declared_transform",
        "oriented_width",
        "oriented_height",
    ]
    assert report.to_json() == (
        '{"declared_transform":{"mirrored":true,"rotation_degrees":270},'
        '"orientation":5,"oriented_height":200,"oriented_width":100,'
        '"reason_codes":["orientation_duplicate"],'
        '"required_transform":{"mirrored":true,"rotation_degrees":270},'
        f'"schema_version":"{ORIENTATION_PREFLIGHT_SCHEMA_VERSION}",'
        '"status":"aligned"}'
    )
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.is_aligned is True


def test_reports_contain_only_categorical_and_numeric_values() -> None:
    report = check_image_orientation(
        [2, 7],
        declared_transform=ImageTransform(rotation_degrees=90),
        stored_size=(10, 20),
    )

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            assert (
                value in ORIENTATION_REASON_CODES
                or value in {status.value for status in OrientationStatus}
                or value == ORIENTATION_PREFLIGHT_SCHEMA_VERSION
            )
        else:
            assert value is None or type(value) in (int, bool)

    walk(report.to_dict())
