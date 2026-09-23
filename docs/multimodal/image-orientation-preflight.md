# Image Orientation Preflight

`openmed.multimodal.orientation_preflight.check_image_orientation()` compares
the numeric EXIF/TIFF `Orientation` values of a stored image with the
transform an image manifest declares. A disagreement between the two silently
moves region coordinates to the wrong place after a rotation or mirror, so the
check runs before any pixels are decoded.

The check reads no pixels and no metadata strings. It accepts integers, an
optional declared transform, and optional dimensions, and returns a report
with categorical and numeric fields only.

## Transforms

`ImageTransform(rotation_degrees, mirrored)` describes how to display stored
pixels upright: mirror horizontally first when `mirrored` is true, then rotate
clockwise by 0, 90, 180, or 270 degrees. These eight combinations match the
eight EXIF orientations one to one.

| EXIF orientation | Rotation | Mirrored | Width and height swap |
| ---: | ---: | :---: | :---: |
| 1 | 0 | no | no |
| 2 | 0 | yes | no |
| 3 | 180 | no | no |
| 4 | 180 | yes | no |
| 5 | 270 | yes | yes |
| 6 | 90 | no | yes |
| 7 | 90 | yes | yes |
| 8 | 270 | no | yes |

`ImageTransform.from_exif_orientation(6)` returns the transform for a value,
and `transform.exif_orientation` maps back.

## Example

```python
from openmed.multimodal.orientation_preflight import (
    ImageTransform,
    OrientationStatus,
    check_image_orientation,
)

report = check_image_orientation(
    [6],
    declared_transform=ImageTransform(rotation_degrees=90),
    stored_size=(4000, 3000),
    declared_size=(3000, 4000),
)
assert report.status is OrientationStatus.ALIGNED

report = check_image_orientation([6])
assert report.status is OrientationStatus.TRANSFORM_REQUIRED
assert report.reason_codes == ("transform_required",)
```

## Statuses and reason codes

The status is the most severe verdict implied by the reason codes:
`aligned` < `transform_required` < `ambiguous` < `invalid`. Reason codes are
reported in this fixed order:

| Reason code | Status | Meaning |
| --- | --- | --- |
| `orientation_value_limit` | invalid | More than 8 orientation values were supplied. |
| `orientation_value_invalid` | invalid | A value is outside EXIF orientations 1 to 8. |
| `dimensions_invalid` | invalid | A width or height is not between 1 and 1,000,000. |
| `orientation_conflict` | ambiguous | Metadata sources report different orientations. |
| `transform_conflict` | ambiguous | The declared transform contradicts the orientation. |
| `transform_without_orientation` | ambiguous | A rotation or mirror was declared, but no orientation is recorded. |
| `dimensions_inconsistent` | ambiguous | Declared upright dimensions do not match the effective transform. |
| `transform_not_applied` | transform_required | The manifest declares no change, but the orientation needs one. |
| `transform_required` | transform_required | Nothing was declared and the orientation needs a transform. |
| `orientation_duplicate` | aligned | Several sources agree on the same orientation. |
| `orientation_missing` | aligned | No orientation is recorded, so the image is treated as upright. |

When dimensions are supplied, the declared transform (or, if none, the
transform the orientation requires) determines the expected upright
`oriented_width` and `oriented_height` in the report.

Wrong argument types, such as strings, booleans, floats, or lists where a
`(width, height)` tuple is expected, raise `OrientationPreflightError`. Its
`category` is a stable code and the message never repeats the submitted value.

## Serialization

`report.to_dict()` keeps a fixed field order, and `report.to_json()` returns
compact JSON with sorted keys and the schema identifier
`openmed.multimodal.orientation_preflight.v1`.

## Out of scope

The preflight does not rotate or mirror pixels, run OCR, detect landmarks, or
interpret images clinically.
