# Image geometry derivation

`derive_image_geometry` turns validated image dimensions into the derived
numbers preflight code repeatedly needs: a checked pixel count, a reduced
aspect ratio, a portrait or landscape class, and an optional allocation-sized
memory estimate. It is an explicit helper, not a decoder or a pipeline gate.

```python
from openmed.multimodal.image_geometry import derive_image_geometry

geometry = derive_image_geometry(1920, 1080, bytes_per_pixel=4)
print(geometry.pixel_count, geometry.aspect_ratio_width, geometry.estimated_bytes)
```

## Supported boundary

The inputs are dimension values that a bounded header preflight has already
produced, or any values a caller wants validated fail-closed. Each dimension
must be a plain positive `int` of at most `MAX_IMAGE_DIMENSION` (2^31 - 1,
the manifest count bound). `None`, booleans, non-finite floats, other types,
and zero or negative values are rejected with stable, value-free categories.
No image is read, decoded, resized, or allocated, and no EXIF orientation is
applied; a declared dimension stays the number the caller supplies.

`bytes_per_pixel` is optional. When supplied it must be a positive integer of
at most 1024, and `estimated_bytes` is the overflow-checked product
`pixel_count * bytes_per_pixel`. The result dataclass carries dimensions and
derived numbers only, and serializes through `to_dict()`/`to_json()` in a
fixed field order with schema version 1.

## Limits and checked arithmetic

The default pixel ceiling is `max_pixels=100_000_000` (inclusive: equal
passes, above fails). The pixel product is checked by division before it is
materialized, so a caller can never receive an allocation-relevant number
that exceeds the declared ceiling. `max_pixels` itself must be a positive
integer below 2^63. The byte estimate is checked against the same 63-bit
bound before it is produced. Lower limits can be supplied per call.

## Failures

`ImageGeometryError` is a `ValueError` with a stable `.category`; its string
is the same category, and no input value is ever echoed. Categories are
`image_geometry_width_missing`, `image_geometry_height_missing`,
`image_geometry_width_boolean`, `image_geometry_height_boolean`,
`image_geometry_width_not_finite`, `image_geometry_height_not_finite`,
`image_geometry_width_not_integer`, `image_geometry_height_not_integer`,
`image_geometry_width_not_positive`, `image_geometry_height_not_positive`,
`image_geometry_width_overflow`, `image_geometry_height_overflow`,
`image_geometry_pixel_limit_exceeded`, and
`image_geometry_byte_estimate_overflow`. Invalid `max_pixels` or
`bytes_per_pixel` API arguments are reported separately as plain `ValueError`
with constant messages. Unsupported inputs should be handled explicitly by
the caller rather than retried with larger limits.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_image_geometry.py -q
```

Tests use synthetic dimensions and hand calculations only. They cover square,
portrait, landscape, reducible, one-pixel, and maximum cases, inclusive pixel
and byte boundaries, and every rejection category, offline.
