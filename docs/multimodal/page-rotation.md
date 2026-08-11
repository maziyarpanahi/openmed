# OCR page-rotation transforms

Scanned pages can be rotated before OCR runs. OpenMed exposes an explicit,
local-only transform for carrying page-space points, OCR boxes, and source-span
references into the rotated page coordinate system.

Coordinates use a top-left origin. A page of width `W` and height `H` has the
closed outer bounds `0 <= x <= W` and `0 <= y <= H`; OCR boxes must have
positive area and remain within those bounds. Rotation values are clockwise
quarter turns only: `0`, `90`, `180`, or `270`. A 90° or 270° turn swaps the
output width and height.

```python
from openmed.multimodal import PageTransform, transform_bbox, transform_point

transform = PageTransform(page_size=(1000, 800), rotation=90)
point = transform.point((120, 80))
box = transform.bbox((120, 80, 360, 180))

assert point == (720, 120)
assert box == (620, 120, 720, 360)
assert transform.inverse().bbox(box) == (120, 80, 360, 180)

# The functional form is equivalent:
assert transform_point((120, 80), (1000, 800), 90) == point
assert transform_bbox((120, 80, 360, 180), (1000, 800), 90) == box
```

`transform_ocr_result()` returns a new `OcrResult` with the same word text,
confidence, page indexes, metadata, and input order; only each word's bounding
box changes. Call `to_layout()` on the transformed result when its new
geometry should drive a fresh deterministic reading-order reconstruction.
`transform_document()` similarly preserves text offsets, pages, and span
metadata while transforming source bounding boxes.

Malformed dimensions, missing or conflicting rotation, unsupported orientation,
inverted boxes, non-finite coordinates, and out-of-bounds geometry raise
validation errors. Inputs are rejected rather than clipped so a transform
cannot silently change source provenance. Error messages contain stable field
names and reason codes, not coordinates, OCR text, or source references.
