# OCR box-coordinate normalization

OCR adapters do not share one coordinate convention. Some return pixels from
the top-left of an image, PDF-oriented adapters may return points from the
bottom-left of a page, and layout models commonly return fractions of the
page. OpenMed normalizes these inputs to one deterministic page-space
contract:

```text
(x0, y0, x1, y1), top-left origin, each coordinate in [0, 1]
```

The normalizer is local-only and does not download models or make network
calls. It validates input rather than clipping it, so a malformed source box
cannot silently change span-to-region provenance.

## Normalize one box

Pixel and point inputs require the page dimensions in the same unit as the
box. A point is a PDF point (or another caller-defined point unit); no DPI
conversion is guessed.

```python
from openmed.multimodal import normalize_box

normalized = normalize_box(
    (120, 80, 360, 180),
    unit="pixel",
    page_size=(1200, 800),
    page=0,
    source_ref="page-0-word-4",
)

assert normalized.bbox == (0.1, 0.1, 0.3, 0.225)
assert normalized.source_ref == "page-0-word-4"
```

For a bottom-left source, the y-axis is flipped into the top-left output
contract:

```python
normalized = normalize_box(
    (72, 576, 216, 720),
    unit="point",
    page_size=(612, 792),
    origin="bottom-left",
    source_ref="page-0-word-5",
)
```

Normalized inputs already use a unit page:

```python
normalized = normalize_box(
    {"bbox": (0.2, 0.25, 0.8, 0.75), "unit": "normalized"},
)
```

The result is immutable and includes the canonical `bbox`, page index, source
unit, source origin, and the optional opaque `source_ref`. `to_dict()` returns
a JSON-ready provenance record with those fields. `normalize_boxes()` applies
the same contract to an ordered iterable and preserves each mapping's source
reference and order.

## Rejected input

The normalizer raises `AmbiguousBoxError` or `BoxValidationError` for:

- missing or conflicting units, coordinate representations, page dimensions,
  or source options;
- inverted or zero-area rectangles;
- `NaN`, infinite, non-numeric, or out-of-bounds coordinates;
- polygon-like sequences when an axis-aligned rectangle was expected.

Four-value sequences are always interpreted as `xyxy`. Use an explicit
mapping with `x`, `y`, `width`, and `height` when the source uses `xywh`.
There is no automatic clipping or format guessing.

## Privacy boundary

`source_ref` is for an opaque stable reference such as a page and word index;
it is not a place for OCR text, a patient identifier, or a file payload. The
normalizer does not accept OCR text, does not include input values in
exceptions, and emits only geometry and provenance metadata. Tests and
examples use synthetic references only.
