# Fixed OpenPHI-Vision fixtures

`openmed.eval.openphi_vision_fixtures` loads six hand-specified synthetic cases
from the bundled `openphi-vision-v1` catalog. Loading and rendering are offline;
Pillow is required only when loading or accessing images. The catalog supplies
evaluation inputs and gold annotations for OpenPHI-Vision (#3205). It adds no
scoring, release gate, model qualification, clinical validation, configurable
generator, random seed API or training export.

These fixtures are non-diagnostic synthetic test inputs. They do not authorize
clinical decisions or replace explicit reviewer confirmation for consequential
outputs in a consuming workflow. No providers, model weights, restricted data,
real patient records or cloud fallback are included.

## Catalog coverage

| Case | Track | Fixed layout |
| --- | --- | --- |
| `vision-v1-001` | `documents` | Identifiers in header and footer, date in body |
| `vision-v1-002` | `documents` | Footer identifiers, header date |
| `vision-v1-003` | `documents` | Header name marker and aligned form fields |
| `vision-v1-004` | `slide_labels` | Name marker, printed accession and specimen codes |
| `vision-v1-005` | `mixed_script` | Latin and Cyrillic markers in separated regions |
| `vision-v1-006` | `packets` | Four interleaved pages from two synthetic subjects |

Slide codes are printed synthetic text, not barcode/QR decoding fixtures.
The catalog does not cover every script, OCR conflict, DICOM overlay, real
document format or attack required by #3205. It complements the existing
synthetic DICOM suite; general document and overlay generators remain in #2848
and #2849. This issue is scoped to the Python evaluation catalog, not a new
OpenMedKit runtime API. The in-memory PGM bytes can also serve as shared image
inputs for downstream platform tests without changing their pixel contract.

## Loader and gold contract

```python
from openmed.eval.openphi_vision_fixtures import load_openphi_vision_fixtures

cases = tuple(load_openphi_vision_fixtures())
case = cases[0]
page = case.pages[0]
image = page.to_image()  # Detached grayscale Pillow image for a local provider.
gold = page.gold_boxes  # Keep local; do not include values in evidence.
assert case.version == "openphi-vision-v1"
assert case.expected_packet_splits == ((0,),)
```

Cases, pages and gold boxes are frozen objects. Page text, gold text and PGM
bytes are omitted from every nested `repr`; explicit `.text` access remains
available for local span checks. Identifiers and names in the source catalog are
obvious synthetic markers. Case/page IDs are opaque controlled codes.

Each gold box has a canonical `PERSON`, `ID_NUM` or `DATE` label, a `critical`
flag, a region code, half-open `(x0, y0, x1, y1)` ink bounds and `[start, end)`
character offsets into the page's newline-joined text. Every page and track has
critical direct identifiers. Dates are deliberately noncritical controls;
consumer gates must assess critical leakage independently of aggregate scores.

`expected_packet_splits` is a tuple of groups of zero-based page indices. The
packet's gold is `((0, 2), (1, 3))`: grouping consecutive pages would mix subjects.
Each group retains its original page order and all pages appear exactly once.
This is gold membership, not an implementation of packet splitting.

## Consumption by #3205

The suite can iterate cases, send **only** `page.to_image()` to an offline OCR or
vision provider, and compare its returned local annotations against the gold.
It must retain `page_id`, coordinates and offsets, evaluate each critical box,
and compare packet membership without crossing the gold patient boundaries.
The suite owns label matching, OCR conflict handling, leakage scoring, thresholds,
review requirements and release evidence; no proposed suite API is assumed here.

Safe diagnostics can include `case_id`, `page_id`, `version`, `track`, counts,
box coordinates, offsets and SHA-256 digests. Do not serialize dataclasses with
`asdict`, log explicit `.text`, or emit images/source declarations into audit
reports: hiding values in `repr` is not a general serialization sanitizer.

Example metadata-only summary:

```python
summary = {
    "case_id": case.case_id,
    "version": case.version,
    "sha256": case.sha256,
    "page_count": len(case.pages),
    "critical_count": sum(box.critical for p in case.pages for box in p.gold_boxes),
}
assert summary["case_id"] == "vision-v1-001"
assert summary["page_count"] == 1
assert summary["critical_count"] == 3
```

## Reproducibility and drift

The catalog, font, case declarations and every page have pinned SHA-256 digests.
Case digests bind the complete declaration, including gold values, boxes, flags,
layout and packet splits. Page digests cover binary PGM: the exact ASCII header
`P5\nWIDTH HEIGHT\n255\n` followed by grayscale bytes in row order. There is no
PNG compression, timestamp, EXIF, private path or model output in these bytes.

Rendering reuses the existing burned-in font-fitting and raster-measurement
helpers with the bundled IBM Plex Mono Regular face (OFL-1.1, copied unchanged
from the repository's font assets with its license). Pillow's BASIC layout
engine avoids optional shaping and system-font fallbacks. The loader rejects
drift, including differing rasterizer output, with value-free errors; it never
silently repins or qualifies a different environment. A changed corpus requires
a reviewed version change and independently measured pins.

Focused checks:

```bash
.venv/bin/python -m pytest tests/unit/eval/test_openphi_vision_fixtures.py \
  tests/integration/test_openphi_vision_fixtures.py -q
```

Tests independently bound rendered ink and spans, regenerate all pages, check
critical flags and both scripts, verify packet membership and hidden values,
and reject corrupted digests/boxes. Model-free integration controls mask every
gold box, then deliberately omit a critical identifier to demonstrate residual
ink. They exercise fixture consumption and do not claim provider recall or
clinical safety.
