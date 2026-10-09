# PDF Page Geometry Preflight

`openmed.multimodal.pdf_geometry.read_pdf_geometry()` reads a PDF's version,
page count, and each page's `MediaBox`, `CropBox`, and rotation before any OCR
or rasterization starts. Batch planners and memory estimators can then size
work from numbers alone.

The reader needs no optional dependencies. It never decodes content streams,
text, images, fonts, outlines, or document metadata, and its report contains
only numbers and stable reason codes. No text, metadata strings, filenames, or
paths are returned.

## Example

```python
from openmed.multimodal.pdf_geometry import PdfGeometryStatus, read_pdf_geometry

with open("synthetic.pdf", "rb") as handle:
    report = read_pdf_geometry(handle, max_pages=500)

if report.status is PdfGeometryStatus.READABLE:
    for page in report.pages:
        print(page.page_index, page.width, page.height, page.rotation)
```

`source` may be `bytes`, `bytearray`, `memoryview`, or a readable binary
stream. Seekable streams are restored to their starting position and are never
closed.

## What is read

- **Version:** the `%PDF-M.m` header in the first 1,024 bytes, raised by a
  later catalog `/Version` when one is present.
- **Page tree:** leaves of the catalog's `/Pages` tree in document order,
  with `MediaBox`, `CropBox`, and `Rotate` inherited from ancestor nodes.
- **Boxes:** normalized to `(x0, y0, x1, y1)` with the lower-left corner first.
  The effective crop box is clipped to the media box and defaults to it.
- **Rotation:** normalized to 0, 90, 180, or 270 degrees clockwise. `width`
  and `height` describe the crop box as displayed, so they swap for 90 and 270.

The reader scans indirect objects, uses the latest definition of an object for
incremental updates, and expands uncompressed or FlateDecode object streams.

## Statuses and reason codes

| Status | Meaning |
| --- | --- |
| `readable` | Every page has valid geometry. |
| `review` | Pages were read, but at least one finding needs attention. |
| `rejected` | The document was not read and `pages` is empty. |

| Reason code | Status | Meaning |
| --- | --- | --- |
| `pdf_size_limit` | rejected | The input is larger than `max_bytes`. |
| `pdf_header_missing` | rejected | No `%PDF-` header in the first 1,024 bytes. |
| `pdf_encrypted` | rejected | A trailer declares `/Encrypt`. |
| `pdf_object_limit` | rejected | More than `max_objects` indirect objects. |
| `pdf_decompression_limit` | rejected | Object streams exceed `max_decompressed_bytes`. |
| `pdf_object_stream_unsupported` | rejected | Needed objects sit in an object stream that uses another filter or is corrupt. |
| `pdf_catalog_missing` | rejected | No resolvable document catalog. |
| `pdf_page_tree_invalid` | rejected | Missing, cyclic, too deep, or malformed page tree. |
| `pdf_page_limit` | rejected | More than `max_pages` pages. |
| `page_count_mismatch` | review | The root `/Count` differs from the pages found. |
| `media_box_missing` | review | A page has no media box; its geometry is `None`. |
| `media_box_invalid` | review | A media box is not four finite numbers with a non-zero area. |
| `crop_box_invalid` | review | A crop box is malformed or misses the media box; the media box is used. |
| `crop_box_outside_media_box` | review | A crop box extends past the media box and was clipped. |
| `rotation_invalid` | review | `Rotate` is not an integer multiple of 90; 0 is used. |

Reason codes appear in this fixed order on the report, and page-level codes
also appear on the page that produced them.

## Limits

| Argument | Default |
| --- | ---: |
| `max_bytes` | 64 MiB |
| `max_pages` | 10,000 |
| `max_objects` | 250,000 |
| `max_decompressed_bytes` | 32 MiB |

Streams are read in chunks and stop one byte past `max_bytes`. Limits that are
not positive integers, and sources that are neither bytes-like nor readable,
raise `PdfGeometryError` with a stable `category` and no submitted value.

## Serialization

`report.to_dict()` keeps a fixed field order, and `report.to_json()` returns
compact JSON with sorted keys and the schema identifier
`openmed.multimodal.pdf_geometry.v1`.

## Out of scope

Rendering, OCR, password recovery, repairing damaged files, and full PDF
conformance validation.

## Hidden-content inventory

`openmed.multimodal.pdf_inventory.read_pdf_inventory()` uses the same bounded,
dependency-free parser to inventory content outside the page text layer. This
Python parser slice does not render, extract attachments, redact form values,
or introduce a Swift PDF parser. Existing geometry results remain unchanged.

```python
from openmed.multimodal import PdfContentProfile, read_pdf_inventory

report = read_pdf_inventory(pdf_bytes, profile=PdfContentProfile.STRICT)
print(report.status.value, report.reason_codes)
if report.inventory is not None:
    print(report.inventory.field_value_count, report.inventory.revision_count)
```

The default `strict` profile rejects embedded attachments, JavaScript, Launch
and **any** OpenAction declaration, including a destination-only OpenAction.
The `review` profile reports these categories for explicit caller review.
Other hidden content requires review under both profiles. Nothing executes
scripts or actions, and neither profile authorizes downstream processing or
clinical decisions. Callers must resolve review findings before consequential
use; this reader does not implement reviewer approval or redaction.

| Code | Inventory evidence |
| --- | --- |
| `pdf_annotations` | Annotation definitions, with counts in closed subtype buckets. |
| `pdf_form_values` | Explicit non-null field `/V` or `/DV` declarations. |
| `pdf_embedded_files` | Embedded streams, `/EF` targets, file-attachment annotations or name trees. |
| `pdf_xfa` | XFA stream declarations or packet-array pairs. |
| `pdf_optional_content` | Optional-content groups or layer declarations. |
| `pdf_javascript` | JavaScript actions, `/JS` or JavaScript name trees. |
| `pdf_open_action` | OpenAction declarations. |
| `pdf_launch_action` | Launch actions, including nested additional actions. |
| `pdf_incremental_revisions` | Multiple top-level `startxref`/EOF pairs. |
| `pdf_inventory_invalid` | Malformed inventory structures, unresolved references or cyclic field/name trees. |

All geometry reason codes and ceilings apply. Unsupported or corrupt object
streams reject the inventory even if the current page tree can be read from
direct objects. Uncompressed object-stream payloads also consume the expansion
budget. Inventory failures return no counts; strict policy rejection retains
successfully read counts. Strings and stream payloads cannot introduce fake
revision markers or trailers into the inventory.

Counts describe parsed **definitions**, including superseded and unreferenced
indirect objects, rather than only the latest visible state. A field redefined
once with a different value contributes two field/value definitions and two
revisions. Fields count AcroForm-tree dictionaries and dictionaries with field
keys; a field with both `/V` and `/DV` contributes one value count. Attachments
count distinct embedded stream/target dictionaries, not attachment filenames.
XFA counts packet pairs or single declarations. Layer counts count `/OCG`
definitions; a layer declaration may still flag review with zero groups. Empty
name-tree declarations are conservatively flagged. Missing revision markers
use a baseline count of one, not a conformance assertion. Hybrid xref streams
and classic trailers in one revision do not create an additional revision.

Annotation subtype keys come from `PDF_ANNOTATION_SUBTYPES`; unrecognized names
become `other`. Reports and ordinary diagnostic representations contain counts
and controlled codes only: no field names/values, attachment names, scripts,
source paths, or source payloads. Strings are skipped rather than retained.
Serialization uses `openmed.multimodal.pdf_inventory.v1` with fixed fields.

### Asset preflight and abstention

`preflight_pdf_asset(manifest, source, content_profile=PdfContentProfile.STRICT)`
combines the existing manifest/type/resource/digest checks with inventory
findings under the additional `pdf_content` check. Input is read into memory
once, bounded by the minimum of the declared byte size, the resource profile
and 64 MiB, with at most one probe byte. Seekable streams are restored;
nonseekable streams are consumed once. The actual page tree is bounded by the
resource profile's page ceiling. A failed byte/digest/type check prevents
inventory evaluation.

The standalone inventory maps review to preflight `phi_uncertainty`, strict
policy rejection to `unsupported_media`, parser failure to `malformed_media`
and exhausted budgets to `resource_limit`. The asset preflight preserves any
earlier abstention reason and appends inventory codes after existing findings,
as described by `PDF_PREFLIGHT_CHECKS`.

The existing PDF manifest cannot express raster pixel budgets, so generic PDF
asset preflight currently reports `insufficient_metadata` for pixel rules.
These resource findings remain abstentions, including for plain PDFs; inventory
does not turn absent render-budget evidence into acceptance. `preflight_asset`
keeps its existing behavior and does not automatically inventory PDFs. Use the
standalone inventory when only the structural three-way verdict is needed.

A `readable` inventory means no supported hidden-content categories were found
within these parser bounds. It does not establish PHI absence, successful
redaction, full PDF conformance, model qualification or clinical validation.
