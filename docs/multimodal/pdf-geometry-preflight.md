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
