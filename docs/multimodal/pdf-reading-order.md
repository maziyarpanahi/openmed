# PDF Reading Order

Clinical reports and research papers often store a page as two or three
independent text columns. Reading every physical line from left to right can
interleave those columns, separate a name from its address, and give a PII
detector character offsets that no longer describe a coherent phrase.

OpenMed's PDF extractor detects repeated horizontal whitespace gutters and
reconstructs those pages in column-major order: top to bottom in the left
column, then top to bottom in the next column. Detection is automatic and
conservative. If a page does not have enough parallel lines to establish a
multi-column layout, its text, source spans, and metadata remain identical to
the original source-order extraction.

```python
from openmed.multimodal import extract_pdf, project_text_spans

document = extract_pdf("local-report.pdf")
finding_start = document.text.index("Avery Sample")
finding_end = finding_start + len("Avery Sample")

rectangles = project_text_spans(
    document,
    [(finding_start, finding_end)],
)
```

Every reordered word receives new character offsets but keeps its original
zero-based page number and `(x0, top, x1, bottom)` bbox. Consequently,
`project_text_spans` still returns rectangles at the word's original location
on the page. Reconstructed spans also record the source word index and detected
column index; full-width headings are marked as spanning columns. This metadata
contains geometry and indexes, not copied document text.

`redact_document` uses the same automatic extraction before invoking a supplied
PII detector, so detection sees coherent column text and its returned spans
project back to the source PDF:

```python
from openmed.multimodal import redact_document

result = redact_document(
    "local-report.pdf",
    models={"detector": local_detector},
)
```

Both parsing and detection remain local. OpenMed does not upload the PDF, add
telemetry, or copy raw source text into layout metadata.

## Source-order compatibility

Use `reading_order="source"` when an integration explicitly needs the original
OM-060/pdfplumber text-flow sequence:

```python
legacy = extract_pdf("local-report.pdf", reading_order="source")
```

The default `"auto"` mode changes only confidently multi-column pages. A
single-column document returned by auto mode compares equal to the same
document returned by source mode, including its text, word bboxes, offsets, and
metadata.

## Positioned-word API

Call `detect_pdf_columns` when positioned words have already been extracted:

```python
from openmed.multimodal import detect_pdf_columns

layout = detect_pdf_columns(words, page_width=page.width)
if layout.is_multicolumn:
    ordered_words = layout.ordered_words(words)
```

`PdfPageLayout.reading_order` is a permutation of the source word indexes.
`PdfPageLayout.columns` exposes each column's bbox and source indexes, and
`word_columns` maps every source word to a column. The detector uses only the
Python standard library and adds no dependency beyond the existing optional
`pdfplumber` extraction stack.

## Boundaries

- The detector supports one-, two-, and three-column pages by default.
- Repeated wide whitespace is required; ambiguous pages fall back to source
  order instead of guessing.
- Full-width headings are retained around column-major sections rather than
  duplicated or dropped.
- Reflowing or re-rendering a PDF is not performed.
- Rotated or skewed pages and ML layout models are outside this feature's
  scope.

All committed PDF tests use synthetic identities and addresses. Applications
should still review de-identification results before releasing clinical data;
layout reconstruction is an assistive document-processing feature, not a
clinical decision system.
