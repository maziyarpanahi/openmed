# Redacted PDF Rendering And Fidelity

OpenMed can turn projected PDF PHI rectangles into a clean, locally rendered
PDF and produce enforceable, PHI-safe fidelity evidence. Install the optional
PDF stack first:

```bash
pip install "openmed[multimodal]"
```

## Render A Redacted PDF

`render_redacted_pdf` accepts the top-origin, 0-based page rectangles returned
by `project_text_spans`:

```python
from pathlib import Path

from openmed.multimodal import (
    extract_pdf,
    project_text_spans,
    render_redacted_pdf,
)

source = Path("synthetic-source.pdf")
output = Path("synthetic-redacted.pdf")
document = extract_pdf(source)

# These offsets normally come from an OpenMed PII detector.
start = document.text.index("SYNTHETIC_PATIENT")
end = start + len("SYNTHETIC_PATIENT")
rectangles = project_text_spans(document, [(start, end)])

result = render_redacted_pdf(source, output, rectangles)
assert result.passed
print(result.to_dict())
```

The destination is written atomically only after all verification gates pass.
Existing destinations are protected unless `overwrite=True` is explicit.

## What The Renderer Preserves

Each page is rendered locally at a fixed resolution, the requested rectangles
plus a one-point glyph-overhang safety margin are burned into the pixels, and a
new PDF is assembled at the original page size. This preserves visible
non-redacted content, page order, pagination, and page geometry while severing
the output from the source content streams.

An invisible clean text layer is rebuilt from extractable WinAnsi words that do
not touch a redaction rectangle. This keeps common non-PHI text selectable and
searchable without copying selected source words. Visible Unicode content is
still preserved in the page raster; words that cannot be represented safely in
the clean text-layer font are omitted from that optional layer rather than
encoded incorrectly.

The output intentionally does not preserve interactive forms, annotations,
embedded files, thumbnails, search indexes, or source metadata. If those are
required, treat them as separate data that must be independently de-identified.

## Mandatory Verification

Rendering runs three independent checks before publishing the output:

1. `verify_redacted_pdf` confirms each rectangle has no selectable residual
   text and is covered by an opaque box.
2. `verify_redacted_text_removed` accounts for every selected source-word
   occurrence across the complete output text layer. It detects partial,
   separated, reordered, split, merged, moved, and duplicated residual text
   while allowing unrelated identical words to remain. Use
   `assert_redacted_text_removed` for a raising helper.
3. `measure_pdf_layout_fidelity` confirms page count and dimensions, then masks
   the requested rectangles and measures changed pixels everywhere else.

```python
from openmed.multimodal import (
    assert_redacted_text_removed,
    measure_pdf_layout_fidelity,
)

assert_redacted_text_removed(source, output, rectangles)
layout = measure_pdf_layout_fidelity(source, output, rectangles, strict=True)
assert layout.pagination_preserved
```

The layout report records, per page, the compared non-redacted pixel count,
changed-pixel count and fraction, normalized mean absolute error, page sizes,
and pass/fail result. The default gate ignores channel differences of 4 or less
and allows at most a `0.0005` changed-pixel fraction outside redaction regions.
Pin the OpenMed lockfile and PDF stack when comparing reports across machines.

All report dictionaries omit input/output paths and plaintext. They contain
only geometry, counts, thresholds, and SHA-256 digests. Caller-supplied entity
labels are hashed before serialization, so reports can be attached to privacy
audit evidence without copying identifiers.

## Resource And Font Safety Limits

Raster work is bounded before page images are allocated. Defaults accept at
most 100 pages, 40 million pixels on one page, 100 million pixels in total, and
10,000 redaction rectangles. Trusted local callers can lower or deliberately
raise these limits with `max_pages`, `max_page_pixels`, `max_total_pixels`, and
`max_regions`; the active pixel limits are recorded in the fidelity report.

Type 3 fonts are rejected before rendering because their character procedures
can paint arbitrary content outside logical glyph bounds. Standard digital PDF
fonts receive the one-point burn margin described above. A rejected source is
never published as an output.

## Security And Licensing Boundary

The implementation uses only the existing `multimodal` extra and never sends a
document to a network service:

| Component | Role | License |
| --- | --- | --- |
| pdfplumber | Word geometry, extraction, and local page rendering | MIT |
| pypdfium2 / PDFium | Local raster backend used by pdfplumber | BSD-3-Clause / Apache-2.0 |
| Pillow | In-memory pixel redaction and comparison | MIT-CMU |
| pikepdf / qpdf | Fresh PDF assembly and deterministic serialization | MPL-2.0 / Apache-2.0 |

No AGPL renderer is imported or bundled, and no GPL bridge is needed.

Scanned or image-only source PDFs are outside this API's text-removal proof:
their redaction rectangles must come from the OCR pipeline, and residual text
must be verified by re-OCR. PDF redaction is an assistive privacy control, not a
diagnostic or clinical decision system.
