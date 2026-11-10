# OCR without temporary input files

`StreamingTesseractEngine` is an explicit adapter for a locally installed
Tesseract executable and language data. It encodes one image in memory and uses
Tesseract's documented stdin/stdout interface with TSV output. It does not
download models, change the default OCR engine, or create temporary image/text
files. [Tesseract command-line reference](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc)

```python
from openmed.multimodal.ocr import ocr
from openmed.multimodal.ocr_tesseract_stream import StreamingTesseractEngine

engine = StreamingTesseractEngine(
    timeout_seconds=30,
    max_pixels=16_000_000,
    max_words=25_000,
    max_text_chars=100_000,
)
result = ocr("local-synthetic-scan.png", engine=engine, languages=["de", "en"])
document = result.to_document(preserve_lines=True)
```

Pillow and the native Tesseract binary/language files are required. The adapter
accepts a Pillow image, encoded image bytes, a path or a binary image stream.
Input must contain one frame. The pixel budget is checked before conversion;
transparent pixels are composited on white without changing source dimensions.
Coordinates refer to decoded pixels; EXIF orientation is not applied.
The native deadline applies to Tesseract execution, not image decoding. A
hosting service must separately bound its entire ingestion request and isolate
native image/PDF parsing.

Native output is read while input is written, preventing pipe deadlocks. A
deadline or optional thread-safe `cancel_check` stops and reaps the owned OCR
process. TSV output is bounded before accumulating it, and word/text limits
are checked during parsing. Native stderr and exception details are excluded
from errors. The process receives a restricted environment and one OpenMP
thread. An explicit executable or `tessdata_dir` is trusted host configuration;
neither should be supplied by untrusted document content or request arguments.

Returned words retain zero-based page indexes, pixel rectangles, normalized
confidence and numeric block/paragraph/line identifiers. Invalid UTF-8, malformed
records, duplicate word IDs, unsupported languages and invalid/out-of-image
geometry fail explicitly. Confidence is an OCR score, not a calibrated estimate
of clinical correctness. Empty OCR output may represent either a blank image or
missed content; it is not a completeness guarantee.

`result.text` keeps the common space-joined OCR contract. For clinical headings,
use `result.to_document(preserve_lines=True)` to retain the native engine's
line boundaries and word order. This conversion requires valid, contiguous
native line IDs and preserves each word's page, confidence and rectangle. With
the default single-space separator, word offsets also remain unchanged. Review
reading order separately, especially for columns or tables; this conversion
does not reconstruct layout or establish content completeness. OCR outputs
remain `preview`; no clinical language or
privacy capability is qualified by this adapter. It performs OCR only, without
redacting pixels, sanitizing a PDF, or verifying that all identifiers were found.

## Reconstruct scanned-page layout

`parse_layout` orders positioned OCR words by page and column, retaining a
pixel box for every emitted character range. It separates visual lines so
clinical section headings can be detected on the reconstructed text. When the
OCR result includes `page_dimensions` metadata, it validates each pixel box
against that page and identifies isolated top and bottom bands. Repeated rows
with at least three aligned cells become tables. Ambiguous rows remain ordinary
text; no table structure is guessed from a two-column note. Without page
dimensions, only a clearly isolated first or last line is treated as a band.

```python
from openmed.multimodal import parse_layout

layout = parse_layout(result)
sections = layout.detect_sections()
for table in layout.tables:
    structured_table = table.as_structured_table()
    # Each cell's start/end indexes layout.text and retains its pixel boxes.
```

`layout.bbox_for_span(start, end)` projects extracted spans back to source
pixels. `layout.offsets_for_bbox(page, bbox)` performs the exact reverse lookup.
Table cells use the existing `openmed.structured.Table` shape, so downstream
structured extraction can consume their offsets without a second grid schema.
Synthetic layout fixtures report exact-position reading-order and table-cell
assignment accuracy through `evaluate_layout`; the committed two-column lab
fixture scores 1.00 for both. OCR confidence is carried through unchanged and
does not certify the extracted clinical content.

Layout parsing is bounded to 4,096 input words, 4,096 characters per word or
separator, and 1 MiB of source word text. Confidence must be finite and within
[0, 1]; boolean coordinates, ambiguous geometry, conflicting page dimensions,
and non-integer projection offsets are rejected. Public parse and projection
errors discard raw conversion and iterator exception context. Output text and
metadata remain sensitive source data and must not be logged as audit records.
