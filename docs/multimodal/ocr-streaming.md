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
