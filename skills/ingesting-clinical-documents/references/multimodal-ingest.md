# OpenMed multimodal intake — reference

Detailed lookup for `openmed.multimodal`. The SKILL.md covers the common path;
this page documents the data contract, OCR engines, and the tabular pipeline. Use
only the names listed here — they are verified against the package's public
exports.

## Public surface (`openmed.multimodal.__all__`)

```text
ExtractedDocument, SourceSpan            # the universal intake contract
redact_document, register_handler        # dispatcher + extension registration
ensure_multimodal_available              # raises if the [multimodal] extra is absent
MissingDependencyError, UnsupportedDocumentError
OcrResult, OcrWord, OcrEngine, FakeOcrEngine, register_ocr_engine
ColumnDecision, TableView, RedactedTable # tabular types
read_table, classify_columns, redact_table
```

Note: **`ocr()` is not in `__all__`** — import it from the submodule:
`from openmed.multimodal.ocr import ocr`. It is kept there so the function name does
not shadow the re-exported data types.

## The `ExtractedDocument` contract

Every intake path (OCR, CSV, CDA) converges on this frozen dataclass.

```text
ExtractedDocument
  .text: str                      # normalized text
  .spans: tuple[SourceSpan, ...]  # char offset -> source location
  .metadata: Mapping[str, Any]    # format-specific
  .location_at(offset) -> SourceSpan | None   # map a char offset to its source
  .text_for(span) -> str          # extract the text for a span
  .from_blocks(...)               # classmethod: assemble from ordered blocks

SourceSpan
  .start: int                     # inclusive char offset
  .end: int                       # exclusive char offset
  .page: int = 0
  .bbox: tuple[float, float, float, float] | None   # (x0, y0, x1, y1)
  .metadata: Mapping[str, Any]
```

Use `location_at`/`spans` to project a detected PHI offset back to a page and
bounding box on the original scan — enabling pixel-level redaction, not just text
redaction.

## `redact_document` dispatch

```python
redact_document(path, *, policy=None, models=None) -> ExtractedDocument
```

Routes by file **extension** to a registered handler and returns an already
de-identified `ExtractedDocument`. Registered live handlers:

| Extensions | Handler |
| --- | --- |
| `.png .jpg .jpeg .tif .tiff .bmp .gif .webp` | OCR image handler |
| `.csv .tsv` | tabular CSV redactor |
| `.xml` (detected as C-CDA) | stdlib CDA adapter |

Unknown extensions (including **`.pdf` and `.docx`**, which have no live handler
yet) raise `UnsupportedDocumentError`. To extend, register your own:
`register_handler(extensions, handler, *, detector=None, requires_multimodal=True)`.

## OCR

```python
from openmed.multimodal.ocr import ocr
ocr(image, *, engine=None) -> OcrResult
```

`image` may be a path or a loaded image; `engine` is `None` (auto-select),
`"tesseract"`, `"paddleocr"`, or an `OcrEngine` instance.

```text
OcrResult
  .words: tuple[OcrWord, ...]
  .metadata: Mapping[str, Any]
  .text  -> str                 # words joined with spaces
  .to_document(*, separator=" ") -> ExtractedDocument

OcrWord
  .text: str
  .bbox: tuple[float, float, float, float]
  .confidence: float
  .page: int = 0
```

### Engines and extras

| Engine | `engine=` | Install |
| --- | --- | --- |
| Tesseract | `"tesseract"` | `pytesseract` + system Tesseract binary (`brew install tesseract` / `apt-get install tesseract-ocr`) |
| PaddleOCR | `"paddleocr"` | `pip install "openmed[ocr-paddle]"` |
| Fake (tests) | `FakeOcrEngine` | bundled; deterministic, for unit tests |

`engine=None` auto-selects the first installed backend. A missing backend raises
`MissingDependencyError` with an actionable install hint. Register custom engines
with `register_ocr_engine(name, factory)`.

## Tabular CSV/TSV pipeline

Columns are classified **before** any cell is touched, so structured data is not
processed as free text.

```python
read_table(source, *, delimiter=None, has_header=None, ...) -> TableView
classify_columns(headers, rows, ...) -> tuple[ColumnDecision, ...]
redact_table(source, *, policy=None, keep_year=True, date_shift_days=None,
             lang="en", ...) -> RedactedTable
```

### Column classes (`ColumnDecision.assigned_class`)

| Value | Meaning |
| --- | --- |
| `DIRECT_ID` | Direct identifier (name, MRN, SSN, email). |
| `QUASI_ID` | Quasi-identifier (DOB, ZIP, dates) — re-identifying in combination. |
| `SAFE` | No identifying signal detected. |

### Actions (`ColumnDecision.action`)

| Value | Effect |
| --- | --- |
| `mask` | Replace cell values with placeholders. |
| `hash` | One-way consistent token (links rows without revealing the value). |
| `drop` | Remove the column entirely. |
| `date_shift` | Shift dates by a consistent per-row offset (`shift_dates`). |
| `free_text_redact` | Run free-text PHI redaction on note-like columns. |
| `keep` | Leave the column unchanged. |

`ColumnDecision` also exposes `index`, `name`, `canonical_label`, `policy_label`,
`detection_source`, `confidence`, and `sampled_values`.

```text
TableView      .headers .rows .delimiter .has_header .columns
RedactedTable  .text .headers .rows .delimiter .has_header .columns
               .manifest  -> tuple[dict, ...]  (PHI-SAFE: counts/actions, no raw values)
               .to_document() -> ExtractedDocument
```

The `.manifest` is the audit trail and is deliberately **PHI-safe**: it records
per-column class, action, and counts — never raw cell contents.

## Privacy notes

- Everything runs **on-device**; OCR backends are local processes. Do not route
  scans to a cloud OCR API in a PHI workflow.
- Keep OCR intermediates (extracted text, page images) on-device and out of logs.
- The tabular `manifest` and any audit output must stay PHI-free — record labels,
  offsets, hashes, and counts, never raw identifiers.
- Use only permissively licensed OCR backends; restricted terminologies stay
  out-of-process under the user's own license.

## Verified import map

```python
from openmed.multimodal import (
    ExtractedDocument, SourceSpan, redact_document, register_handler,
    OcrResult, OcrWord, OcrEngine, FakeOcrEngine, register_ocr_engine,
    ColumnDecision, TableView, RedactedTable,
    read_table, classify_columns, redact_table,
    MissingDependencyError, UnsupportedDocumentError,
)
from openmed.multimodal.ocr import ocr   # not re-exported from the package root
```
