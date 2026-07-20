---
name: ingesting-clinical-documents
description: "Turn scanned faxes, images, and CSV/CDA exports into clean text ready for OpenMed de-identification and NER, fully on-device. Use when the user has clinical documents (image scans, photographed/faxed notes, tabular CSV/TSV exports, C-CDA XML) and needs OCR or structured intake before openmed.deidentify and openmed.analyze_text, asks about openmed.multimodal, OCR engines (Tesseract / PaddleOCR), tabular redaction, or layout and reading order. Covers the verified ocr() and redact_document() entry points and the ExtractedDocument contract. Pairs before deidentifying-clinical-text and extracting-clinical-entities."
license: Apache-2.0
metadata:
  project: OpenMed
  category: imaging-ocr
  pairs: before
  version: "1.0"
---

# Ingesting Clinical Documents

Clinical text often arrives as scanned faxes, photographed notes, CSV exports, or
C-CDA XML — not plain text. `openmed.multimodal` converts these into a normalized
`ExtractedDocument` (clean text + character-offset → source-location spans) so you
can run de-identification and NER. It runs **on-device**: OCR backends are local,
no document leaves the machine.

## When to use

- You have **images / scanned faxes** of clinical notes and need text out (OCR).
- You have **CSV/TSV** patient exports that need column-aware handling.
- You have **C-CDA XML** to flatten into text.
- You are building the **intake stage** that feeds `openmed.deidentify` and
  `openmed.analyze_text`.

This is the **first** stage. After intake, hand off to
`deidentifying-clinical-text` then `extracting-clinical-entities`.

## What is supported today

`redact_document` dispatches by file extension. Live handlers:

| Input | Extensions | Path |
| --- | --- | --- |
| Images / scans | `.png .jpg .jpeg .tif .tiff .bmp .gif .webp` | OCR (`ocr()` / image handler) |
| Tables | `.csv .tsv` | column-aware tabular redaction |
| C-CDA | `.xml` (detected as CDA) | stdlib CDA adapter |

**PDF and DOCX have no live handler yet** — `redact_document("x.pdf")` raises
`UnsupportedDocumentError`. Convert PDFs to page images first (or to text with your
own tool) and feed the images through OCR. See
[references/multimodal-ingest.md](references/multimodal-ingest.md) for the full
contract, engines, and the tabular pipeline.

## Install

```bash
pip install "openmed[multimodal]"      # document intake contract + image deps
pip install "openmed[ocr-paddle]"      # add the PaddleOCR engine
# Tesseract engine also needs the system binary, e.g.:  brew install tesseract
```

## Quick start: OCR an image, then de-identify

The clean two-step intake path. `ocr()` lives in the submodule (it is intentionally
not re-exported from `openmed.multimodal`):

```python
from openmed.multimodal.ocr import ocr
import openmed

# 1) OCR a scanned/faxed note -> OcrResult -> ExtractedDocument -> plain text
result = ocr("fax_page.png", engine=None)   # None = auto-select an installed engine
doc    = result.to_document()                # ExtractedDocument
text   = doc.text                            # clean text for downstream OpenMed

# 2) De-identify, then run NER (privacy-first order)
deid = openmed.deidentify(text, method="mask", policy="hipaa_safe_harbor")
ner  = openmed.analyze_text(deid.deidentified_text, output_format="dict")

for ent in ner.entities:
    print(ent.label, ent.text, ent.confidence)
```

`engine` may be `None` (auto-select), `"tesseract"`, `"paddleocr"`, or an
`OcrEngine` instance. `OcrResult` exposes `.text` and per-word boxes via `.words`
(each `OcrWord` has `text`, `bbox`, `confidence`, `page`).

## One-step intake + redaction with `redact_document`

For images, CSV/TSV, and CDA, `redact_document` performs intake **and**
de-identification in a single, format-aware call, returning an already-redacted
`ExtractedDocument`:

```python
from openmed.multimodal import redact_document

# Image scan: OCR + redact in one call
doc = redact_document("fax_page.png")
print(doc.text)        # redacted text
print(doc.spans[:3])   # SourceSpan offsets -> page / bbox in the original scan

# CSV export: per-column classification (direct id / quasi-id / safe) + redaction
table_doc = redact_document("patients.csv")
print(table_doc.text)
```

Use `redact_document` when you want OpenMed to own intake **and** redaction
(especially for tables, where redaction is column-scoped, not free-text NER). Use
the `ocr()` → `to_document()` → `deidentify` path when you want to control the
de-identification method, policy, or mapping yourself.

## Tabular CSV/TSV redaction

CSV columns get classified before any cell is touched, so a free-text NER pass is
not run blindly over structured data:

```python
from openmed.multimodal import read_table, redact_table

view = read_table("patients.csv")            # TableView with column decisions
for col in view.columns:
    print(col.name, "->", col.assigned_class, col.action, col.canonical_label)

redacted = redact_table("patients.csv", keep_year=True)
print(redacted.text)            # redacted CSV
for entry in redacted.manifest: # PHI-SAFE audit: counts/actions per column, no raw values
    print(entry)
```

`redact_table(...)` returns a `RedactedTable` with `.text`, `.headers`, `.rows`,
`.columns`, and a PHI-safe `.manifest` (no raw cell values). See
[references/multimodal-ingest.md](references/multimodal-ingest.md) for column
classes and actions.

## Preserve layout / reading order and map back to the source

Every `ExtractedDocument` keeps character offset → source location. After detecting
PHI on `doc.text`, project a span's offset back to its page and bounding box:

```python
from openmed.multimodal.ocr import ocr
import openmed

doc  = ocr("fax_page.png").to_document()
deid = openmed.deidentify(doc.text, method="mask")

for ent in deid.pii_entities:
    loc = doc.location_at(ent.start)   # SourceSpan or None
    if loc is not None:
        print(ent.label, "page", loc.page, "bbox", loc.bbox)
```

This lets you redact pixels on the original scan, not just the extracted text.

## Hand-off to / from OpenMed

- **To `deidentifying-clinical-text`:** pass `doc.text` to `openmed.deidentify(...)`
  with a policy profile; this is the required next stage for PHI.
- **To `extracting-clinical-entities`:** run `openmed.analyze_text` on the
  **redacted** text, not raw OCR output.
- **From file conversion (out-of-process):** for PDFs/DOCX, render to page images
  with your own tool, then OCR those images through this skill.

## Edge cases & gotchas

- **`ocr()` is imported from the submodule:** `from openmed.multimodal.ocr import
  ocr`. It is deliberately not re-exported from `openmed.multimodal`.
- **No PDF/DOCX handler yet:** `redact_document` raises `UnsupportedDocumentError`
  for them. Rasterize to images first.
- **OCR needs a backend:** install `[ocr-paddle]` for PaddleOCR, or the system
  Tesseract binary for `pytesseract`. Missing backends raise
  `MissingDependencyError` with an install hint.
- **OCR is noisy:** misreads lower downstream recall. Prefer higher-DPI scans;
  inspect `OcrWord.confidence` to flag low-quality pages.
- **Tables are not free text:** `redact_table` redacts per column classification —
  don't run whole-table NER and expect structured columns to be handled correctly.
- **No raw PHI in artifacts:** the table `manifest` and any logs record
  counts/actions/labels, never raw values. Keep OCR intermediates on-device and out
  of logs.
- **Local-first:** OCR engines run locally; do not send scans to a cloud OCR API in
  a PHI workflow.

## Standards & references

- HIPAA de-identification:
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- HL7 C-CDA: https://www.hl7.org/implement/standards/product_brief.cfm?product_id=492
- Full intake contract, engines, and table pipeline:
  [references/multimodal-ingest.md](references/multimodal-ingest.md)
