---
name: extracting-lab-tables
description: "Detects and extracts tabular laboratory panels from PDFs, scans, and images into structured rows ready for OpenMed and FHIR. Use when the user has a CBC, CMP, lipid panel, or other lab report as a scanned image / PDF / spreadsheet and needs the test name, value, unit, reference range, and abnormal flag as clean rows. Trigger keywords: lab table extraction, lab panel, OCR labs, table detection, layout analysis, header detection, reference range column, abnormal flag column, LOINC, UCUM, CBC, CMP, structured labs. Pairs before OpenMed: OCR/parse the table on-device (openmed.multimodal.ocr.ocr, read_table), de-identify embedded PHI with openmed.deidentify, then hand structured rows to LOINC/UCUM mapping and openmed.clinical lab flagging. Image/CSV/TSV intake is supported; PDF/DOCX raise UnsupportedDocumentError — render those to images or text first."
license: Apache-2.0
metadata:
  project: OpenMed
  category: imaging-ocr
  pairs: before
  version: "1.0"
---

# Extracting lab tables from documents and scans

Lab results arrive as **tables**: a column of test names, a value column, units,
a reference range, and an abnormal flag (H/L/Crit). To use them downstream you
must recover that grid from a PDF, scan, or spreadsheet into clean rows — then
code each test to **LOINC**, normalize units with **UCUM**, and flag abnormals.

This skill is the **intake** step: it OCRs/parses the table on-device with
`openmed.multimodal`, de-identifies any embedded PHI, and emits structured rows.
It pairs **before** OpenMed's clinical helpers — the LOINC/UCUM coding and the
high/low/critical flag are downstream (see `parsing-lab-values`).

## When to use

- You have a lab report as a **scanned image / photo / PDF page** and need the
  panel as rows, not pixels.
- The source is a **CSV/TSV** export and you need columns classified (which is
  the value, the unit, the range, the flag) and PHI columns redacted.
- You need machine-readable rows to feed LOINC mapping and a FHIR
  `Observation`/`DiagnosticReport`.

## What OpenMed gives you here

`openmed.multimodal` ships the intake primitives (no heavy deps at import; the
OCR backend loads lazily):

- `openmed.multimodal.ocr.ocr(image, engine=...)` → an `OcrResult` whose `.words`
  are `OcrWord(text, bbox, confidence, page)` and `.text` is the joined string.
  `OcrResult.to_document()` bridges each word (with its pixel bbox) into an
  `ExtractedDocument` so detected PHI can project back to the source location.
- `read_table(...)` → a `TableView` (`headers`, `rows`, `delimiter`,
  `has_header`, `columns`) for delimited text; `classify_columns(...)` labels
  each column; `redact_table(...)` → a `RedactedTable` with a PHI-safe `manifest`.

Engines: Tesseract (`pip install "openmed[multimodal]"` + the system binary) or
PaddleOCR (`pip install "openmed[ocr-paddle]"`). `ocr()` auto-selects the first
installed backend.

## Quick start

```python
from openmed.multimodal.ocr import ocr
from openmed.multimodal import read_table, classify_columns, redact_table

# A) Scanned / image lab report -> words with pixel boxes.
result = ocr("cbc_report.png")               # OcrResult
for w in result.words[:5]:
    print(repr(w.text), w.bbox, round(w.confidence, 2), "p", w.page)

doc = result.to_document()                   # ExtractedDocument; bbox preserved

# B) Delimited lab export (CSV/TSV) -> classified, PHI-redacted rows.
csv_text = (
    "PatientName,Test,Value,Unit,RefRange,Flag\n"
    "Jane Roe,Hemoglobin,9.1,g/dL,12.0-15.5,L\n"
    "Jane Roe,Glucose,148,mg/dL,70-99,H\n"
)
view = read_table(csv_text)                  # TableView
view = classify_columns(view)                # tag PHI vs data columns
redacted = redact_table(view)                # RedactedTable: PatientName redacted

for row in redacted.rows:
    print(row)                               # name column masked; lab data intact
for col in redacted.manifest:                # PHI-safe per-column audit manifest
    print(col["column_name"], col["assigned_class"], col["action"])
```

For an OCR'd (image) table, you reconstruct the grid yourself from word
boxes (next section) — OCR yields positioned words, not a delimited table.

## Workflow

1. **Detect the source type.** CSV/TSV → `read_table`. Image/scan → `ocr()`.
   PDF/DOCX are **not** directly parseable (they raise `UnsupportedDocumentError`);
   render PDF pages to images first, or extract their text layer, then OCR.
2. **OCR with positions.** `ocr()` returns `OcrWord`s carrying `bbox` and `page`.
   Keep the boxes — they let you cluster words into rows/columns and project PHI
   redaction back to pixels.
3. **Reconstruct the grid.** Cluster words by their `bbox` *y* into rows, by *x*
   into columns. The header row names the columns; align body cells to those x
   bands. Confidence (`OcrWord.confidence`) flags shaky cells for review.
4. **Identify the lab columns.** Map headers to roles: *test name*, *value*,
   *unit*, *reference range*, *flag*. For delimited input, `classify_columns`
   tags PHI columns (name/MRN/DOB) so `redact_table` masks them.
5. **De-identify embedded PHI.** Patient name/MRN often sit in the table header or
   a leading column. Redact those columns (`redact_table`) and run free-text
   cells through `openmed.deidentify` before the rows leave the device.
6. **Emit structured rows** `{test, value, unit, ref_range, flag}` per result and
   hand off to LOINC/UCUM coding and `parsing-lab-values`.

## Hand-off to / from OpenMed

- **To** `parsing-lab-values` (`openmed.clinical.parse_reference_range`,
  `derive_abnormal_flag`): pass the parsed value + `ref_range` (+ any explicit
  lab flag) to get a structured low/normal/high/critical signal.
- **To** `mapping-loinc`: code each test name to a LOINC code; normalize the unit
  with UCUM. OpenMed emits the row; the terminology binding is out-of-process.
- **To** FHIR (`exporting-to-fhir`): each row becomes an `Observation`
  (`code`=LOINC, `valueQuantity` with UCUM `unit`, `referenceRange`,
  `interpretation`) grouped under a `DiagnosticReport`.
- **De-identify** with `deidentifying-clinical-text` (`openmed.deidentify`)
  before export. OCR words carry pixel boxes so redaction maps back to the image.
- Everything here runs **on-device**; no scan or row leaves the process
  un-de-identified.

## Edge cases & gotchas

- **PDF/DOCX raise `UnsupportedDocumentError`.** The multimodal dispatcher has no
  PDF/DOCX handler — rasterize PDF pages to PNG (or pull the text layer) before
  calling `ocr()`. Image formats (PNG/JPG/TIFF/…) and CSV/TSV are handled.
- **OCR returns words, not a table.** You must reconstruct rows/columns from
  `bbox` geometry. Multi-line cells, wrapped test names, and merged header cells
  break naive x/y bucketing — tune the clustering tolerance per template.
- **Reference ranges are easy to mis-split.** "12.0-15.5", "<5", "70 - 99", and
  en/em dashes must survive OCR and tokenization as one cell. Don't let a space
  or a misread dash fracture the range — `parse_reference_range` downstream
  expects it whole.
- **Units belong to the value, not the range.** Keep "9.1 g/dL" and the range
  "12.0-15.5" in separate fields; the value's unit must match the range's unit or
  the abnormal flag will be wrong (the flag helper is unit-agnostic).
- **Low-confidence cells.** Gate on `OcrWord.confidence`; a 0.4-confidence value
  in a lab table is a patient-safety risk — route it to human review, don't
  silently accept it.
- **PHI hides in tables.** Patient name, MRN, DOB, and accession numbers commonly
  occupy the header or first column. Classify and redact them; never log the raw
  table.
- **Engine availability.** `ocr()` raises a clear `MissingDependencyError` if no
  backend is installed — install Tesseract or PaddleOCR per the extras.

## Standards & references

- LOINC — universal lab observation codes: https://loinc.org/
- UCUM — Unified Code for Units of Measure: https://ucum.org/
- HL7 FHIR R4 Observation (`valueQuantity`, `referenceRange`, `interpretation`): https://hl7.org/fhir/R4/observation.html
- HL7 FHIR R4 DiagnosticReport (lab grouping): https://hl7.org/fhir/R4/diagnosticreport.html
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- OpenMed source: `openmed/multimodal/ocr.py`, `openmed/multimodal/tabular_csv.py`.
