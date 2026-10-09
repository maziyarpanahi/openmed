# XLSX Cell Redaction

OpenMed can de-identify string cells in `.xlsx` workbooks while preserving
worksheets, formulas, numeric and date values, styles, and other workbook
structure supported by `openpyxl`.

Install the multimodal dependencies first:

```bash
pip install "openmed[multimodal]"
```

Then write a redacted copy to a different path:

```python
from openmed.multimodal import redact_xlsx

result = redact_xlsx(
    "clinical-workbook.xlsx",
    "clinical-workbook.redacted.xlsx",
)

for entry in result.redaction_report:
    print(entry)
```

By default, row 1 of each worksheet is treated as its header. OpenMed reuses
the CSV/TSV header and value-sampling rules to identify PHI columns. Every
other string cell is also analyzed individually, which catches PHI embedded in
free text even when its column looks safe. Use `header_row` to select a
different one-based header row, or pass the same `header_heuristics` and
`action_overrides` mappings accepted by the CSV adapter.

Formula cells are identified from their workbook cell type and are never sent
to the de-identifier. Numeric, Boolean, date, blank, and error cells are also
left untouched. The source workbook cannot be used as the output path, so a
failed or partial run cannot overwrite the original PHI-bearing file.

## PHI-safe report

Each changed cell produces a record shaped like this:

```python
{
    "sheet_index": 0,
    "coordinate": "C7",
    "labels": ["PERSON", "PHONE"],
}
```

The report intentionally omits original values, replacements, and worksheet
names. Worksheet names can themselves contain PHI, so worksheets are addressed
by their zero-based position instead. Store and transmit the redacted workbook
under the same controls used for other clinical artifacts; formulas can still
encode sensitive logic even when their displayed string cells are redacted.

Charts, pivot tables, macros, and VBA content are refused at the package
verification boundary. Only `.xlsx` files are accepted.

## Office package verification

The DOCX, XLSX and PPTX writers apply a mandatory local, fail-closed policy
before publishing any output. The original ZIP is checked as well as the staged
output, so a writer silently dropping an unknown part cannot bypass the gate.
Visible-text extraction, offsets, cell classification and formula handling are
unchanged. This gate verifies coverage, not detector accuracy or clinical safety.

- Core, app and custom properties are cleared; custom XML, thumbnails and printer
  settings are removed with their relationships and content-type entries.
- Drawing alt-text descriptions/titles and shape names are cleared. Presentation
  master and layout text is cleared while retaining formatting. This can remove template
  prompts, accessibility descriptions and document metadata.
- Tracked changes (including authors and deletions), comments, foot/endnotes,
  hidden sheets, defined names, pivot caches, external links, embedded objects
  and opaque media are refused. Unknown parts, XML comments/processing
  instructions, malformed XML, ZIP comments or extra fields and text outside covered runs/cells are also refused.
- DOCX headers/footers and PPTX speaker-note runs already handled by extraction
  stay covered. Orphan headers/footers and other note text are refused; XLSX
  headers/footers are refused. Unused shared strings are refused.

The policy has no permissive bypass or metadata allowlist. Technical formatting
XML and validated table-style UUIDs are retained. Cell headers, formulas and
worksheet names remain governed by the existing adapter contract; they can
contain sensitive content and need separate caller review. Image/OCR processing
must happen separately; this gate does not inspect image pixels.

Catch the typed refusal without logging a source payload or path:

```python
from openmed.multimodal.ooxml_residual import OoxmlResidualError

try:
    result = redact_xlsx(
        "clinical-workbook.xlsx",
        "clinical-workbook.redacted.xlsx",
    )
except OoxmlResidualError as refusal:
    evidence = [finding.to_dict() for finding in refusal.findings]
    # Example fields: category="hidden_sheets", count=1, digest=<sha256>.
```

`inspect_ooxml(package_bytes)` and `verify_ooxml(package_bytes)` use only the
standard library. Findings and typed errors include only controlled categories,
counts and digests, never part names, authors, text or paths. Without internal
writer coverage, ordinary visible text is also reported as unverified. Coverage
is an internal child-address map built from the actual run/cell write paths;
callers must not invent coverage to qualify a file.

Office bytes are staged in memory and verified before an atomic disk replacement
or the first write to a caller-supplied stream. Streams must be seekable and
writable; successful writes replace their contents and truncate any old trailing
bytes. Unsupported streams receive an `invalid_destination` refusal. Refusal preserves sources and
existing destinations. The XLSX API still disallows in-place writes; DOCX/PPTX
allow verified in-place writes. No unverified Office package is written to a
temporary file. Package verification refuses more than 4,096 entries or 128 MiB
of decompressed content. It adds no network calls or dependencies.
