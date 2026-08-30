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

Charts, pivot tables, macros, and VBA content are outside this adapter's scope.
Only `.xlsx` files are accepted.
