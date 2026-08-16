# Notebook Cell Redaction

OpenMed can redact markdown cell text and code cell outputs from Jupyter
notebooks without modifying code cell source. The notebook redactor processes
cells in natural order, redacting displayed text and removing binary image
data while preserving cell structure, output types, metadata, and execution
counts.

Install the optional notebook dependency when working with `.ipynb` files:

```bash
uv pip install -e ".[notebook]"
```

## Basic Use

```python
from openmed.interop.notebook_redaction import (
    NotebookRedactionPolicy,
    redact_notebook,
)

result = redact_notebook(
    "analysis.ipynb",
    policy=NotebookRedactionPolicy(
        redact_markdown=True,
        redact_outputs=True,
        action_overrides={"markdown": "mask"},
    ),
    output_path="analysis.redacted.ipynb",
)
```

The returned `NotebookRedactionResult` includes:

- `notebook`: the redacted notebook (an `nbformat.NotebookNode`), returned
  in-memory when `dry_run=True`.
- `summary`: a `NotebookCellSummary` with PHI-safe aggregate counts.

The `NotebookCellSummary` includes:

- `total_cells`: total number of cells in the notebook.
- `redacted_cells`: number of cells that were modified.
- `cell_type_counts`: mapping of cell type to count (e.g. `{"markdown": 3,
  "code": 4, "raw": 1}`).
- `cell_records`: per-cell `NotebookCellRecord` entries preserved in cell
  index order.

Each `NotebookCellRecord` includes:

- `index`: cell position in the notebook's cell list.
- `cell_type`: the notebook cell type (`"markdown"`, `"code"`, or `"raw"`).
- `action`: the redaction action applied (from `ACTION_VALUES`).
- `redacted`: whether this cell was modified.

## Privacy Contract

Cell records contain only structural metadata — indices, cell types, actions,
and booleans. They do not include raw cell text, redacted cell text,
hashes, output content, or any cell source. The summary and cell records are
designed for audit and workflow coordination without exposing sensitive
values.

## MIME Type Handling

Code cell outputs are redacted based on MIME type. Text MIME types have their
string content replaced in-place. Binary MIME types are removed entirely from
the output `data` dict while preserving `output_type`, `metadata`, and
`execution_count`.

| MIME type | Handling |
|-----------|----------|
| `text/plain` | Redacted (string content replaced) |
| `text/html` | Redacted (string content replaced) |
| `application/json` | Redacted (string values replaced) |
| `image/png` | Removed from `data` dict |
| `image/jpeg` | Removed from `data` dict |
| `image/gif` | Removed from `data` dict |
| `image/svg+xml` | Removed from `data` dict |
| `application/pdf` | Removed from `data` dict |
| Unknown MIME types | Removed from `data` dict (conservative) |

Stream outputs (`output_type="stream"`) have their `text` field redacted
in-place.

## Notes

- SVG with embedded text is not redacted; the `image/svg+xml` MIME entry is
  removed from the `data` dict and treated as binary. A follow-up issue may
  add SVG text-element redaction.
- Code cell source is never modified. Only markdown cell source and code cell
  outputs are touched.
- Binary image data cannot be inspected for embedded PHI without OCR; this is
  out of scope for the notebook redactor.
- `application/json` output values are redacted as a whole — the entire value
  is replaced with the mask token, not individual fields within it. Distinguishing
  sensitive from non-sensitive fields in arbitrary JSON is out of scope for this
  helper. Users should not rely on this for selective field-level redaction of
  JSON output.
