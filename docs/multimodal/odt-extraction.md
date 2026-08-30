# ODT/ODF Text Extraction

OpenMed can extract text from OpenDocument Text (`.odt`) files through the
shared multimodal document contract. The ingester reads the ODF `content.xml`
stream in document order and returns an `ExtractedDocument` with character
offsets back to paragraphs, list items, and table cells.

```python
from openmed.multimodal import extract_odt

document = extract_odt("clinical-note.odt")

print(document.text)
print(document.metadata["paragraph_count"])
```

Each `SourceSpan` identifies the structure containing a text fragment without
copying raw source text into metadata:

```python
offset = document.text.index("Jane Roe")
span = document.location_at(offset)

print(span.metadata["block_type"])
print(span.metadata["paragraph_index"])
print(document.text_for(span))
```

## Reading order and tables

- Paragraphs, headings, and nested list items follow their order in
  `content.xml`.
- ODF explicit spaces, tabs, and line breaks are preserved and included in the
  offset map.
- Table cells are joined with tabs, and table rows are joined with newlines.
- Table spans include `table_index`, `row_index`, and `cell_index` metadata, so
  extracted offsets round-trip to a predictable source cell.
- `.odt` files are discoverable through `redact_document` without an additional
  parser dependency.

The extractor accepts OpenDocument Text only. ODS spreadsheets, ODP
presentations, encrypted archives, unsafe XML declarations, and writing
redactions back into ODT files are outside its supported surface.
