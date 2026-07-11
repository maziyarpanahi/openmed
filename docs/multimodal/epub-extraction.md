# EPUB Extraction

OpenMed can extract text from EPUB books through the shared multimodal document
contract. The EPUB ingester walks the package spine in reading order, reads each
XHTML/HTML content document, strips markup, and returns an `ExtractedDocument`
with offsets back to the source content item.

```python
from openmed.multimodal import extract_epub

document = extract_epub("patient-education.epub")

print(document.text)
print(document.metadata["sections"])
```

Each `SourceSpan` maps a range in `document.text` to one XHTML source item:

```python
offset = document.text.index("Jane Roe")
span = document.location_at(offset)

print(span.metadata["section_href"])
print(span.metadata["source_start"], span.metadata["source_end"])
```

The metadata is PHI-safe: it includes section identifiers, EPUB paths, document
offsets, and source character ranges, but it does not store raw XHTML content.

## Behavior

- `.epub` files are discoverable through `redact_document`.
- EPUB text is extracted from `application/xhtml+xml` and `text/html` spine
  items.
- `head`, `script`, and `style` content is ignored.
- Character references such as `&amp;` are decoded in extracted text while the
  source span still points back to the original reference range.
- Section boundaries are available in `document.metadata["sections"]`.

DRM-protected or encrypted EPUB entries are unsupported. Repackaging a redacted
EPUB is also out of scope; callers can use source offsets to project findings
back to XHTML content documents when they need custom write-back behavior.
