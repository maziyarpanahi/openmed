# RTF Extraction

RTF is a common export format for legacy clinical and dictation systems.
OpenMed extracts its body text through the shared multimodal document contract:
the ingester walks the RTF control stream, skips destination groups that hold
markup or metadata rather than document text, decodes control words, control
symbols, and Unicode escapes, and returns an `ExtractedDocument` with offsets
back into the original file.

```python
from openmed.multimodal import extract_rtf

document = extract_rtf("dictation.rtf")

print(document.text)
print(document.metadata["encoding"])
```

Each `SourceSpan` maps a range in `document.text` to the range of RTF source it
came from:

```python
offset = document.text.index("Jane Roe")
span = document.location_at(offset)

print(span.metadata["source_start"], span.metadata["source_end"])
```

RTF keeps its control stream in 7-bit ASCII and encodes everything else through
escapes, so a `source_start`/`source_end` pair is both a character offset into
the source read as a single-byte stream and a byte offset into the file on disk.
That lets redaction locate a span in the original document without re-parsing
it.

The metadata never embeds raw RTF content. It records the source path, resolved
codepage, RTF version, and character ranges. Because filenames and paths can
themselves be sensitive, treat `source_path` as input provenance and do not log
it verbatim.

## Behavior

- `.rtf` files are discoverable through `redact_document`, with no dependency
  beyond the standard library.
- Destination groups such as `\fonttbl`, `\colortbl`, `\stylesheet`, `\info`,
  `\pict`, and headers and footers are skipped, as is any unknown destination
  marked with `\*`.
- `\par`, `\line`, `\sect`, `\page`, `\row`, and an escaped line break become
  newlines. Consecutive breaks collapse and document-final breaks are trimmed,
  so the structural newlines carry no source range.
- `\tab`, `\cell`, the dash and quote control words, and control symbols such as
  `\{`, `\}`, `\\`, and `\~` are decoded to the characters they stand for and
  keep a source range covering the escape.
- `\'hh` escapes are decoded with the document codepage, taken from `\ansicpg`
  when present and defaulting to `cp1252`. Adjacent escapes are decoded together
  so multi-byte codepages round-trip.
- `\uN` escapes are decoded to Unicode, honoring the group-scoped `\ucN`
  fallback count so the replacement characters never reach extracted text.
  Surrogate pairs are combined.
- `\binN` payloads are skipped by length rather than parsed as text.

A file that does not start with an `{\rtf` header, or that yields no body text,
raises `UnsupportedDocumentError`. Re-rendering a redacted RTF with its original
styling is out of scope; callers can use source offsets to project findings back
onto the RTF when they need custom write-back behavior.
