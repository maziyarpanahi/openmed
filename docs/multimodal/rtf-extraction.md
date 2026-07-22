# RTF Extraction

OpenMed can extract text from RTF (Rich Text Format) documents through the
shared multimodal document contract. RTF is a common export format for legacy
clinical and dictation systems. The ingester is stdlib-only: RTF's own syntax
is plain ASCII, so no third-party parser is required to read it.

```python
from openmed.multimodal import extract_rtf

document = extract_rtf("dictation-note.rtf")

print(document.text)
```

Each `SourceSpan` maps a range in `document.text` back to the exact byte
range in the original RTF file:

```python
offset = document.text.index("John Q. Public")
span = document.location_at(offset)

print(span.metadata["source_start"], span.metadata["source_end"])
```

The metadata is PHI-safe: spans carry numeric byte offsets only, never a copy
of the raw source text.

## Behavior

- `.rtf` files are discoverable through `redact_document`.
- Font, color, style, and list tables, document `\info` metadata (author,
  title, ...), headers, footers, footnotes, annotations, and embedded
  picture/object payloads are walked (to keep group nesting correct) but
  never contribute to extracted text.
- `\uN` Unicode escapes are decoded to the real character; the
  `\ucN`-controlled fallback run that follows (for older readers) is
  consumed rather than duplicated into the output. Astral-plane characters
  (code points above U+FFFF, e.g. most emoji) are represented in RTF as a
  UTF-16 surrogate *pair* -- two consecutive `\uN` escapes -- and are
  recombined into the single real character they encode, rather than being
  emitted as two invalid lone surrogate code points (which would later crash
  a UTF-8 encode of the extracted text). A high or low surrogate escape with
  no matching partner is malformed/truncated input; it is replaced with the
  Unicode replacement character (`�`) rather than emitted raw.
- `\'hh` hex escapes are decoded using the document's declared code page
  (`\ansicpg`, defaulting to Windows-1252) rather than treated as raw
  Unicode code points, so escapes like curly quotes and em dashes decode to
  the correct character.
- `\*` marks a destination as ignorable; any such group is skipped even when
  its control word is not one OpenMed recognizes by name.
- `\binN` raw binary payloads (used by embedded OLE objects) are consumed
  verbatim, so arbitrary bytes inside them cannot be misread as RTF syntax
  and desynchronize group nesting for the rest of the document.
- `\par` maps to a newline; `\pard` (paragraph formatting reset, which
  commonly appears immediately after `\par`) does not, so paragraphs are not
  double-spaced.
- Table cells are linearized with `\t` per cell and `\n` per row, matching
  how DOCX and ODT table extraction are documented.

The source file is decoded with `latin-1`, not UTF-8: RTF's control syntax is
always ASCII, and anything outside that range is represented through an
escape, never a raw multi-byte sequence. `latin-1` is therefore a lossless,
one-byte-to-one-character decode, which is what lets every span's
`source_start`/`source_end` be an exact byte offset into the original file
rather than an offset into some intermediate re-encoding.

Re-rendering a redacted RTF file with its original styling, and OCR of
embedded images, are out of scope; callers can use source offsets to locate
and redact the corresponding bytes directly when they need custom write-back
behavior.
