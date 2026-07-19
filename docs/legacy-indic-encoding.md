# Legacy Indic encoding conversion

OpenMed can convert Devanagari text stored as ISCII-1991 bytes or as a
caller-described, ASCII-remapped legacy font before script routing and PII
detection. Conversion is deterministic and local; it makes no network calls
and does not log or persist source text.

## ISCII conversion and offsets

Pass raw bytes when source-byte offsets matter:

```python
from openmed.processing import iscii_to_unicode, unicode_to_iscii

source = bytes.fromhex("cf cc e1 d5 20 d5 cf e8 cc da")
conversion = iscii_to_unicode(source)

assert conversion.text == "रमेश शर्मा"
assert unicode_to_iscii(conversion.text) == source

start = conversion.text.index("रमेश शर्मा")
original_start, original_end = conversion.to_original_span(
    start,
    start + len("रमेश शर्मा"),
)
assert source[original_start:original_end] == source
```

The converter handles the ISCII Devanagari attribute sequence, standard
nukta combinations, double danda, and the explicit and soft halant sequences.
Output is placed in logical Unicode order, normalized to NFC, and aligned to
the original byte stream. A Latin clinical note does not meet the conservative
ISCII detection gate and is returned unchanged by `convert_legacy_encoding`.

## User-supplied legacy-font maps

OpenMed intentionally bundles no proprietary legacy-font mapping data. Supply
a JSON or YAML file for a font whose mapping you are licensed to use:

```json
{
  "name": "hospital-archive-font",
  "provenance": "licensed by the data owner",
  "mapping": {
    "0x41": "र",
    "B": "ा",
    "67": "म"
  }
}
```

Keys may be byte integers, single-byte characters, decimal byte strings, or
hexadecimal byte strings. Values are Unicode strings and may expand a single
legacy glyph into multiple Unicode characters.

```python
from openmed.processing import LegacyFontMap, convert_legacy_encoding

font_map = LegacyFontMap.from_file("hospital-archive-font.json")
conversion = convert_legacy_encoding(
    b"ABC",
    encoding="legacy-font",
    legacy_font_map=font_map,
)
assert conversion.text == "राम"
```

Auto-detection converts only dense runs with at least three mapped bytes and
at least two resulting Devanagari letters. Use
`encoding="legacy-font"` when a known, very short run is below that deliberately
conservative threshold.

## Provenance and limitations

The built-in ISCII assignments follow the public Indian standard
IS 13194:1991 and the Unicode Consortium's published ISCII semantics. The
mapping table was constructed independently for OpenMed and is distributed
under Apache-2.0. Caller-supplied legacy-font maps retain their own provenance
and license; OpenMed does not redistribute them.

Vedic stress extension sequences can be decoded, but complete Vedic Sanskrit
accent and contextual invisible-letter fidelity is outside the byte-lossless
guarantee. OCR and image processing are also outside this text-only feature.
