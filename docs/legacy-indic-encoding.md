# Legacy Indic encoding conversion

OpenMed can convert Devanagari text stored as ISCII-1991 bytes or as a
caller-described, ASCII-remapped legacy font, and Burmese text stored in the
Zawgyi font encoding, before script routing and PII detection. Conversion is
deterministic and local; it makes no network calls and does not log or persist
source text.

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
Auto-detection also requires the complete candidate to pass strict ISCII
decoding, including candidates with a Devanagari attribute prefix. Malformed
candidates fall back to Unicode: Python strings are preserved without partial
conversion or replacement characters. Byte input still follows the normal
UTF-8 decoding and error policy on that fallback. Explicit `encoding="iscii"`
keeps strict error handling by default; `errors="replace"` remains opt-in.

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
legacy glyph into multiple Unicode characters. Mapping files are limited to
1 MiB and each replacement to 64 Unicode code points to bound untrusted input
and expansion costs.

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

## Zawgyi Burmese detection and conversion

Zawgyi reuses Unicode Myanmar code points (U+1000-U+109F) with different
meanings and stores syllables in visual order, so Zawgyi text looks like
valid Unicode Burmese while breaking Unicode tokenizers and models. With the
default `encoding="auto"`, `convert_legacy_encoding` classifies any
Myanmar-script text with `detect_zawgyi` and converts it only when Zawgyi is
suspected:

```python
from openmed.processing.legacy_encoding import convert_legacy_encoding

# Synthetic Zawgyi: vowel sign E before its consonant, U+1039 as the killer.
zawgyi_name = "ေမာင္ေမာင္"
note = f"Patient: {zawgyi_name}, MRN 12345"
conversion = convert_legacy_encoding(note)

assert conversion.encoding == "zawgyi"
assert conversion.offset_unit == "code_point"
unicode_name = "မောင်မောင်"
start = conversion.text.index(unicode_name)
source_start, source_end = conversion.to_original_span(
    start,
    start + len(unicode_name),
)
assert note[source_start:source_end] == zawgyi_name
```

Offsets index whatever was passed in: UTF-8 byte offsets for `bytes` input
and code-point offsets for `str` input (`offset_unit` says which). Converted
syllables are reordered into Unicode storage order, so a character moved
across others maps to the whole source range it moved across; a detected span
never maps to a source range narrower than the characters that produced it.
Characters outside the Myanmar block, including Latin text and digits, are
copied unchanged with identity offsets.

The detector is deterministic and model-free. It counts structural signals
that are invalid in Unicode storage order but required by Zawgyi (vowel sign E
or medial RA written before the consonant, U+1039 ending a syllable) and
signals only Unicode storage produces (vowel sign E followed by a dependent
sign, the U+1004 U+103A U+1039 kinzi sequence, dot below before asat, and
code points the Zawgyi layout never assigns). Its confidence is the smoothed
Zawgyi share `(z + 0.5) / (z + u + 1)`, where Zawgyi presentation forms count
half. Text converts only when the confidence reaches
`ZAWGYI_CONFIDENCE_THRESHOLD` (0.85) and at least two structural signals are
present, so at least three independent signals are needed when there is no
Unicode evidence.

| Decision | Meaning | Text | `reason` |
| --- | --- | --- | --- |
| `zawgyi` | Zawgyi suspected above the threshold | Converted | `None` |
| `zawgyi` with `convert_zawgyi=False` | Suspected, conversion disabled | Unchanged | `legacy_encoding_unconverted` |
| `unicode` | Unicode Burmese confirmed above the threshold | Unchanged | `None` |
| `undetermined` | Short, mixed or ambiguous Myanmar text | Unchanged | `legacy_encoding_undetermined` |
| `not-myanmar` | No Myanmar-script code points | Unchanged | `None` |

`Pipeline(convert_zawgyi=False)` raises an `InputError` with
`details["reason"] == "legacy_encoding_unconverted"` instead of routing
suspected Zawgyi as if it were Unicode. Pipeline stage-one metadata and
`normalize_for_pii_detection(...).to_metadata()` expose the decision, the
confidence and the evidence counts. They never contain source text. Use
`encoding="zawgyi"` (or `zawgyi_to_unicode`) to force conversion of a known
Zawgyi run that is too short to classify, and `encoding="unicode"` to skip
detection for text known to be Unicode.

### Limits for Mon, Shan and other Myanmar-script languages

Mon, Shan, Karen, Kayah, Rumai Palaung and other Myanmar-script languages use
Unicode code points that Zawgyi reassigns to Burmese presentation forms, for
example Shan letters and tone marks in U+1075-U+108F. Those forms are only
weak evidence and can never trigger conversion on their own, so Unicode text
in these languages is reported as `undetermined` or `unicode` and left
unchanged. The detector and rule table cover Zawgyi Burmese only: legacy Mon
and Shan fonts and Zawgyi-styled text in those languages are not converted.
For known non-Burmese Myanmar-script input, pass `encoding="unicode"`.
Typing substitutions that also occur in Unicode text, such as digit zero for
WA or digit seven for RA, are left unchanged so numeric identifiers are never
altered.

## Provenance and limitations

The built-in ISCII assignments follow the public Indian standard
IS 13194:1991 and the Unicode Consortium's published ISCII semantics. The
mapping table was constructed independently for OpenMed and is distributed
under Apache-2.0. Caller-supplied legacy-font maps retain their own provenance
and license; OpenMed does not redistribute them.

The Zawgyi rule table and detector were written for OpenMed from the publicly
documented Zawgyi-One code-point layout, the Unicode Myanmar block chart and
the storage order in Unicode Technical Note #11. No third-party converter
code, rule file or detector model is copied or bundled, and no dependency is
added. `ZAWGYI_RULE_PROVENANCE` records the rule-set name, version, license
(Apache-2.0) and sources.

Vedic stress extension sequences can be decoded, but complete Vedic Sanskrit
accent and contextual invisible-letter fidelity is outside the byte-lossless
guarantee. OCR and image processing are also outside this text-only feature.
