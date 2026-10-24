# Burned-in font coverage and shaping

`BurnedInGenerator` draws synthetic names, dates, and MRNs into image pixels
and returns pixel boxes as OCR truth. A box is only truthful if the font can
draw every character and, for complex scripts, if the text is shaped. The
generator therefore checks each span before drawing it and fails closed with a
typed, value-free error instead of rendering missing-glyph boxes or unjoined
letters.

## What is checked

For every rendered span (`PERSON`, `DATE`, `ID_NUM`):

1. **Glyph coverage.** Each code point must map to a real glyph in the
   character map (`cmap`) of the exact font bytes used to draw it. Controls
   and default-ignorable formatting characters such as ZWJ and ZWNJ are
   exempt. The reader follows FreeType's choice of Unicode subtable and treats
   glyph 0 (`.notdef`) and glyph ids beyond `maxp.numGlyphs` as missing.
2. **Complex layout.** Spans containing a script in `COMPLEX_LAYOUT_SCRIPTS`
   (Arabic, Hebrew, Syriac, Thaana, N'Ko, the Indic scripts, Thai, Lao, Khmer,
   Myanmar, Tibetan, Mongolian, and Ethiopic) or an unrecognised script need a
   font whose layout engine shapes text. Pillow shapes, reorders right-to-left
   runs, and positions marks only with its Raqm engine, which needs HarfBuzz
   and FriBiDi at runtime. Check with
   `PIL.features.check_feature("raqm")`; without Raqm these spans are refused.

Latin, Greek, Cyrillic, Armenian, Georgian, Han, kana, and Hangul spans need
coverage only.

## Supplying fonts

OpenMed does not bundle fonts. Pass font files you are licensed to use as
`BurnedInFont` entries with an SPDX license identifier from
`PERMISSIVE_FONT_LICENSES` (the permissive SPDX allowlist plus `OFL-1.1` and
its RFN variants, `Bitstream-Vera`, and `CC0-1.0`). Other identifiers raise
`BurnedInFontError` with reason `font_license_not_permissive`.

```python
from openmed.training.synthetic.burned_in import BurnedInFont, BurnedInGenerator

latin = BurnedInFont("/fonts/NotoSans-Regular.ttf", license="OFL-1.1")
thai = BurnedInFont("/fonts/NotoSansThai-Regular.ttf", license="OFL-1.1")

generator = BurnedInGenerator(
    seed=3,
    font_names=(latin,),            # pool for dates, MRNs, and other scripts
    script_fonts={"Thai": thai},    # used for spans that contain Thai
)
example = generator.generate("th")
```

Each span's font is chosen in this order: an exact per-call `font_names`
control, then the first `script_fonts` entry matching a script in the span
(keys come from `FONT_SCRIPTS`), then the constructor's `font_names` pool.
Plain font names and paths are still accepted in `font_names`; they are
recorded with license `NOASSERTION`, and the default DejaVu names with their
declared `Bitstream-Vera` license. Use `BurnedInFont(..., index=n)` to select
a face inside a `.ttc` collection.

## Missing fonts and opt-in fallback

A font that cannot be loaded raises `BurnedInFontUnavailableError`. It is
never replaced silently. With `font_fallback=True`, a missing default DejaVu
font is replaced by another installed default or, as a last resort, by
Pillow's embedded Aileron Regular (`PillowDefault`, `CC0-1.0`). The
replacement is still coverage-checked, so it can render basic Latin text but
refuses scripts it cannot draw. The `generate_burned_in_example` helper
accepts the same `script_fonts` and `font_fallback` arguments.

## Provenance in metadata

Each `BurnedInTextBox.metadata` records `font_sha256` (SHA-256 of the font
file bytes that were rendered), `font_license`, `font_fallback`,
`layout_engine` (`raqm`, `basic`, or `bitmap`), and the span's `scripts`.
`font_name` is the file name only, never the directory. The example metadata
adds `font_fallback_allowed`, `font_fallback_used`, and `fonts`, one
provenance record per distinct font in first-use order. The same seed, font
digest, and Pillow build produce byte-identical images.

## Errors

All errors subclass `BurnedInFontError`, itself a `ValueError`. Messages and
attributes carry only a reason code, script name, canonical label, counts,
font digest, or an opaque `font_ref`; never span text, code points, or paths.

| Error | `reason` | Raised when |
| --- | --- | --- |
| `BurnedInFontUnavailableError` | `font_unavailable`, `fallback_font_unavailable` | A font cannot be loaded and no usable fallback was requested. |
| `BurnedInGlyphCoverageError` | `glyph_coverage_missing` | The selected font lacks glyphs for a span; `script`, `font_sha256`, and `missing_count` identify the gap. |
| `BurnedInShapingUnavailableError` | `complex_layout_unavailable` | A complex-script span would be drawn without a shaping layout engine. |
| `BurnedInFontError` | `font_unparseable`, `font_license_not_permissive` | The font bytes are not a readable sfnt face, or a declared license is not allowed. |

## Test hooks

Tests and custom pipelines can inject `character_map_provider` (called with
font bytes and face index, returning a container of covered code points) and
`layout_capability_provider` (called with a loaded Pillow font, returning
whether it shapes). The defaults are `read_font_character_map` and
`font_supports_complex_layout`.

## Known limits

- Script detection is a block-level approximation of the Unicode Script
  property, conservative for punctuation inside script blocks.
- Conjoining Hangul jamo are treated like precomposed syllables and are not
  shaping-checked.
- Coverage proves glyphs exist, not that a font's shaping rules are complete
  for every conjunct; review samples for new script and font pairs.
- Character-map subtable formats 0, 4, 6, and 12 are decoded. A font whose
  selected subtable uses another format is treated as covering nothing.
- Declared licenses are recorded as stated by the caller, not verified
  against the font file.
