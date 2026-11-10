# Indic Unicode normalization

`IndicNormalizer` canonicalizes text in Devanagari, Bengali/Assamese, Tamil,
Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, and Odia before clinical model
inference. It is local, deterministic, and uses only Python's standard Unicode
support.

```python
from openmed.processing import IndicNormalizer

normalizer = IndicNormalizer()
canonical = normalizer.normalize("क़ासिम और अङ्क")
mapped = normalizer.normalize_with_offsets("ID: ൻ രോഗി")
raw_start, raw_end = mapped.remap_span(4, 5)
```

The default policy applies NFC, retains nuktas, strips ZWJ/ZWNJ encoding
variants, converts strict explicit-nasal clusters to anusvara, normalizes
chandrabindu/chandra variants, and keeps word endings unchanged. The following
options are available:

- `remove_nuktas=False`: nukta removal is opt-in because it can erase a
  phonemic distinction.
- `nasals_mode="to_anusvara"`: use `"preserve"`,
  `"to_anusvara_relaxed"`, or `"to_nasal_consonants"` when a downstream model
  expects another convention.
- `normalize_chandra=True`: set to `False` to retain chandra forms.
- `normalize_vowel_ending=False`: explicit vowel-ending insertion is opt-in.
- `joiner_policy="strip"`: use `"preserve"` when joiner distinctions must be
  retained outside PII inference.

`normalize_with_offsets()` returns one source range for every normalized code
point. A model span can therefore be mapped back to the original text without
storing text in audit metadata or slicing through a raw grapheme cluster.

## Rule and license provenance

The script-rule behavior is modeled on the Indic NLP Library normalizers by
Anoop Kunchukuttan, released under the MIT License. OpenMed independently
implements compact, data-only equivalents for nukta decomposition, nasal and
chandra normalization, two-part vowel signs, punctuation, Malayalam chillu,
Gurmukhi addak/tippi and vowel bases, and Odia vowel/va mappings. OpenMed does
not copy or bundle the upstream package, its resources, or any third-party
model weights.

The OpenMed implementation deliberately differs in two safety areas: defaults
never remove script-bearing characters, and every transformation carries raw
offset provenance for clinical redaction. The relevant upstream sources are:

- [Indic NLP Library normalizer](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/master/indicnlp/normalize/indic_normalize.py)
- [Indic NLP Library MIT license](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/master/LICENSE)
