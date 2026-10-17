# Locale PHI notes

`LocalePhiGenerator` writes a short, one-paragraph clinical note for a chosen
language, containing a person name, date of birth, identifier, phone number,
street address, and a follow-up date. Each value is synthetic and comes with a
gold span whose offsets point at the exact characters in the note. The notes
are meant for augmenting de-identification training data, not for evaluating
it. See the [overview](index.md) for the shared conventions.

## Generate examples

```python
from openmed.training.synthetic import generate_locale_phi_examples

examples = generate_locale_phi_examples(languages=["en", "fr"], seed=7)
for example in examples:
    print(example.language, example.locale)
    print(example.text)
    for span in example.gold_spans:
        assert example.text[span.start:span.end] == span.text
        print("  ", span.label, span.start, span.end, span.text)
```

The first example is an English (`en_US`) note; its `PERSON`,
`DATE_OF_BIRTH`, `ID_NUM`, `PHONE`, `STREET_ADDRESS`, and `DATE` spans each
satisfy `text[start:end] == span.text`. `generate_locale_phi_examples` takes
keyword-only `languages` and `seed`; with no `languages` it covers every
supported language.

For a stateful generator that yields one example per call, use the class:

```python
from openmed.training.synthetic import LocalePhiGenerator

generator = LocalePhiGenerator(seed=7)
english = generator.generate("en")
everything = generator.generate_all()   # one example per supported language
```

## Training items

`LocalePhiExample.to_training_item()` returns a plain dictionary ready for a
JSONL writer:

```python
item = english.to_training_item()
print(sorted(item))
# ['is_synthetic', 'labels', 'language', 'locale', 'metadata', 'synthetic_source', 'text']
print(item["synthetic_source"], item["metadata"]["contains_real_phi"])
# locale_phi False
```

`labels` is the list of span dictionaries (`start`, `end`, `label`, `text`,
`metadata`). The identifier span's metadata names the identifier method and
the validator that accepted the value (for example
`pii_i18n.validate_french_nir` for French), so you can audit which checksum
rules were applied.

## Labels and languages

- `LOCALE_PHI_LABELS` lists the six labels: `PERSON`, `DATE_OF_BIRTH`,
  `ID_NUM`, `PHONE`, `STREET_ADDRESS`, `DATE`.
- `SUPPORTED_LOCALE_PHI_LANGUAGES` lists the 42 language codes the generator
  accepts. Any other code raises `ValueError` and the message lists the
  supported codes.

```python
from openmed.training.synthetic import LocalePhiGenerator

try:
    LocalePhiGenerator().generate("xx")
except ValueError as error:
    print(error)
```

## Seeds and determinism

With an integer `seed`, every field is derived from a hash of the seed, the
language, and the field name, so adding or removing a language from
`languages` does not change the other languages' output:

```python
a = LocalePhiGenerator(seed=7).generate("en")
b = LocalePhiGenerator(seed=7).generate("en")
assert a == b
```

With `seed=None` the values are drawn from fresh randomness and the result is
not reproducible. `metadata["seed"]` records which of the two was used.

## Known limits

- **Dates are Gregorian and ASCII.** Dates of birth and follow-up dates are
  rendered as ISO `YYYY-MM-DD` digits in every language, including languages
  that normally use other numerals or calendars. A model trained only on this
  data has not seen locale-native date writing.
- **Faker-dependent values.** Names, phone numbers, and street addresses come
  from Faker locale providers, so they depend on the installed Faker version
  and on how complete that provider is for the language. Identifiers use the
  repository's checksum validators where one exists.
- **Borrowed Faker locales.** Languages without a native Faker locale (for
  example `as`, `kn`, `ml`, `mr`, `pa`, `te`, `am`, `ur`, `xh`) emit a
  `UserWarning` and take names, phones, and addresses from a neighboring
  locale such as `en_IN` or `hi_IN`. Those values are valid synthetic data but
  are not written in the target language's own conventions.
- **One fixed template per language.** Sentence structure is fixed, so the
  notes teach value formats, not varied clinical prose.
- **Not clinical ground truth.** The note content is a fictional patient
  record. `metadata` always carries `augmentation_only: True` and
  `contains_real_phi: False`.
