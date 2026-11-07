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
keyword-only `languages`, `seed` and optional `date_profiles`; with no
`languages` it retains the default model-backed language set.

For a stateful generator that yields one example per call, use the class:

```python
from openmed.training.synthetic import LocalePhiGenerator

generator = LocalePhiGenerator(seed=7)
english = generator.generate("en")
everything = generator.generate_all()   # the default model-backed set
```

The six explicit augmentation-only routes (`ha`, `ig`, `yo`, `rw`, `tl`,
`ne`) are also available through `generate(code)` or an explicit `languages`
list. Use `generator.generate_all(include_augmentation_only=True)` to append
them to the default set. Their examples record `bundled_model: False` and
`language_route`; synthetic generation does not add bundled models or
graduate language support.

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
- `SUPPORTED_LOCALE_PHI_LANGUAGES` lists the default model-backed template
  set. `AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES` in
  `openmed.training.synthetic.locale_phi` lists the six explicit extra routes.
  Codes outside both sets raise `ValueError` with the available code lists.

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

- **Default dates are Gregorian and ASCII.** Without `date_profiles=`, dates
  of birth and follow-up dates stay ISO `YYYY-MM-DD` in every language. Opt-in
  native-digit and native-calendar profiles are documented in the
  [language guide](../../languages.md). Profiles carry value-free calendar,
  numeral and detection-gap metadata; generating them does not establish
  detector coverage or clinical validity. For example,
  `LocalePhiGenerator(seed=7, date_profiles={"fa": "fa_solar_hijri_month_name"})`
  opts Persian examples into that profile while other languages keep defaults.
- **Faker-dependent values.** Names, phone numbers, and street addresses come
  from Faker locale providers, so they depend on the installed Faker version
  and on how complete that provider is for the language. Identifiers use the
  repository's checksum validators where one exists.
  `metadata["surrogate_providers"]` records the provider stamp for later drift
  checks; reproducibility assumes the same providers and date profiles.
- **Approximated Faker fields.** Some language packs use a neighboring Faker
  backend for fields such as phones or generic addresses and emit a
  `UserWarning`. Script-specific providers can still generate native names
  and places; the warning does not mean every field is borrowed. Consult
  [language-pack declarations](../../language-packs.md) for the provider and
  approximation boundaries.
- **One fixed template per language.** Sentence structure is fixed, so the
  notes teach value formats, not varied clinical prose.
- **Not clinical ground truth.** The note content is a fictional patient
  record. `metadata` always carries `augmentation_only: True` and
  `contains_real_phi: False`.
