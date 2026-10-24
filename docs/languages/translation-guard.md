# Clinical translation guard

Back-translation augmentation can return fluent text that quietly changes a
dose, drops a negation, flips laterality, expands an abbreviation, or rewrites a
de-identification placeholder. `openmed.training.languages.translation_guard`
wraps a translator you supply, protects those facts during translation, checks
them afterwards, and rejects any example that changed, with a reason code.

The guard does not translate anything itself. It calls your local translator
once per translation leg and never opens a network connection. Frozen items
are replaced with opaque placeholders before the call, so the translator never
sees identifiers, doses, PHI placeholders, or annotated spans. Results keep the
translated text for your training pipeline. Reports carry counts, reason codes,
labels, and offsets only. Translator exception messages are discarded because
they can quote input text.

## Wrap a local translator

Pass an object with `translate(text, source_lang, target_lang)` or a plain
callable with the same arguments. The example uses the offline dictionary
backend from the translation augmentation module.

```python
from openmed.training.languages.translation_guard import (
    TranslationGuard,
    TranslationGuardConfig,
)
from openmed.training.synthetic.translation_augment import DictionaryTranslator

translator = DictionaryTranslator(
    lexicon={
        ("en", "es"): (("denies", "niega"), ("no", "sin"), ("fever", "fiebre")),
        ("es", "en"): (("niega", "denies"), ("sin", "no"), ("fiebre", "fever")),
    }
)
guard = TranslationGuard(
    translator,
    config=TranslationGuardConfig(glossary={("en", "es"): {"metformin": "metformina"}}),
)

text = "Patient [NAME] denies chest pain. Takes metformin 500 mg BID. No fever."
spans = [{"start": 40, "end": 49, "label": "DRUG"}]
result = guard.round_trip(text, source_language="en", pivot_language="es", spans=spans)

if result.accepted:
    augmented_text, augmented_spans = result.text, result.spans
else:
    print(result.to_report()["reasons"])
```

`translate()` runs one leg, `round_trip()` runs source to pivot to source, and
`guard_examples()` (or `guard_round_trip_examples()`) gates a batch of
span-annotated examples. A rejected result has `text=None` and no spans.

## What is protected

Protected items are detected in priority order. An item never overlaps an
earlier one.

| Kind | Examples | Source |
| --- | --- | --- |
| `annotated_span` | caller spans, always protected | your NER labels |
| `phi_placeholder` | `[NAME]`, `<<OPENMED_PHI_...>>`, `{{DATE}}`, `***` | OpenMed mask and proxy placeholder shapes |
| `clinical_term` | `protected_terms` and glossary terms | your configuration |
| `measurement` | `500 mg`, `128/82 mmHg`, `37.5 °C`, `7.2%` | `openmed.clinical.units` and the unit aliases in `openmed.clinical.lexicons` |
| `identifier` | `MRN-0042817`, `HbA1c`, `q8h`, `COVID-19` | mixed letter-digit tokens |
| `number` | `3`, `2024-03-01`, `120/80` | digit groups and ranges |
| `abbreviation` | `BID`, `BP`, `b/p`, `bid`, `temp` | upper-case tokens, the `clinical_norm` abbreviation keys, and sig frequencies from `medication_sig` |

Glossary entries are terminology constraints. A term or annotated span that
matches a glossary entry for the `(source, target)` pair is restored as the
target term, for example `metformin` as `metformina` in the pivot text. On the
way back every item is restored as its original source surface, so the
back-translated example keeps the exact clinical facts. Everything else is
restored verbatim.

Disable detected kinds with `protected_kinds` when you want the translator to
localize them. Digits that are not protected are still verified.

For scripts written without spaces (Chinese and Japanese), detection treats
only ASCII letters and digits as word characters. The measurement `37.5°C`
in `体温37.5°C` and an identifier such as `MRN-77` are found inside running
text, and localized unit aliases such as `毫米汞柱` count as measurement units.

## What is verified

| Check | When | Reason code |
| --- | --- | --- |
| Each placeholder appears exactly once | every leg | `protected_item_dropped`, `protected_item_duplicated`, `unknown_placeholder` |
| No split or re-cased placeholder fragments remain | every leg | `placeholder_residue` |
| Digits outside protected items survive as a multiset | every leg | `number_changed` |
| Negation cue counts match across languages | forward leg and one-way translation | `negation_changed` |
| Negation, uncertainty, hypothetical, and historical cue counts match | same-language output and end of a round trip | `*_changed` for each axis |
| Severity, laterality, and position modifiers match | English same-language output and English round trips | `severity_changed`, `laterality_changed`, `position_changed` |
| Projected spans point at their text | every accepted leg | `span_integrity_failed` |

Assertion cues come from the shared ConText language packs
(`openmed.clinical.lexicons.context_cues`). Pseudo-negation phrases such as
`no increase` are matched first so their inner `no` never counts as a negation.
Cues inside protected items are ignored on both sides. Set
`cross_lingual_assertion_axes` to add axes to the cross-language check, or to
`()` when a pivot language pack is too sparse to compare.

When a language has no cue pack, the check cannot run. For a one-way
translation, `fail_on_unverifiable=True` (the default) rejects it with
`assertion_unverifiable`. In a round trip, a pivot language without a pack only
adds a warning, because the end-to-end check runs in the source language. The
modifier lexicon is English only, so other source languages carry a
`modifier_unverifiable` warning. For cross-language laterality or anatomy,
protect the terms through annotated spans or the glossary.

Other reason codes are `translator_error` (exception or non-string output),
`empty_translation`, `placeholder_collision` (the source already contains the
placeholder shape, so the translator is not called), `invalid_input` and
`same_language` (batch only).

## Span map

Every result has a span map of protected item offsets. A round trip records
`source`, `pivot`, and `back` offsets for each item. Accepted batch examples
keep the map in `to_training_item()` under
`metadata.translation_guard.span_map`, next to the projected `gold_spans`.
Projected spans carry `projected_from_start` and `projected_from_end`, the same
fields that `openmed.training.synthetic.offset_projection` writes.

## Gate a batch

```python
from openmed.training.languages.translation_guard import guard_round_trip_examples

batch = guard_round_trip_examples(
    examples,  # mappings with text, language, id, and gold_spans
    translator=translator,
    pivot_language="es",
)
rows = [example.to_training_item() for example in batch.accepted]
report = batch.report.to_dict()  # counts, codes, offsets; no text or ids
```

The report lists one decision per input position with its reason codes,
warnings, protected counts, and span map. `report.fingerprint` is a SHA-256
reference of the canonical JSON, so two runs with the same deterministic
translator produce the same fingerprint.

## Limits

The checks compare surface forms. They catch dropped, duplicated, or altered
protected items and changed cue counts. They do not prove that a translation
is clinically equivalent, and an accepted example can still read badly. The
guard is training-data tooling, not a compliance certification, and it must not
drive clinical decisions.

Run the offline tests with:

```bash
.venv/bin/python -m pytest tests/unit/training/languages/test_translation_guard.py -q
```
