# Translation augmentation

`augment_span_annotated_examples` turns span-annotated NER examples into
translated and round-trip back-translated copies whose spans are re-aligned to
the new text, using the checks on the
[offset projection](offset-projection.md) page. Everything runs locally: the
default translator is a small offline lexicon, and any model you bring runs in
your own process. See the [overview](index.md) for shared conventions.

## Augment examples

```python
from openmed.training.synthetic import augment_span_annotated_examples

seed = {
    "id": "demo-1",
    "text": "Patient reports fever and cough.",
    "language": "en",
    "labels": [
        {"start": 16, "end": 21, "label": "SYMPTOM"},
        {"start": 26, "end": 31, "label": "SYMPTOM"},
    ],
}
examples = augment_span_annotated_examples([seed], target_languages=["hi"])
for example in examples:
    print(example.transform, "|", example.text)
    print("  ", [(s.start, s.end, s.text) for s in example.gold_spans])
```

For this seed it yields two examples: a translation (`translation:en->hi`,
"रोगी बताता है बुखार और खांसी.") and a back-translation
(`backtranslation:en->hi->en`) with its spans re-aligned. Each seed needs an
id (a fallback is generated), `text`, a `language` and a span list under any
key `normalize_span_annotations` accepts. `target_languages` defaults to
`DEFAULT_TARGET_LANGUAGES` (`hi`, `te`).

Options:

- `include_translations` / `include_backtranslations` switch each output kind
  off.
- `drop_identical` (default `True`) drops outputs whose text equals the seed.
- `translator` replaces the default `DictionaryTranslator`.

Examples that cannot be fully re-aligned are dropped, never emitted with
approximate spans. Duplicates are removed.

## Held-out evaluation data

Pass `heldout_eval_ids` and `heldout_eval_texts` to keep evaluation data out
of training. A seed is skipped if its id or normalized text is held out, or if
its `metadata["split"]` (or `source_split`) is `dev`, `eval`, `heldout`,
`test`, or `validation`; generated texts that match held-out text are dropped
too.

```python
held_out = augment_span_annotated_examples(
    [seed], target_languages=["hi"], heldout_eval_texts=[seed["text"]]
)
print(len(held_out))   # 0
```

## Bring your own translator

Any object with `translate(text, source_lang, target_lang)` works.
`ModelBackedTranslator` wraps a callable you supply; you load the model and
the adapter adds no network behavior:

```python
from openmed.training.synthetic import ModelBackedTranslator

translator = ModelBackedTranslator(
    lambda text, source, target: text.replace("fever", "Fieber"),
    backend_name="stub",
)
examples = augment_span_annotated_examples(
    [seed], translator=translator, target_languages=["de"],
    include_backtranslations=False,
)
print(examples[0].text)                       # Patient reports Fieber and cough.
print([s.text for s in examples[0].gold_spans])  # ['Fieber', 'cough']
```

`DictionaryTranslator(lexicon=...)` takes a mapping from
`(source_lang, target_lang)` to `(source, target)` phrase pairs and does
longest-match replacement.

## Training items and files

```python
item = examples[0].to_training_item()
print(item["synthetic_source"], item["is_synthetic"], item["metadata"]["transform"])
```

Training items carry `id`, `text`, `language`, `labels`/`gold_spans`,
`source_id`, `source_language`, and metadata including `augmentation_only`,
`synthetic`, a `source_text_hash`, and the transform chain. `SYNTHETIC_SOURCE`
(`translation_backtranslation`) is the `synthetic_source` value.

```python
import tempfile
from pathlib import Path

from openmed.training.synthetic import load_span_jsonl, write_augmented_jsonl

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "augmented.jsonl"
    write_augmented_jsonl(examples, path)
    print(len(load_span_jsonl(path)))   # 1
```

`write_augmented_jsonl` writes sorted-key UTF-8 JSON, one example per line;
`load_span_jsonl` reads span JSONL (such as your seed data) back as dicts.

## Seeds and determinism

There is no randomness: the same inputs, translator, and options always give
the same examples and ids.

## Known limits

- The built-in lexicon is small and word-level. It leaves unknown words as
  is and cannot inflect or reorder, so output is not fluent. Back-translation
  can drift: the example above turns "Patient" into "case". Judge the data
  by its spans, not its prose.
- Spans are kept only when each one can be re-found in the translation; with
  morphologically rich languages many examples are dropped.
- Back-translation depends on the translator; with a lossy one the output may
  equal the seed (dropped by `drop_identical`) or drift.
- The held-out filter compares exact normalized text and hashes, not
  paraphrases, so near-duplicates of evaluation data are not caught.
- Outputs are augmentation data and inherit any label errors in the seeds.
