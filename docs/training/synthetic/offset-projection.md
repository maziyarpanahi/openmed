# Span offset projection

Translation augmentation moves entity spans from a source text to a translated
text. The offset projection helpers in `openmed.training.synthetic` do the
bookkeeping: they normalize span records, check that offsets still point at
the right characters, and re-locate spans in translated text. If a span cannot
be recovered, they raise rather than guess, so the caller can drop the example
instead of emitting a noisy label. See the [overview](index.md) for shared
conventions and [translation augmentation](translation-augmentation.md) for
the pipeline that uses them.

## Normalize spans

`normalize_span_annotations` accepts a record (a mapping with a span list
under `gold_spans`, `labels`, `entities`, or `spans`) or a plain iterable of
span mappings, and returns `SpanAnnotation` rows (`start`, `end`, `label`,
`text`, `metadata`):

```python
from openmed.training.synthetic import normalize_span_annotations

record = {
    "text": "Patient reports fever and cough.",
    "labels": [
        {"start": 16, "end": 21, "label": "SYMPTOM"},
        {"start": 26, "end": 31, "label": "SYMPTOM"},
    ],
}
spans = normalize_span_annotations(record)
print([(s.label, s.text) for s in spans])
# [('SYMPTOM', 'fever'), ('SYMPTOM', 'cough')]
```

A missing `text` on a span is filled from the record's text; an unusable
start or end, or a missing label, raises `SpanProjectionError`.

## Validate offsets

```python
from openmed.training.synthetic import (
    SpanAnnotation,
    SpanProjectionError,
    span_corruption_count,
    validate_span_integrity,
)

validate_span_integrity(record["text"], spans)           # passes silently
bad = [SpanAnnotation(start=0, end=3, label="X", text="Zzz")]
print(span_corruption_count("Patient", bad))            # 1
try:
    validate_span_integrity("Patient", bad)
except SpanProjectionError as error:
    print(error)                                         # span text mismatch for X: 0:3
```

`validate_span_integrity` raises for out-of-range, empty, mismatched, or (by
default) overlapping spans; pass `allow_overlap=True` to permit overlaps.
`span_corruption_count` is the non-raising variant: it returns how many spans
no longer match their text, which is useful as a dataset quality metric.

## Realign spans in translated text

`realign_translated_spans` translates each source span on its own with the
supplied translator, then finds the result in the already translated text,
left to right, and only on token boundaries:

```python
from openmed.training.synthetic import realign_translated_spans


class LexiconTranslator:
    def translate(self, text, source_lang, target_lang):
        return {"fever": "Fieber", "cough": "Husten"}.get(text, text)


projected = realign_translated_spans(
    source_text=record["text"],
    translated_text="Patient berichtet Fieber und Husten.",
    source_spans=spans,
    translator=LexiconTranslator(),
    source_language="en",
    target_language="de",
)
print([(s.start, s.end, s.text) for s in projected])
# [(18, 24, 'Fieber'), (29, 35, 'Husten')]
print(dict(projected[0].metadata))
# {'projected_from_end': 21, 'projected_from_start': 16}
```

Any translator with a `translate(text, source_lang, target_lang)` method
works (the `SpanTranslator` protocol). If a span cannot be found, the call
raises `SpanProjectionError`:

```python
try:
    realign_translated_spans(
        source_text=record["text"],
        translated_text="Patient berichtet nichts.",
        source_spans=spans,
        translator=LexiconTranslator(),
        source_language="en",
        target_language="de",
    )
except SpanProjectionError as error:
    print(error)    # could not realign SYMPTOM span 16:21
```

## Known limits

- Matching is by exact translated string with word-boundary checks. A
  translator that inflects a span differently when translated alone than in
  context (common in morphologically rich languages) makes the span
  unrecoverable, and the example is dropped.
- Spans are located in order; reordering by the translator can cause a miss.
- Offsets are Python string indices (code points), not bytes or grapheme
  clusters.
- These helpers check offsets, not translation quality or label correctness.
