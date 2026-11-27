# medspaCy ConText Adapter

OpenMed's medspaCy adapter transfers assertion context from a processed `Doc`
to OpenMed spans. It creates no pipeline, loads no model, and downloads no
resources. Configure the pipeline yourself and pass its `Doc` to a converter.

Conversions work with Python 3.11+. The optional medspaCy runtime extra needs
Python 3.12+ because supported releases require a newer spaCy security baseline.

Install the extra in an environment that uses Python 3.12 or newer:

```bash
pip install "openmed[medspacy]"
```

## Choose a function

- `to_canonical(doc, *, text=None, openmed_spans=None,
  existing_spans=None, spans=None, config=None)` converts medspaCy entities
  into `PIIEntity` objects when no OpenMed spans are supplied. If spans are
  supplied, it copies them and enriches only exact-offset matches. Use one of
  `openmed_spans`, `existing_spans`, or `spans`; supplying multiple is an error.
- `attach_context(doc, openmed_spans, *, text=None, config=None)` is the
  explicit form for enriching existing OpenMed spans. Unmatched spans are
  returned as copies without added context.
- `process_to_canonical(text, *, nlp, openmed_spans=None, config=None)` runs
  the caller-provided `nlp` pipeline, then applies the same conversion as
  `to_canonical`. The pipeline must be callable and already configured.

For all three functions, `config` is an optional
`MedspacyContextAdapterConfig`: `source` labels metadata, `default_confidence`
sets confidence for rule-based annotations, and `preserve_context_flags`
controls whether source booleans are retained. Canonical context fields are
always written.

`text` on `to_canonical` and `attach_context` is optional but must match
`doc.text` when provided. `process_to_canonical` passes its first `text`
argument to the caller's pipeline.

## Exact-offset alignment

OpenMed spans receive context only when their half-open `(start, end)` offsets
exactly match a medspaCy span from the same, unmodified `Doc.text`. There is no
token, normalized-text, overlap, or approximate-position matching, preventing
context from attaching to a neighboring or partial entity.

In `No pneumonia was found.`, `pneumonia` matches only the same character
range in the processed document. A span starting one character later does not
match. After editing or normalizing text, use its exact offsets or process it
again.

## Metadata shape

Matched spans receive a `clinical_context` mapping with four canonical fields:

```python
{
    "negation": "negated",       # or "affirmed"
    "uncertainty": "certain",   # or "uncertain"
    "temporality": "recent",    # or "historical"
    "experiencer": "patient",   # or "family"
}
```

Metadata also records `adapter`, `source`, `clinical_context_sources`, and
`medspacy_offset_alignment: "exact"`. By default, `medspacy_context` retains
`is_negated`, `is_uncertain`, `is_historical`, and `is_family`. Set
`preserve_context_flags=False` to omit those booleans; canonical fields remain.

Without supplied OpenMed spans, each medspaCy entity becomes a `PIIEntity`.
With existing spans, supported types are preserved, inputs are not mutated,
and existing metadata is copied before context fields are added.

## Synthetic example

This example uses a short synthetic sentence and a caller-configured pipeline.
It does not load a language model or send text to a service.

```python
import medspacy
from medspacy.ner import TargetRule

from openmed.core.pii import PIIEntity
from openmed.interop.medspacy_context import attach_context

text = "No pneumonia was found."
nlp = medspacy.load()
nlp.get_pipe("medspacy_target_matcher").add(
    [TargetRule("pneumonia", "PROBLEM")]
)
doc = nlp(text)

start = text.index("pneumonia")
span = PIIEntity(
    text="pneumonia",
    label="PROBLEM",
    entity_type="PROBLEM",
    confidence=1.0,
    start=start,
    end=start + len("pneumonia"),
)
enriched = attach_context(doc, [span], text=text)
print(enriched[0].metadata["clinical_context"])
```

The adapter uses flags from the pipeline; it does not infer or change them.

## Runtime and privacy boundaries

`to_canonical` and `attach_context` accept a processed document without
importing medspaCy or spaCy. `process_to_canonical` imports the optional
packages only when called and invokes only your pipeline. OpenMed creates no
default pipeline, model, or network call.

Use synthetic text in examples and tests. Keep real clinical text in your
approved local environment and follow your organization's data policy.
Pipeline failures omit the original exception message.

Existing offline tests in `tests/unit/interop/test_medspacy_context_adapter.py`
cover metadata, exact matches, unmatched spans, and lazy imports.
