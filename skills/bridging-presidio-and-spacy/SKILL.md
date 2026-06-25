---
name: bridging-presidio-and-spacy
description: "Combine OpenMed clinical NLP with Microsoft Presidio, spaCy, or LangChain through OpenMed's built-in interop adapter registry (openmed.interop). Covers the lazy adapter registry (available_adapters, get_adapter, adapter_spec), the presidio/spacy/langchain pip extras, and the verified callables — Presidio to_canonical/from_canonical/merge_with_openmed, the spaCy openmed_deid pipeline factory, and the LangChain create_redaction_runnable. Use when the user wants to add Presidio recognizers, embed OpenMed PII detection in a spaCy pipeline, or use OpenMed de-identification as a LangChain runnable. Pairs adjacent to the OpenMed PII skills."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: adjacent
  version: "1.0"
---

# Bridging Presidio, spaCy & LangChain

OpenMed interoperates with the dominant PII/NLP ecosystems through a single,
**lazy** adapter registry: `openmed.interop`. Adapters live behind explicit
imports, so importing `openmed` never drags in Presidio, spaCy, or LangChain —
each is an optional extra you install only when you need that bridge.

## When to use

Reach for a bridge when:

- you already run **Microsoft Presidio** and want OpenMed's clinical PII recall
  on top (or to feed OpenMed spans back into Presidio's anonymizer);
- you have a **spaCy** pipeline and want OpenMed PII spans on the `Doc`;
- you build **LangChain** chains and want to redact PHI *before* text reaches an
  LLM (the on-device guardrail in front of a cloud model);
- you need OpenMed's de-identification reachable from an existing framework
  instead of rewriting the pipeline around `openmed.deidentify`.

## The lazy adapter registry (verified)

```python
import openmed.interop as interop

interop.available_adapters()
# ('cda', 'hl7v2', 'langchain', 'presidio', 'spacy')

spec = interop.adapter_spec("presidio")
# AdapterSpec(name='presidio', module='openmed.interop.presidio',
#             extra='presidio', description='Presidio RecognizerResult adapter')

mod = interop.get_adapter("presidio")        # imports openmed.interop.presidio
# Attribute access also works lazily:
openmed.interop.presidio                      # same module, imported on first touch
```

`available_adapters()` and `adapter_spec()` never import the adapter module, so
they are safe to call for discovery even without the extra installed.
`get_adapter(name)` (and attribute access) triggers the import — and the
adapter's own optional dependency.

Install only the extra you need:

```bash
pip install "openmed[presidio]"     # Presidio RecognizerResult adapter
pip install "openmed[spacy]"        # spaCy openmed_deid component
pip install "openmed[langchain]"    # LangChain redaction runnable
# cda and hl7v2 adapters ship in core (no extra) — see their own skills
```

## Presidio bridge (verified callables)

Module `openmed.interop.presidio` converts between Presidio
`RecognizerResult`s and OpenMed canonical `PIIEntity`s, and merges both
detectors through OpenMed's semantic-unit merger.

```python
from openmed.interop.presidio import (
    to_canonical,        # RecognizerResult(s) -> [PIIEntity]
    from_canonical,      # [PIIEntity] -> [RecognizerResult]  (needs presidio extra)
    merge_with_openmed,  # combine OpenMed + Presidio spans, resolve overlaps
    PresidioAdapterConfig,
)
import openmed

text = "Dr. Smith called patient at 617-555-0123 on 2024-03-02."

# Presidio gives you RecognizerResults; OpenMed gives PIIEntities.
openmed_spans = openmed.extract_pii(text).entities
presidio_results = analyzer.analyze(text=text, language="en")   # your Presidio analyzer

merged = merge_with_openmed(
    openmed_spans, presidio_results, text=text,
    config=PresidioAdapterConfig(preserve_presidio_labels=True),
)
# -> de-duplicated [PIIEntity]; overlaps resolved by score, length, OpenMed-origin
```

Why merge instead of union: `merge_with_openmed` runs both detectors' spans
through `merge_entities_with_semantic_units`, so overlapping/adjacent detections
collapse into one correct span (e.g. `PHONE` from Presidio vs a partial OpenMed
hit) rather than producing double redactions. Label mapping is built in
(Presidio `PHONE_NUMBER` ↔ OpenMed `PHONE`, `US_SSN` ↔ `SSN`, etc.).

To push OpenMed spans into Presidio's **anonymizer**, convert back:

```python
results = from_canonical(openmed_spans)        # [RecognizerResult]
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
```

## spaCy bridge (verified factory)

Module `openmed.interop.spacy_component` registers a spaCy pipeline factory
named **`openmed_deid`**. Add it to a pipeline and OpenMed PII spans land on the
`Doc`.

```python
import spacy
import openmed.interop.spacy_component   # registers the @Language.factory

nlp = spacy.blank("en")
nlp.add_pipe("openmed_deid", config={
    "confidence_threshold": 0.5,
    "lang": "en",
    "target": "openmed_pii",     # doc.spans key
    "merge_ents": False,         # set True to also write doc.ents
    "alignment_mode": "expand",  # char->token alignment: strict|contract|expand
})

doc = nlp("Patient John Doe, MRN 12345, seen today.")
for span in doc.spans["openmed_pii"]:
    print(span.label_, span.text)
# raw char-offset spans also available on doc._.openmed_pii
```

`merge_ents=True` writes the spans into `doc.ents`, resolving overlaps with
spaCy's `filter_spans`. Use `OpenMedDeidComponent` / `OpenMedDeidConfig`
directly if you construct the component outside `add_pipe`.

## LangChain bridge (verified runnable)

Module `openmed.interop.langchain` exposes a `Runnable`-shaped redactor you drop
*in front of* an LLM step so PHI never leaves the device.

```python
from openmed.interop.langchain import (
    create_redaction_runnable, LangChainRedactionConfig,
)

redactor = create_redaction_runnable(
    config=LangChainRedactionConfig(method="mask", policy="hipaa_safe_harbor"),
    input_key="text",        # redact this key in a dict payload (optional)
    output_key="text",
)

chain = redactor | prompt | llm          # redact -> prompt -> model
chain.invoke({"text": "John Doe, MRN 12345, has type 2 diabetes."})
```

The transform redacts strings, LangChain `Document`s (`page_content`), lists,
tuples, and mapping payloads. Use `create_redaction_transform(...)` for the
dependency-light object (no `langchain-core` needed) and `.as_runnable()` when
you want the `RunnableLambda`. `LangChainRedactionConfig` forwards the full
`openmed.deidentify` surface (`method`, `policy`, `confidence_threshold`,
`keep_year`, `consistent`, `lang`, ...).

## Hand-off to / from OpenMed

- **Into OpenMed:** Presidio `RecognizerResult`s and (implicitly) spaCy text
  become OpenMed `PIIEntity`s via the adapters; from there use the normal
  OpenMed de-id/audit/policy skills.
- **Out of OpenMed:** `from_canonical` → Presidio anonymizer; the spaCy
  component → downstream spaCy components; the LangChain runnable → any chain.
- The canonical object everywhere is `openmed.core.pii.PIIEntity`
  (`text`, `label`, `confidence`, `start`, `end`, `entity_type`, `metadata`).

## Edge cases & gotchas

- **Discovery is free; import is not.** Call `available_adapters()` /
  `adapter_spec()` to probe without installing the extra. Touching the module
  (`get_adapter`/attribute access) raises a clear `ImportError` telling you the
  extra to install if it is missing.
- **Offsets must match the same text.** `merge_with_openmed` and the spaCy
  alignment both assume all spans index the *same* string. De-identify or
  normalise once, up front; do not mix offsets from pre- and post-normalised
  text.
- **`alignment_mode="expand"`** (spaCy default here) snaps char spans out to
  token boundaries; use `"strict"` if you need exact char alignment and accept
  dropped spans that do not align.
- **LangChain redaction is a guardrail, not a guarantee.** Gate de-id quality
  with `openmed.eval` leakage gates (`evaluating-with-leakage-gates`) before
  trusting it in front of a cloud LLM.
- **Local-first holds across bridges.** OpenMed inference stays on-device; only
  *your* downstream LLM/cloud step (if any) leaves the machine — which is
  exactly why you redact first.

## Standards & references

- Microsoft Presidio: https://microsoft.github.io/presidio/
- Presidio RecognizerResult: https://microsoft.github.io/presidio/api/analyzer_python/#presidio_analyzer.RecognizerResult
- spaCy custom pipeline components: https://spacy.io/usage/processing-pipelines#custom-components
- spaCy `Language.factory`: https://spacy.io/api/language#factory
- LangChain Runnable interface: https://python.langchain.com/docs/concepts/runnables/
