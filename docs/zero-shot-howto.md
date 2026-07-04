# Zero-shot NER How-to

This guide shows a short zero-shot NER workflow with custom labels, synthetic
clinical text, and returned span inspection. For the full API reference, see
[Zero-shot Toolkit](./zero-shot-ner.md).

## Before you start

Install the GLiNER optional dependencies:

```bash
uv pip install -e ".[gliner]"
```

The inference examples require a local zero-shot model entry in
`models/index.json`. Replace `gliner-biomed-tiny` with a model identifier from
your local index if needed.

All example text below is synthetic and does not contain real patient data.

## Discover domains and default labels

Use the label helpers to list available domains and inspect a domain's default
labels:

```python
from openmed.ner import available_domains, get_default_labels

print(available_domains())
print(get_default_labels("biomedical"))
```

Default labels are useful when you want the packaged domain map to drive
extraction. For a focused task, pass custom labels directly in the request.

## Run extraction with custom labels

Define a small label set and run inference over a synthetic clinical sentence:

```python
from openmed.ner import NerRequest, infer

text = "Patient A was prescribed metformin 500 mg for type 2 diabetes."

request = NerRequest(
    model_id="gliner-biomed-tiny",
    text=text,
    labels=["Medication", "Dosage", "Condition"],
    threshold=0.5,
)

response = infer(request)
```

Inspect the returned entities to see each detected span, character offsets, and
confidence score:

```python
for entity in response.entities:
    print(
        entity.label,
        entity.text,
        entity.start,
        entity.end,
        f"{entity.score:.3f}",
    )
```

The `start` and `end` offsets point back into the input text, which makes the
spans easy to highlight or audit without storing raw identifiers elsewhere.

## Compare thresholds

Run the same sentence with different confidence thresholds to see how filtering
changes the output:

```python
from openmed.ner import NerRequest, infer

text = "Patient A was prescribed metformin 500 mg for type 2 diabetes."
labels = ["Medication", "Dosage", "Condition"]

for threshold in (0.35, 0.70):
    response = infer(
        NerRequest(
            model_id="gliner-biomed-tiny",
            text=text,
            labels=labels,
            threshold=threshold,
        )
    )

    print(f"\nThreshold: {threshold}")
    for entity in response.entities:
        print(entity.label, entity.text, f"{entity.score:.3f}")
```

A lower threshold can return more spans, including weaker matches. A higher
threshold usually returns fewer spans with stronger confidence scores.

## Learn more

- [Zero-shot Toolkit](./zero-shot-ner.md) covers indexing, domain defaults,
  inference APIs, and token-classification conversion.
- [ZeroShot NER Tour notebook][zero-shot-tour-notebook] walks through the
  broader zero-shot workflow in a notebook.

[zero-shot-tour-notebook]: https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/ZeroShot_NER_Tour.ipynb
